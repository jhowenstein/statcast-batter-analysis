import numpy as np
import pandas as pd
import scipy as sp

from scipy import signal

import os
import glob

import matplotlib.pyplot as plt


swing_events = ['hit_into_play','foul','swinging_strike','swinging_strike_blocked']
contact_events = ['hit_into_play','foul']
no_contact_swing_events = ['swinging_strike','swinging_strike_blocked']

batter_outcome_events = ['single', 'field_out', 'grounded_into_double_play',
       'strikeout', 'home_run', 'double', 'fielders_choice',
       'force_out', 'triple', 'field_error','double_play', 'sac_fly',
       'strikeout_double_play', 'fielders_choice_out']


class Batter:

    zone_half_width = 10.5 / 12
    sz_left = -zone_half_width
    sz_right = zone_half_width

    def __init__(self,name,data,process_games=True):
        self.name = name
        self.data = data

        self.games = []
        self.atBats = []

        self.analyze_pitch_decision()
        self.analyze_pitch_location()

        if process_games:
            self.process_games()

    def __str__(self):
        return self.name

    @property
    def sz_top(self):
        precision = 3
        mean_top = self.data['sz_top'].mean()
        return round(mean_top, precision)

    @property
    def sz_btm(self):
        precision = 3
        mean_btm = self.data['sz_bot'].mean()
        return round(mean_btm, precision)

    @property
    def sz_vert_step(self):
        return (self.sz_top - self.sz_btm) / 3

    @property
    def sz_horiz_step(self):
        return (self.sz_right - self.sz_left) / 3

    @property
    def sz_mid_btm(self):
        return self.sz_btm + self.sz_vert_step

    @property
    def sz_mid_top(self):
        return self.sz_btm + self.sz_vert_step * 2

    @property
    def sz_mid_left(self):
        return self.sz_left + self.sz_horiz_step

    @property
    def sz_mid_right(self):
        return self.sz_left + self.sz_horiz_step * 2

    @property
    def sz_center_height(self):
        return (self.sz_top + self.sz_btm) / 2

    @property
    def zone_half_height(self):
        return (self.sz_top - self.sz_btm) / 2

    @property
    def zone_height(self):
        return (self.sz_top - self.sz_btm)

    @property
    def zone_width(self):
        return 2 * self.zone_half_width

    @property
    def babip_count(self):
        babip_df = self.data[self.data['event']=='hit_into_play']
        return babip_df.shape[0]

    @property
    def PA(self):
        count = 0
        for game in self.games:
            count += len(game.atBats)
        return count

    @property
    def K(self):
        count = 0
        for game in self.games:
            for atBat in game.atBats:
                if atBat.isStrikeout:
                    count += 1
        return count

    @property
    def BB(self):
        count = 0
        for game in self.games:
            for atBat in game.atBats:
                if atBat.isWalk:
                    count += 1
        return count

    def add_game(self,gameID,data):
        self.games.append(Game(gameID,data))

    def add_atBat(self,number,data):
        # Note: If AtBat is called from the 'Batter' object, '{game_id}-{at_bat_number}' identifier should be used
        self.atBats.append(AtBat(number,data))

    def process_games(self):
        game_IDs = self.data['game_pk'].unique()
        for game_ID in game_IDs:
            game_data = self.data[self.data['game_pk']==game_ID]
            self.add_game(f'{game_ID}',game_data)
        self.sort_games()

    def process_atBats(self):
        game_IDs = self.data['game_pk'].unique()
        for game_ID in game_IDs:
            game_data = self.data[self.data['game_pk']==game_ID]

            atBat_numbers = game_data['at_bat_number'].unique()

            for atBat_number in atBat_numbers:
                atBat_data = game_data[game_data['at_bat_number']==atBat_number]

                atBat_tag = int(f'{game_ID}-{atBat_number}')
                self.add_atBat(atBat_tag,atBat_data)

    def sort_games(self):
        def sortFunc(game):
            return int(game.date.replace('-',''))

        self.games.sort(key=sortFunc)

    def filter_data(self, key, value_key=None, high_value=None, low_value=None, in_place=True):
        pass
        
    def filter_hit_into_play(self, in_place=True):
        _df = self.data[self.data['event']=='hit_into_play']

        if in_place:
            self.data = _df
        else:
            return _df

    def analyze_pitch_location(self):
        x_loc = self.data['plate_x'].values
        z_loc = self.data['plate_z'].values

        hand = self.data['stand'].values
        hand_normalization = -(hand == 'L').astype(int) + (hand == 'R').astype(int)
        #hand_normalization = np.ones(len(hand))

        norm_plate_x = x_loc * hand_normalization

        center = (0,(self.sz_btm + self.sz_top) / 2)

        dx = x_loc - center[0]
        dz = z_loc - center[1]

        dist_to_center = (dx**2 + dz**2)**.5 

        horiz_dist_to_edge = np.absolute(x_loc) - self.zone_half_width

        dist_to_top = z_loc - self.sz_top
        dist_to_btm = self.sz_btm - z_loc

        vert_dist_to_edge = np.maximum(dist_to_top,dist_to_btm)

        dist_to_left = self.sz_left - x_loc
        dist_to_right = x_loc - self.sz_right

        dist_to_inside = []
        dist_to_outside = []

        for l,r,_hand in zip(dist_to_left,dist_to_right,hand):
            if _hand == 'R':
                dist_to_inside.append(l)
                dist_to_outside.append(r)
            else:
                dist_to_inside.append(r)
                dist_to_outside.append(l)   

        dist_to_inside = np.array(dist_to_inside)
        dist_to_outside = np.array(dist_to_outside)

        dist_to_zone = []
        for I,O,T,B in zip(dist_to_inside,dist_to_outside,dist_to_top,dist_to_btm):
            if I > 0:
                if T > 0:
                    dist = (I**2 + T**2)**.5
                elif B > 0:
                    dist = (I**2 + B**2)**.5
                else:
                    dist = I
            elif O > 0:
                if T > 0:
                    dist = (O**2 + T**2)**.5
                elif B > 0:
                    dist = (O**2 + B**2)**.5
                else:
                    dist = O
            elif T > 0:
                dist = T
            elif B > 0:
                dist = B
            else:
                dist = max([I,O,T,B])
                
            dist_to_zone.append(dist)
            
        dist_to_zone = np.array(dist_to_zone)

        proportional_plate_x = norm_plate_x / self.zone_half_width
        proportional_plate_z = (z_loc - self.sz_center_height) / self.zone_half_height

        prop_dist_to_zone = []
        for px, pz in zip(proportional_plate_x, proportional_plate_z):
            if abs(px) > 1 and abs(pz) > 1:
                prop_dist = (px**2 + pz**2)**.5
            elif abs(px) > 1:
                prop_dist = abs(px)
            elif abs(pz) > 1:
                prop_dist = abs(pz)
            else:
                prop_dist = max(abs(px),abs(pz))
            
            prop_dist_to_zone.append(prop_dist)

        prop_dist_to_zone = np.array(prop_dist_to_zone)


        self.data['dist_to_center'] = dist_to_center
        self.data['horiz_dist_to_edge'] = horiz_dist_to_edge
        self.data['dist_to_top'] = dist_to_top
        self.data['dist_to_btm'] = dist_to_btm
        self.data['vert_dist_to_edge'] = vert_dist_to_edge
        self.data['dist_to_inside'] = dist_to_inside
        self.data['dist_to_outside'] = dist_to_outside
        self.data['dist_to_zone'] = dist_to_zone
        self.data['prop_plate_x'] = proportional_plate_x
        self.data['prop_plate_z'] = proportional_plate_z
        self.data['norm_plate_x'] = norm_plate_x
        self.data['prop_dist_to_zone'] = prop_dist_to_zone

    def filter_data_by_location(self,bottom_left,top_right,df=None,inplace=False):
        if df is None:
            df = self.data

        _df = df[(df['prop_plate_x'] >= bottom_left[0]) & (df['prop_plate_x'] <= top_right[0]) & (
            df['prop_plate_z'] >= bottom_left[1]) & (df['prop_plate_z'] <= top_right[1])]

        if inplace:
            df = _df
        else:
            return _df

    def filter_data_by_pitch_type(self,pitch_type,df=None,inplace=False):
        if df is None:
            df = self.data

        if isinstance(pitch_type,str):
            _df = df[df['pitch_type']==pitch_type]

        if isinstance(pitch_type,list):
            _dfs = []
            for _type in pitch_type:
                _df = df[df['pitch_type']==_type]
                _dfs.append(_df)
            _df = pd.concat(_dfs)

        if inplace:
            self.data = _df
        else:
            return _df

    def isStrike(self):
        self.data['isStrike'] = (self.data['plate_x'] >= self.sz_left) & (self.data['plate_x'] <= self.sz_right) & (
            self.data['plate_z'] >= self.sz_btm) & (self.data['plate_z'] <= self.sz_top)

    def isSwing(self):
        # swing_events = ['hit_into_play','foul','swinging_strike','swinging_strike_blocked']
        
        self.data['isSwing'] = (self.data['event']=='hit_into_play') | (self.data['event']=='foul') | (
            self.data['event']=='swinging_strike') | (self.data['event']=='swinging_strike_blocked')

    def isCorrectDecision(self):
        self.data['isCorrectDecision'] = self.data['isStrike'] == self.data['isSwing']

    def isContact(self):
        self.data['isContact'] = (self.data['event']=='hit_into_play') | (self.data['event']=='foul')

    def calculate_contact_rate(self,start=None,end=None,df=None,precision=3):
        if df is None:
            df = self.data

        nContact = sum(df['isContact'].values.astype(int))
        nSwing = sum(df['isSwing'].values.astype(int))

        contact_rate = nContact / nSwing

        return round(contact_rate, precision)

    def calculate_zone_contact_rate(self,start=None,end=None,df=None,precision=3):
        if df is None:
            df = self.data

        _df = df[df['isStrike']==True]

        contact_rate = self.calculate_contact_rate(start=start,end=end,df=_df,precision=precision)

        return contact_rate

    def calculate_outside_contact_rate(self,start=None,end=None,df=None,precision=3):
        if df is None:
            df = self.data

        _df = df[df['isStrike']==False]

        contact_rate = self.calculate_contact_rate(start=start,end=end,df=_df,precision=precision)

        return contact_rate

    def calculate_max_exit_velocity(self,start=None,end=None,df=None,precision=1,n=1):
        if df is None:
            df = self.data

        df = df[df['launch_speed'].notna()]
        exit_velos = df['launch_speed'].values
        exit_velos.sort()

        if len(exit_velos) > (n-1):
            max_ev = exit_velos[-n:].mean()
            return round(max_ev,precision)
        else:
            return None
        
    def calculate_average_exit_velocity(self,start=None,end=None,df=None,precision=1):
        if df is None:
            df = self.data

        df = df[df['launch_speed'].notna()]
        exit_velos = df['launch_speed'].values

        if len(exit_velos) > 0:
            avg_ev = exit_velos.mean()
            return round(avg_ev,precision)
        else:
            return None

    def calculate_average_launch_angle(self,start=None,end=None,df=None,precision=1):
        if df is None:
            df = self.data

        df = df[df['launch_angle'].notna()]
        launch_angles = df['launch_angle'].values

        if len(launch_angles) > 0:
            avg_la = launch_angles.mean()
            return round(avg_la,precision)
        else:
            return None


    def calculate_estimated_bat_speed(self,start=None,end=None,df=None,top_percentile=0.1,precision=1):
        if df is None:
            df = self.data

        e = 0.2

        batted_ball_df = df[(df['launch_speed'].notna()) & (df['launch_angle'].notna()) & (df['event']=='hit_into_play')]

        batted_ball_df = batted_ball_df.sort_values(by='launch_speed',ascending=False)

        total_babip_count = batted_ball_df.shape[0]

        top_babip_count = int(top_percentile * total_babip_count)

        if total_babip_count == 0:
            return None
        if top_babip_count == 0:
            top_babip_count = 1

        top_ev_mean = batted_ball_df['launch_speed'].values[:top_babip_count].mean()

        avg_pitch_speeds = batted_ball_df['release_speed'].values[:top_babip_count].mean()

        estimated_bat_speed = (top_ev_mean / (1 + e)) - (avg_pitch_speeds * e)

        return round(estimated_bat_speed, precision)

    def calculate_estimated_attack_angle(self,start=None,end=None,df=None,top_percentile=0.1,precision=1):
        if df is None:
            df = self.data

        e = 0.2

        batted_ball_df = df[(df['launch_speed'].notna()) & (df['launch_angle'].notna()) & (df['event']=='hit_into_play')]

        batted_ball_df = batted_ball_df.sort_values(by='launch_speed',ascending=False)

        total_babip_count = batted_ball_df.shape[0]

        top_babip_count = int(top_percentile * total_babip_count)

        if total_babip_count == 0:
            return None
        if top_babip_count == 0:
            top_babip_count = 1

        top_la_mean = batted_ball_df['launch_angle'].values[:top_babip_count].mean()

        return round(top_la_mean, precision)

    def calculate_quality_of_contact(self,start=None,end=None,df=None,precision=3,verbose=False,top_percentile=0.1):
        whiff_value = 0
        foul_value = 0

        if df is None:
            df = self.data

        # Quality of Contact = 1 + (Exit Velocity – Bat Speed)/(Pitch Speed + Bat Speed)

        swing_df = df[df['isSwing']==True]

        babip_df = swing_df[swing_df['event']=='hit_into_play']
        whiff_df = swing_df[(swing_df['event']=='swinging_strike') & (swing_df['event']=='swinging_strike_blocked')]
        foul_df = swing_df[swing_df['event']=='foul']

        foul_with_launch = foul_df[(foul_df['launch_speed'].notna()) & (foul_df['launch_angle'].notna())]
        foul_no_launch = foul_df[(foul_df['launch_speed'].isna()) | (foul_df['launch_angle'].isna())]
        bat_speed = self.calculate_estimated_bat_speed(df=df,top_percentile=top_percentile)

        babip_df['quality of contact'] = 1 + (babip_df['launch_speed'] - bat_speed) / (babip_df['release_speed'] + bat_speed)
        whiff_df['quality of contact'] = whiff_value
        foul_with_launch['quality of contact'] = 1 + (foul_with_launch['launch_speed'] - bat_speed) / (foul_with_launch['release_speed'] + bat_speed)
        foul_no_launch['quality of contact'] = foul_value

        aggregate_df = pd.concat([babip_df,whiff_df,foul_with_launch,foul_no_launch])

        total_qoc = aggregate_df['quality of contact'].mean()

        babip_qoc = babip_df['quality of contact'].mean()
        foul_qoc = foul_with_launch['quality of contact'].mean()

        if verbose:
            output = (round(val, precision) for val in (total_qoc,babip_qoc,foul_qoc))
            return output
        else:
            return round(total_qoc, precision)


    def display_launchAngle_vs_exitVelocity(self,start=None,end=None,df=None,top_percentile=0.1):
        if df is None:
            df = self.data

        batted_ball_df = df[(df['launch_speed'].notna()) & (df['launch_angle'].notna()) & (df['event']=='hit_into_play')]

        fig,ax = plt.subplots(figsize=(10,6))

        if top_percentile is not None:
            
            batted_ball_df = batted_ball_df.sort_values(by='launch_speed',ascending=False)
            N = batted_ball_df.shape[0]

            if top_percentile < 1:
                top_percentile_N = int(top_percentile * N)
            else:
                top_percentile_N = top_percentile

            top_ev = batted_ball_df['launch_speed'].values[:top_percentile_N]
            top_la = batted_ball_df['launch_angle'].values[:top_percentile_N]

            btm_ev = batted_ball_df['launch_speed'].values[top_percentile_N:]
            btm_la = batted_ball_df['launch_angle'].values[top_percentile_N:]

            top_ev_mean = top_ev.mean()
            top_la_mean = top_la.mean()

            e = 0.2
            estimated_bat_speed = (top_ev_mean/(1+e)) - (batted_ball_df['release_speed'].values[:top_percentile_N].mean() * e)

            print(f'Top EV Mean: {top_ev_mean:.1f} - Top EV LA Mean: {top_la_mean:.1f}')
            print(f'Estimated Bat Speed: {estimated_bat_speed:.1f}')

            ax.scatter(top_ev,top_la,color='tab:red',alpha=.5)
            ax.scatter(btm_ev,btm_la,color='tab:blue',alpha=.5)
            ax.scatter(top_ev_mean,top_la_mean,color='k')

        else:
            exit_velos = batted_ball_df['launch_speed'].values
            launch_angles = batted_ball_df['launch_angle'].values

            ax.scatter(exit_velos,launch_angles)


        ax.grid()

        plt.show()

    def calculate_average_to_max_exit_velocity_ratio(self,start=None,end=None,df=None,precision=3,max_ev_samples=3):
        if df is None:
            df = self.data

        avg_ev = self.calculate_average_exit_velocity(df=df)
        max_ev = self.calculate_max_exit_velocity(df=df,n=max_ev_samples)

        ratio = avg_ev/max_ev
        return round(ratio,precision)


    def calculate_percent_above_exit_velocity_threshold(self,threshold,start=None,end=None,df=None,precision=3,max_ev_samples=3):
        if df is None:
            df = self.data

        df = df[df['launch_speed'].notna()]
        exit_velos = df['launch_speed'].values

        if threshold < 1:
            max_ev = self.calculate_max_exit_velocity(n=max_ev_samples)
            hits_above = sum((exit_velos > (threshold * max_ev)).astype(int))
        else:
            hits_above = sum((exit_velos > threshold).astype(int))

        total_hits = len(exit_velos)

        percent = hits_above / total_hits
        return round(percent, precision)

    def calculate_total_wOBA(self,df=None, precision=3):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        batter_woba = woba_df['woba_value'].mean()

        return round(batter_woba,precision)

    def calculate_zone_wOBA(self,df=None, precision=3):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        _df = woba_df[woba_df['isStrike']==True]

        zone_woba = _df['woba_value'].mean()

        return round(zone_woba,precision)

    def calculate_outside_wOBA(self,df=None,precision=3):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        _df = woba_df[woba_df['isStrike']==False]

        outside_woba = _df['woba_value'].mean()

        return round(outside_woba,precision)

    def calculate_horizontal_slice_wOBA(self,df=None):
        # High
        BL = (-1.1,0.5)
        TR = (1.1,1.1)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        high_woba = woba_df['woba_value'].mean().round(3)

        # Middle
        BL = (-1.1,-0.5)
        TR = (1.1,0.5)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        middle_woba = woba_df['woba_value'].mean().round(3)

        # Low
        BL = (-1.1,-1.1)
        TR = (1.1,-0.5)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        low_woba = woba_df['woba_value'].mean().round(3)

        return high_woba, middle_woba, low_woba

    def calculate_vertical_slice_wOBA(self,df=None):
        # Inside
        BL = (-1.1,-1.1)
        TR = (-0.5,1.1)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        inside_woba = woba_df['woba_value'].mean().round(3)

        # Middle
        BL = (-0.5,-1.1)
        TR = (0.5,1.1)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        middle_woba = woba_df['woba_value'].mean().round(3)

        # Outside
        BL = (0.5,-1.1)
        TR = (1.1,1.1)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        outside_woba = woba_df['woba_value'].mean().round(3)

        return inside_woba, middle_woba, outside_woba

    ### Functions for Expected wOBA
    def calculate_total_xwOBA(self,df=None):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        batter_woba = woba_df['estimated_woba_using_speedangle'].mean().round(3)

        return batter_woba

    def calculate_zone_xwOBA(self,df=None):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        _df = woba_df[woba_df['isStrike']==True]

        zone_woba = _df['estimated_woba_using_speedangle'].mean().round(3)

        return zone_woba

    def calculate_outside_xwOBA(self,df=None):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        _df = woba_df[woba_df['isStrike']==False]

        outside_woba = _df['estimated_woba_using_speedangle'].mean().round(3)

        return outside_woba

    def calculate_horizontal_slice_xwOBA(self,df=None):
        # High
        BL = (-1.1,0.5)
        TR = (1.1,1.1)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        high_woba = woba_df['estimated_woba_using_speedangle'].mean().round(3)

        # Middle
        BL = (-1.1,-0.5)
        TR = (1.1,0.5)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        middle_woba = woba_df['estimated_woba_using_speedangle'].mean().round(3)

        # Low
        BL = (-1.1,-1.1)
        TR = (1.1,-0.5)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        low_woba = woba_df['estimated_woba_using_speedangle'].mean().round(3)

        return high_woba, middle_woba, low_woba

    def calculate_vertical_slice_xwOBA(self,df=None):
        # Inside
        BL = (-1.1,-1.1)
        TR = (-0.5,1.1)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        inside_woba = woba_df['estimated_woba_using_speedangle'].mean().round(3)

        # Middle
        BL = (-0.5,-1.1)
        TR = (0.5,1.1)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        middle_woba = woba_df['estimated_woba_using_speedangle'].mean().round(3)

        # Outside
        BL = (0.5,-1.1)
        TR = (1.1,1.1)

        _df = self.filter_data_by_location(BL,TR,df=df)
        woba_df = _df[_df['woba_denom']==1]

        outside_woba = woba_df['estimated_woba_using_speedangle'].mean().round(3)

        return inside_woba, middle_woba, outside_woba

    def analyze_pitch_decision(self):
        self.isStrike()
        self.isSwing()
        self.isContact()
        self.isCorrectDecision()

    def count_plate_appearances(self,start=None,end=None):
        count = 0
        for game in self.games:
            count += len(game.atBats)
        return count

    def count_strikeouts(self,start=None,end=None):
        count = 0
        for game in self.games:
            for atBat in game.atBats:
                if atBat.isStrikeout:
                    count += 1
        return count

    def count_walks(self,start=None,end=None):
        count = 0
        for game in self.games:
            for atBat in game.atBats:
                if atBat.isWalk:
                    count += 1
        return count

    def calculate_strikeout_rate(self,start=None,end=None,precision=3):
        K = self.count_strikeouts(start=start,end=end)
        PA = self.count_plate_appearances(start=start,end=end)
        rate = K / PA
        return round(rate, precision)
    
    def calculate_walk_rate(self,start=None,end=None,precision=3):
        K = self.count_walks(start=start,end=end)
        PA = self.count_plate_appearances(start=start,end=end)
        rate = K / PA
        return round(rate, precision)

    def calculate_chase_rate(self,df=None,start=None,end=None,precision=3):
        if df is None:
            df = self.data

        not_strike_df = df[df['isStrike']==False]
        batter_incorrect = not_strike_df[not_strike_df['isCorrectDecision']==False]
        chase_rate = batter_incorrect.shape[0] / not_strike_df.shape[0]
        return round(chase_rate,precision)

    def calculate_chase_rate_plus(self,start=None,end=None,precision=3):
        not_strike_df = self.data[self.data['isStrike']==False]
        batter_incorrect = not_strike_df[not_strike_df['isCorrectDecision']==False]

        scaling = batter_incorrect['prop_dist_to_zone'].mean()

        chase_rate = (batter_incorrect.shape[0] * scaling) / not_strike_df.shape[0]
        return round(chase_rate,precision)

    def calculate_correct_swing_decision_rate(self,df=None,start=None,end=None,precision=3):
        if df is None:
            df = self.data

        batter_correct = df[df['isCorrectDecision']==True]
        correct_rate = batter_correct.shape[0] / df.shape[0]

        return round(correct_rate,precision)

    def calculate_chase_rate_linear_scaling(self,df=None,start=None,end=None,precision=3):
        if df is None:
            df = self.data

        not_strike_df = df[df['isStrike']==False]
        batter_incorrect = not_strike_df[not_strike_df['isCorrectDecision']==False]

        scaling = batter_incorrect['prop_dist_to_zone'].mean()

        return round(scaling,precision)

    def calculate_chase_rate_plus_rms(self,start=None,end=None,precision=3):
        not_strike_df = self.data[self.data['isStrike']==False]
        batter_incorrect = not_strike_df[not_strike_df['isCorrectDecision']==False]

        scaling = np.sqrt((batter_incorrect['prop_dist_to_zone']**2).mean())
        
        chase_rate = (batter_incorrect.shape[0] * scaling) / not_strike_df.shape[0]
        return round(chase_rate,precision)

    def calculate_chase_rate_plus_exp(self,start=None,end=None,precision=3):
        not_strike_df = self.data[self.data['isStrike']==False]
        batter_incorrect = not_strike_df[not_strike_df['isCorrectDecision']==False]

        scaling = (batter_incorrect['prop_dist_to_zone']**batter_incorrect['prop_dist_to_zone']).mean()
        
        chase_rate = (batter_incorrect.shape[0] * scaling) / not_strike_df.shape[0]
        return round(chase_rate,precision)

    def calculate_guassian_heatmap(self,start=None,end=None,df=None,bin_sigma=4,scale_key='estimated_woba_using_speedangle',blur_sigma=5):
        baseball_diameter = 0.25  # Diameter of baseball in ft (3in)
        baseball_radius = baseball_diameter / 2
        prop_diameter = baseball_diameter / self.zone_half_width

        strikezone_bins = 60
        zone_center = strikezone_bins * 1.5
        prop_bin_width = 2 / strikezone_bins

        s = signal.gaussian(4*bin_sigma+1,bin_sigma)
        kernel_width = int(len(s) / 2)

        kernel_length = len(s)
        kernel = np.zeros((kernel_length,kernel_length))
        for i in np.arange(kernel_length):
            for j in np.arange(kernel_length):
                kernel[i,j] = s[i] * s[j]

        bins = np.zeros((strikezone_bins*3,strikezone_bins*3))

        if df is None:
            df = self.data[self.data['estimated_woba_using_speedangle'].notna()]

        for i in df.index:
            pitch = df.loc[i]

            xLoc = pitch.loc['prop_plate_x']
            yLoc = pitch.loc['prop_plate_z']
            
            if np.isnan(xLoc) or np.isnan(yLoc):
                continue
            
            xBin = int(zone_center + xLoc / prop_bin_width)
            yBin = int(zone_center + yLoc / prop_bin_width)
            
            if yBin < kernel_width or yBin > (strikezone_bins*3 - kernel_width - 1):
                continue
            if xBin < kernel_width or xBin > (strikezone_bins*3 - kernel_width - 1):
                continue
            
            bins[yBin-kernel_width:yBin+kernel_width+1,xBin-kernel_width:xBin+kernel_width+1] += (kernel * pitch.loc[scale_key])

        blurred_bins = sp.ndimage.gaussian_filter(bins,sigma=blur_sigma)

        return blurred_bins

    def plot_heatmap(self,bins,scale_reference=0,display=True,output=False):
        fig,ax = plt.subplots(figsize=(10,10))

        im = ax.matshow(bins - scale_reference,cmap='bwr')
        ax.plot([60,120,120,60,60],[120,120,60,60,120],color='w')
        ax.plot()
        plt.colorbar(im)

        ax.set_xticks(np.arange(-.5,180,30))
        ax.set_xticklabels(np.arange(-3,3.1,1))

        ax.set_yticks(np.arange(-.5,180,30))
        ax.set_yticklabels(np.arange(3,-3.1,-1))

        if output:
            pass
        if display:
            plt.show()

    def plot_wOBA_heatmap(self,df=None,bin_sigma=4,blur_sigma=5):
        bins = self.calculate_guassian_heatmap(df=df,bin_sigma=bin_sigma,blur_sigma=blur_sigma)

        zone_median = np.median(bins[60:121,60:121])

        self.plot_heatmap(bins,scale_reference=zone_median)

    def plot_pitch_scatter(self,df=None):
        if df is None:
            df = self.data

        swing_df = df[df['isSwing']==True]
        no_swing_df = df[df['isSwing']==False]

        babip_df = swing_df[swing_df['event']=='hit_into_play']
        foul_df = swing_df[swing_df['event']=='foul']

        whiff_df = df[(df['isSwing']==True) & (df['isContact']==False)]

        BL = (-1,-1)
        TL = (-1,1)
        TR = (1,1)
        BR = (1,-1)

        fig,ax = plt.subplots(figsize=(10,10))

        x = [it[0] for it in (BL,TL,TR,BR,BL)]
        y = [it[1] for it in (BL,TL,TR,BR,BL)]

        ax.plot(x,y,color='k')

        # Horizontals
        ax.plot([-1,1],[0.333,0.333],color='k',alpha=.5)
        ax.plot([-1,1],[-0.333,0.-.333],color='k',alpha=.5)

        # Verticals
        ax.plot([-0.333,-0.333],[1,-1],color='k',alpha=.5)
        ax.plot([0.333,0.333],[1,-1],color='k',alpha=.5)

        ax.scatter(babip_df['prop_plate_x'],babip_df['prop_plate_z'],color='tab:blue',alpha=.4)
        ax.scatter(no_swing_df['prop_plate_x'],no_swing_df['prop_plate_z'],color='k',alpha=.3)

        ax.scatter(whiff_df['prop_plate_x'],whiff_df['prop_plate_z'],color='tab:red',alpha=.4,marker='x')
        ax.scatter(foul_df['prop_plate_x'],foul_df['prop_plate_z'],color='tab:red',alpha=.4,marker='o')

        ax.grid()

        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        plt.show()

    def return_data_by_index(self,start=None,end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.games)
        
        _dfs = []
        for game in self.games[start:end]:
            _dfs.append(game.data)
            
        _df = pd.concat(_dfs)
        
        return _df

    def return_data_by_date(self,start=None,end=None):
        if start is None:
            start = self.games[0].date
        if end is None:
            end = self.games[-1].date
        
        _dfs = []
        for game in self.games:
            start_int = int(start.replace('-',''))
            game_int = int(game.date.replace('-',''))
            end_int = int(end.replace('-',''))
            
            if start_int > game_int:
                continue
                
            if game_int > end_int:
                break
                
            _dfs.append(game.data)
            
        _df = pd.concat(_dfs)
        
        return _df

class Game:
    def __init__(self,gameID,data):
        self.gameID = gameID
        self.data = data

        self.atBats = []

        self.process_atBats()

    @property
    def date(self):
        return self.data['game_date'].values[0]

    def __str__(self):
        return self.date

    def add_atBat(self,number,data):
        self.atBats.append(AtBat(number,data))

    def process_atBats(self):
        atBat_numbers = self.data['at_bat_number'].unique()
        for atBat_number in atBat_numbers:
            atBat_data = self.data[self.data['at_bat_number']==atBat_number]
            self.add_atBat(f'{atBat_number}',atBat_data)

    def sort_atBats(self):
        def sortFunc(x):
            return x.data['at_bat_number'].values[0]

        self.atBats.sort(key=sortFunc)

class AtBat:
    def __init__(self,number,data):
        self.number = number
        self.data = data

    @property
    def result_pitch(self):
        return self.data[self.data['pitch_number']==self.data['pitch_number'].max()]

    @property
    def result(self):
        return self.result_pitch['event_result'].values[0]

    @property
    def isWalk(self):
        return self.result == 'walk'

    @property
    def isStrikeout(self):
        return self.result == 'strikeout' or self.result == 'strikeout_double_play'

    

    

    




