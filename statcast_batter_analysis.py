import numpy as np
import pandas as pd

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
        swing_events = ['hit_into_play','foul','swinging_strike','swinging_strike_blocked']
        
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

    def calculate_percent_hard_hit(self,start=None,end=None,df=None,precision=3,threshold=.9):
        if df is None:
            df = self.data

        max_ev = self.calculate_max_exit_velocity(n=3)

        df = df[df['launch_speed'].notna()]
        exit_velos = df['launch_speed'].values
        
        hard_hit = sum((exit_velos>(threshold * max_ev)).astype(int))
        total_hit = len(exit_velos)

        hard_hit_rate = hard_hit / total_hit
        return round(hard_hit_rate, precision)

    def calculate_total_wOBA(self,df=None):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        batter_woba = woba_df['woba_value'].mean().round(3)

        return batter_woba

    def calculate_zone_wOBA(self,df=None):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        _df = woba_df[woba_df['isStrike']==True]

        zone_woba = _df['woba_value'].mean().round(3)

        return zone_woba

    def calculate_outside_wOBA(self,df=None):
        if df is None:
            df = self.data

        woba_df = df[df['woba_denom']==1]

        _df = woba_df[woba_df['isStrike']==False]

        outside_woba = _df['woba_value'].mean().round(3)

        return outside_woba

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

    def calculate_chase_rate(self,start=None,end=None,precision=3):
        not_strike_df = self.data[self.data['isStrike']==False]
        batter_incorrect = not_strike_df[not_strike_df['isCorrectDecision']==False]
        chase_rate = batter_incorrect.shape[0] / not_strike_df.shape[0]
        return round(chase_rate,precision)

    def calculate_chase_rate_plus(self,start=None,end=None,precision=3):
        not_strike_df = self.data[self.data['isStrike']==False]
        batter_incorrect = not_strike_df[not_strike_df['isCorrectDecision']==False]

        scaling = batter_incorrect['prop_dist_to_zone'].mean()

        chase_rate = (batter_incorrect.shape[0] * scaling) / not_strike_df.shape[0]
        return round(chase_rate,precision)

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

class Game:
    def __init__(self,gameID,data):
        self.gameID = gameID
        self.data = data

        self.atBats = []

        self.process_atBats()

    @property
    def date(self):
        return self.data['game_date'].values[0]

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

    

    

    




