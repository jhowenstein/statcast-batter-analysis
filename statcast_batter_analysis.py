import numpy as np
import pandas as pd

import os
import glob

import matplotlib.pyplot as plt


swing_events = ['hit_into_play','foul','swinging_strike','swinging_strike_blocked']
contact_swing_events = ['hit_into_play','foul']
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
            self.process_games(process_atBats=process_atBats)

    def __str__(self):
        return self.name

    @property
    def sz_top(self):
        return self.data['sz_top'].mean().round(3)

    @property
    def sz_btm(self):
        return self.data['sz_bot'].mean().round(3)

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

    def isStrike(self):
        self.data['isStrike'] = (self.data['plate_x'] >= self.sz_left) & (self.data['plate_x'] <= self.sz_right) & (
            self.data['plate_z'] >= self.sz_btm) & (self.data['plate_z'] <= self.sz_top)

    def isSwing(self):
        swing_events = ['hit_into_play','foul','swinging_strike','swinging_strike_blocked']
        
        self.data['isSwing'] = (self.data['event']=='hit_into_play') | (self.data['event']=='foul') | (
            self.data['event']=='swinging_strike') | (self.data['event']=='swinging_strike_blocked')

    def isCorrectDecision(self):
        self.data['isCorrectDecision'] = self.data['isStrike'] == self.data['isSwing']

    def analyze_pitch_decision(self):
        self.isStrike()
        self.isSwing()
        self.isCorrectDecision()

class Game:
    def __init__(self,gameID,data):
        self.gameID = gameID
        self.data = data

        self.atBats = []

        self.process_atBats()

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

    

    




