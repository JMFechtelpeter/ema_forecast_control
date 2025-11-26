#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:58:30 2022

@author: janik
"""
import sys
sys.path.append('..')
import pandas as pd
import torch as tc
from bptt.plrnn import PLRNN

if __name__=='__main__':
    
    data_path = '/home/janik.fechtelpeter/Documents/reallaborai4u/data_management_MRT2/processed_csv_no_con_smoothed/12600_12.csv'    
    model_path = '/home/janik.fechtelpeter/Documents/ordinal-bptt/results/v2_MRT2_every_valid_day/data_12600_12.csv_participant_12_date_180.0/001'

    model = PLRNN(load_model_path=model_path)

    data = pd.read_csv(data_path)
    emas = tc.tensor(data[model.args['obs_features']].to_numpy()).float()
    inputs = tc.tensor(data[model.args['input_features']].to_numpy()).float()

    now = 180
    n_steps = 10

    prediction = model.generate_free_trajectory(emas[now], n_steps, 
                                                inputs = inputs[now:now+n_steps],
                                                prewarm_data=emas[now-4:now],
                                                prewarm_inputs=inputs[now-4:now],
                                                )