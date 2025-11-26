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
import numpy as np
import os
import re
import sys

from bptt.plrnn import PLRNN


def prediction_errors_model(model_path, model_nr, now, step_by_step, n_steps, emas, inputs, emas_test, prewarm=False):
    mse_per_step_list = []
    mae_per_step_list = []

    input_emas = emas_test # To easily change between emas smoothed (emas) and emas non smoothed (emas_test) as inputs for the model
    emas_validation = emas_test[now+1:now+n_steps+1] # Validation data is non smoothed

    model_path_specific = os.path.join(model_path, model_nr)
    try:
        model = PLRNN(load_model_path=model_path_specific)
    except AssertionError as e:
        print(f"Error: {e}. No model found at {model_path_specific}. Exiting function.")
        return [],[]

    # Generate model predictions
    predictions_list = []
    if step_by_step:
        for i in range(n_steps):
            predictions = model.generate_free_trajectory(input_emas[now], 1, 
                                                    inputs = inputs[now:now+1],
                                                    prewarm_data=input_emas[now-4:now],
                                                    prewarm_inputs=inputs[now-4:now],
                                                    )
            predictions_list.append(predictions)
    else:
        predictions = model.generate_free_trajectory(input_emas[now], n_steps, 
                                                    inputs = inputs[now:now+n_steps],
                                                    prewarm_data=input_emas[now-4:now],
                                                    prewarm_inputs=inputs[now-4:now],
                                                    )
        predictions_list.append(predictions)
    predictions_tensor = tc.cat(predictions_list, dim=0)
        
    # Compute the prediction error
    for i in range(len(emas_validation)):
        # Skip if there is a NaN value in the validation data or in the current row of the emas (if you generate trajectories from the current row you dont want to give NaN as input)
        if tc.isnan(emas_validation[i]).any() or tc.isnan(input_emas[now+i-1]).any():
            continue
        # Skip if there is a NaN value in the prewarm data (set prewarm = True)
        if prewarm and tc.isnan(input_emas[now+i-4:now+i]).any(): 
            continue
        prediction = predictions_tensor[i]
        target = emas_validation[i]
        mse_per_step = tc.mean((prediction - target)**2).item()
        mae_per_step = tc.mean(tc.abs(prediction - target)).item()
        mse_per_step_list.append(mse_per_step)
        mae_per_step_list.append(mae_per_step)

    return mse_per_step_list, mae_per_step_list

def get_model_paths(participant_nr, folder_path):
    model_paths = {}
    for filename in os.listdir(folder_path):
        match = re.search(rf'data_\d+_\d+\.csv_participant_{participant_nr}_date_(\d+\.\d+)', filename)
        if match:
            timestep = float(match.group(1))
            model_paths[timestep] = os.path.join(folder_path, filename)
    return model_paths

def get_csv_file_path(participant_nr, folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and filename.split("_")[1].split(".")[0] == str(participant_nr):
            return os.path.join(folder_path, filename)
    return None

def load_data(participant_nr, data_directory_smoothed, data_directory):
    ema_labels = ['EMA_mood', 'EMA_disappointed', 'EMA_scared', 'EMA_worry', 'EMA_down', 'EMA_sad', 'EMA_confidence', 'EMA_stress', 'EMA_lonely', 'EMA_energetic', 'EMA_concentration', 'EMA_resilience', 'EMA_tired', 'EMA_satisfied', 'EMA_relaxed']
    input_labels = ['interactive1', 'interactive2', 'interactive3', 'interactive4', 'interactive5', 'interactive6', 'interactive7', 'interactive8']

    # Extract file paths of the csvs that correspond to the participant_nr
    csv_path_smoothed = get_csv_file_path(participant_nr, data_directory_smoothed)
    csv_path = get_csv_file_path(participant_nr, data_directory)
    
    csv_df = pd.read_csv(csv_path_smoothed)
    emas = tc.tensor(csv_df[ema_labels].values, dtype=tc.float32)
    inputs = tc.tensor(csv_df[input_labels].values, dtype=tc.float32)

    csv_df_test = pd.read_csv(csv_path)
    emas_test = tc.tensor(csv_df_test[ema_labels].values, dtype=tc.float32)
    
    return emas, inputs, emas_test

def prediction_errors_per_participant(participant_nr, folder_path_model, model_nr, data_directory_smoothed, data_directory, step_by_step, n_steps):
    mse_per_step_complete_list = []
    mae_per_step_complete_list = []

    emas, inputs, emas_test = load_data(participant_nr, data_directory_smoothed, data_directory)

    model_paths = get_model_paths(participant_nr, folder_path_model)
    for now, model_path in model_paths.items():
        now = int(now)
        mse_per_step_list, mae_per_step_list = prediction_errors_model(model_path, model_nr, now, step_by_step, n_steps, emas, inputs, emas_test)
        mse_per_step_complete_list.extend(mse_per_step_list)
        mae_per_step_complete_list.extend(mae_per_step_list)

    return  mse_per_step_complete_list, mae_per_step_complete_list

def extract_participant_ids(folder_path):
    participant_ids = set()

    for filename in os.listdir(folder_path):
        match = re.search(r'participant_(\d+)', filename)
        if match:
            participant_ids.add(int(match.group(1)))
    
    return sorted(participant_ids)

if __name__=='__main__':
    folder_path_MRT2 = "D:/v2_MRT2_every_valid_day"
    folder_path_MRT3 = "D:/v2_MRT3_every_valid_day"
    data_folder_MRT2_smoothed = "data/MRT2/processed_csv_no_con_smoothed"
    data_folder_MRT2 = "data/MRT2/processed_csv_no_con"
    data_folder_MRT3_smoothed = "data/MRT3/processed_csv_no_con_smoothed"
    data_folder_MRT3 = "data/MRT3/processed_csv_no_con"

    # Set parameters to evaluate the prediction accuracy in different scenarios
    step_by_step = True  # If True: model predicts the next step using current time step's EMAs as input (also prewarm inputs)
    n_steps = 3
    model_nr = "001"

    participants_MRT2 = extract_participant_ids(folder_path_MRT2)
    participants_MRT3 = extract_participant_ids(folder_path_MRT3)

    mse_overall_list = []
    mae_overall_list = []

    for participant in participants_MRT2:
        mse_per_participant_list, mae_per_participant_list = prediction_errors_per_participant(participant, folder_path_MRT2, model_nr, data_folder_MRT2_smoothed, data_folder_MRT2, step_by_step, n_steps)
        mse_overall_list.extend(mse_per_participant_list)
        mae_overall_list.extend(mae_per_participant_list)

    for participant in participants_MRT3:
        mse_per_participant_list, mae_per_participant_list = prediction_errors_per_participant(participant, folder_path_MRT3, model_nr, data_folder_MRT3_smoothed, data_folder_MRT3, step_by_step, n_steps)
        mse_overall_list.extend(mse_per_participant_list)
        mae_overall_list.extend(mae_per_participant_list)

    print(f'Number of valid predictions: {len(mse_overall_list)}')
    print(f'MSE: {np.mean(mse_overall_list)}')
    print(f'MAE: {np.mean(mae_overall_list)}')