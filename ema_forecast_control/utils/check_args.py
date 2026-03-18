from typing import Optional
import os
import pandas as pd

def check_args(args: dict) -> dict:

    check_mandatory_args_present(args)
    args = check_obs_features(args)
    args = check_input_features(args)
    args = check_timestamp(args)
    args = check_preprocessing(args)
    args = check_train_test_split(args)
    return args

def check_mandatory_args_present(args: dict) -> bool:
    mandatory_args = [
        'data_directory',
        'obs_features',
        'input_features',
        'timestamp',
        'preprocessing',
        'train_test_split',
        'model',
    ]
    not_present = [arg for arg in mandatory_args if arg not in args]
    if len(not_present) > 0:
        raise ValueError(f'In project yml file, the following mandatory arguments are missing: {not_present}')
    
def check_obs_features(args: dict) -> dict:
    if not (isinstance(args['obs_features'], list) and len(args['obs_features']) > 0):
        raise ValueError('In project yml file, obs_features must be a non-empty list.')
    return args

def check_input_features(args: dict) -> dict:
    if isinstance(args['input_features'], list) and len(args['input_features']) == 0:
        args['input_features'] = None
    return args

def check_timestamp(args: dict) -> dict:
    if not (isinstance(args['timestamp'], dict)
            and ('absolute_datetime_column' in args['timestamp']
                 or 'relative_time_column' in args['timestamp'])):
        raise ValueError('In project yml file, timestamp must be a dictionary containing'
                         ' either absolute_datetime_column or relative_time_column.')
    if (('relative_time_column' in args['timestamp']) 
        and not ('time_anchor' in args['timestamp'] 
                 and 'datetime_format' in args['timestamp'])):
        raise ValueError('In project yml file, if relative_time_column is provided, time_anchor must also be provided.')
    return args

def check_preprocessing(args: dict) -> dict:
    if not (isinstance(args['preprocessing'], list)
            or args['preprocessing'] is None):
        raise ValueError('In project yml file, preprocessing must be a list of functions to apply, or null.')
    return args

def check_train_test_split(args: dict) -> dict:

    if isinstance(args['train_test_split'], list):
        if len(args['train_test_split']) > 0:
            common_type = type(args['train_test_split'][0])
            items_have_same_type = all(isinstance(item, common_type) for item in args['train_test_split'])
            valid = common_type in [int, str] and items_have_same_type
            if common_type == str:
                valid = valid and all(check_string_is_date(item) for item in args['train_test_split'])        
    elif isinstance(args['train_test_split'], str):
        valid = check_string_is_date(args['train_test_split']) or os.path.isfile(args['train_test_split'])
    elif isinstance(args['train_test_split'], int):
        valid = True
    else:
        valid = False
    if not valid:
        raise ValueError('In project yml file, train_test_split must be one of the following:'
                         ' A path to a file, an integer, a date string, a list of integers or a list of date strings.')
    return args

def check_model(args: dict) -> dict:
    available_models = [
        'PLRNN', 'Transformer', 'KalmanFilter',
        'VAR1', 'InputsRegression', 'MeanPredictor', 'MovingAverage'
    ]
    if not (isinstance(args['model'], str)
            and any(args['model'].lower().startswith(model.lower()) for model in available_models)):
        raise ValueError(f'In project yml file, model must be one of the following: {available_models}')
    return args

def check_string_is_date(string: str) -> bool:
    try:
        pd.to_datetime(string)
        return True
    except ValueError:
        return False
    
