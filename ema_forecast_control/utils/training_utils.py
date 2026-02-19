import logging
log = logging.getLogger(__name__)

from typing import Optional
import os
import pickle
import glob

import yaml
import pynvml
import numpy as np
import torch as tc
import pandas as pd

from ema_forecast_control import ROOT
import ema_forecast_control.preprocessing.preprocessing as preprocessing
import ema_forecast_control.preprocessing.prepare_timestamp as prepare_timestamp
import ema_forecast_control.preprocessing.train_test_split as train_test_split
from ema_forecast_control.dataset.time_series_dataset import TimeSeriesDataset
from ema_forecast_control.utils import data_utils

def get_project_dict(project: str) -> dict:
    with open(os.path.join(ROOT, f'projects/{project}.yml'), 'r') as file:
        project_dict = yaml.safe_load(file)
    project_dict['project_name'] = project
    return project_dict

def parse_project_argument(project_dict: dict, arg: str):
    if arg not in project_dict:
        raise ValueError(f'Argument {arg} not found in project dictionary.')
    
def next_run(trial_path: str) -> str:
        """increase by one each run, if none exists start at '001' """
        run_nrs = get_runs(trial_path)
        if not run_nrs:
            run_nrs = ['000']
        run = str(int(max(run_nrs)) + 1).zfill(3)
        run_dir = os.path.join(trial_path, run)
        return run_dir
    
def get_runs(trial_path: str) -> list:
    try:
        run_nrs = [d for d in os.listdir(trial_path) if os.path.isdir(os.path.join(trial_path, d)) and d.isdigit()]
        return run_nrs
    except FileNotFoundError:
        return []    

def create_model_dir(project_name: str, configuration_name: str, run: Optional[int]=None, overwrite: bool=True) -> str:
    '''Create directory for saving model based on project name, configuration and run number. Return the path.'''
    
    trial_path = os.path.join(ROOT, 'trained_models', project_name, configuration_name)
    if run is None:
        run_path = next_run(trial_path)
    else:
        run_path = os.path.join(trial_path, str(run).zfill(3))
    os.makedirs(run_path, exist_ok=overwrite)

    return run_path

def prepare_device(args: dict) -> tc.device:
    ''' Prepare and return a torch.device instance. '''
    def find_gpu_with_lowest_utilization(criterion: str='load') -> int:
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        deviceMemUtilization = np.zeros(deviceCount)
        deviceWorkload = np.zeros(deviceCount)
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            deviceMemUtilization[i] = mem.used
            deviceWorkload[i] = util.gpu
        pynvml.nvmlShutdown()
        if criterion == 'load':
            return np.argmin(deviceWorkload).item()
        elif criterion == 'mem':
            return np.argmin(deviceMemUtilization).item()
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Use 'load' or 'mem'.")
    # check cuda availability
    if args['use_gpu'] and tc.cuda.is_available():
        id_ = args['device_id']
        if id_ == -1:
            id_ = find_gpu_with_lowest_utilization(criterion='load')
        device = tc.device('cuda', id_)
    else:
        device = tc.device('cpu')
        tc.set_num_threads(1)        
    log.info("Using device %s for training.", str(device))
    return device

def prepare_dataset_update_args(args: dict, preloaded_data: Optional[pd.DataFrame]=None) -> tuple[dict, TimeSeriesDataset, Optional[tc.Tensor], Optional[tc.Tensor]]:
 
    if 'data_dropout_to_level' in args and args['data_dropout_to_level'] not in (None, 'None'):
        valid_ratio = float(args['data_dropout_to_level'])
    else:
        valid_ratio = None
    if preloaded_data is None:
        preloaded_data = pd.read_csv(args['data_path'])
    
    preloaded_data = prepare_timestamp.prepare_timestamp(preloaded_data, **args['timestamp'])
    train_df, test_df = train_test_split.train_test_split(preloaded_data, args['train_test_split'])
    train_df = preprocessing.preprocessing_pipeline(train_df, args['preprocessing'])

    if not args['train_on_last_n_steps'] in (None, 'None'):
        if isinstance(args['train_on_last_n_steps'], str):
            args['train_on_last_n_steps'] = int(args['train_on_last_n_steps'])
        train_df = train_df.iloc[-args['train_on_last_n_steps']:]
    log.info(f"Loaded data with {train_df.shape[0]} time steps.")

    train_data = tc.tensor(train_df[args['obs_features']].values).float()
    test_data = tc.tensor(test_df[args['obs_features']].values).float()
    if args['input_features'] is not None:
        train_inputs = tc.tensor(train_df[args['input_features']].values).float()
        test_inputs = tc.tensor(test_df[args['input_features']].values).float()
    else:
        train_inputs = None
        test_inputs = None
    
    train_dataset = TimeSeriesDataset(train_data, train_inputs, name=os.path.basename(args['data_path']),
                                    seq_len=args.get('seq_len', 0),
                                    partial_missings_are_valid=args.get('partial_missings_are_valid', False),
                                    tolerate_reduced_seq_len=args.get('tolerate_reduced_seq_len', True),
                                    max_valid_data_ratio=valid_ratio,
                                    verbose=args.get('verbose', 'none'))

    if 'min_valid_training_timesteps' in args:
        if train_dataset.n_valid < args['min_valid_training_timesteps']:
            raise ValueError(f'Training dataset contains only {train_dataset.n_valid} valid timesteps, '
                             f'but at least {args["min_valid_training_timesteps"]} are required.')

    log.debug('Successfully created train set.')
    
    args['dim_x'] = train_data.shape[1]
    if train_inputs is not None:
        args['dim_s'] = train_inputs.shape[1]
    else:
        args['dim_s'] = None
    args['participant'] = data_utils.determine_participant_id(preloaded_data)

    return args, train_dataset, test_data, test_inputs 

def save_args(args: dict, save_path: str):
    txt = ''
    for k in args.keys():
        txt += ('{} {}\n'.format(k, args[k]))
    filename = '{}/hypers.txt'.format(save_path)
    with open(filename, 'w') as f:
        f.write(txt)
    filename = '{}/hypers.pkl'.format(save_path)
    with open(filename, 'wb') as f:
        pickle.dump(args, f)


def infer_latest_epoch(run_path: str) -> int:
    chkpts = glob.glob(os.path.join(run_path, 'model_*.pt'))
    assert chkpts, f"No model found in {run_path}"

    latest = 0
    for chkpt in chkpts:
        epoch = int(chkpt.split('_')[-1].strip('.pt'))
        if latest < epoch:
            latest = epoch
    return latest