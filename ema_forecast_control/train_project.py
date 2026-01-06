from typing import Optional
import os
import time
import shutil
import argparse
import pandas as pd

from ema_forecast_control import ROOT
import ema_forecast_control.utils.multitasking_utils as multitasking_utils
import ema_forecast_control.utils.training_utils as training_utils

def train_project(project_name: str, n_runs: int=1, include_participants: Optional[list]=None,
                  n_processes: int=1, n_proc_per_gpu: int=1, verbose: str='print'):
    
    experiment_path = os.path.join(ROOT, 'trained_models', project_name)
    if os.path.exists(experiment_path):
        answer = input(f'Experiment path {experiment_path} already exists. Delete/Overwrite/Update/Abort? (d/o/u/a)')
        if answer=='d':
            shutil.rmtree(experiment_path)
        elif answer=='o':
            configs = multitasking_utils.add_argument_to_configs(configs, 'overwrite', [1], append_to_name=False)
        elif answer=='u':
            configs = multitasking_utils.add_argument_to_configs(configs, 'overwrite', [0], append_to_name=False)
        else:
            return

    project_dict = training_utils.get_project_dict(project_name)    
    project_dict['n_processes'] = n_processes
    project_dict['n_proc_per_gpu'] = n_proc_per_gpu
    project_dict['verbose'] = verbose
    configs, n_workers = multitasking_utils.prepare_configuration_batch(project_dict)
    configs = multitasking_utils.add_data_to_configs(configs, project_dict['data_directory'], project_dict.get('train_test_split', None), include_participants=include_participants)
    configs = multitasking_utils.add_argument_to_configs(configs, 'run', list(range(1, 1 + n_runs)))    
        
    print(f'Running {len(configs)} jobs, {n_workers} in parallel. Proceeds in 10 seconds.')
    time.sleep(1)
    os.makedirs(experiment_path, exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, 'projects', f'{project_name}.yml'), os.path.join(experiment_path, f'{project_name}.yml'))
    multitasking_utils.run_batch(configs, n_workers)

def get_args() -> dict:
    parser = argparse.ArgumentParser(description='Multitasked model training based on project yml file.')

    parser.add_argument('project_name', type=str, nargs='?', default='test_project_kalman')
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--include_participants', type=int, nargs='+', default=[12])
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--n_proc_per_gpu', type=int, default=1)
    parser.add_argument('--verbose', type=str, choices=['none','print','log'], default='none')

    args = vars(parser.parse_args())
    return args
    

if __name__ == '__main__':
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    args = get_args()
    
    train_project(args['project_name'], n_runs=args['n_runs'], include_participants=args['include_participants'],
                    n_processes=args['n_processes'], n_proc_per_gpu=args['n_proc_per_gpu'], verbose=args['verbose'])