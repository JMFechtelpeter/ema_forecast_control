import multiprocessing as mp
import subprocess
import os
from typing import List, Tuple, Optional
import copy
import logging
from logging.handlers import QueueHandler, QueueListener
import numpy as np
import torch as tc
import pandas as pd

from ema_forecast_control.utils import path_utils, data_utils
from ema_forecast_control.plrnn.train_plrnn import train_plrnn
from ema_forecast_control.transformer.train_transformer import train_transformer
from ema_forecast_control.kalman_filter.train_kalman_filter import train_kalman_filter
from ema_forecast_control.simple_models.train_simple_models import train_simple_model

def select_trainer_function(model_type: str):
    if model_type.upper() == 'PLRNN':
        return train_plrnn
    elif model_type.upper() == 'TRANSFORMER':
        return train_transformer
    elif model_type.upper() == 'KALMANFILTER':
        return train_kalman_filter
    else:
        return train_simple_model

def add_argument_to_configs(configs: list[dict], arg_name: str, arg_values: list, append_to_name: bool=True) -> list[dict]:
    new_configs = []
    for config in configs:
        for v in arg_values:
            this_config = copy.deepcopy(config)
            if append_to_name or len(arg_values)>1:
                new_config_name = append_to_config_name(config['configuration_name'], arg_name, v)
            else:
                new_config_name = config['configuration_name']
            this_config.update({'configuration_name': new_config_name, arg_name: v})
            new_configs.append(this_config)
            
    return new_configs

def add_combined_arguments_to_configs(configs: list[dict], combined_args: list[dict], append_to_name: bool=True) -> list[dict]:
    if len(combined_args) == 0:
        return configs
    new_configs = []
    last_argument_names = None
    for arg_combination in combined_args:
        if last_argument_names is None:
            last_argument_names = set(arg_combination.keys())
        else:
            assert last_argument_names == set(arg_combination.keys()), "All combined hyperparameter definitions must have the same keys."
        these_configs = copy.deepcopy(configs)
        for c in these_configs:
            for arg, v in arg_combination.items():
                if append_to_name or len(combined_args)>1:
                    new_config_name = append_to_config_name(c['configuration_name'], arg, v)
                else:
                    new_config_name = c['configuration_name']
                c.update({'configuration_name': new_config_name, arg: v})
        new_configs.extend(these_configs)
    return new_configs

def add_data_to_configs(configs: list[dict], data_directory: str, train_test_split: Optional[str|int|list], include_participants: Optional[list]=None) -> list[dict]:    
    data_files = path_utils.get_data_files(data_directory)
    if include_participants is not None:
        data_files, _ = filter_participants(data_files, include_participants)

    if isinstance(train_test_split, str) and os.path.isfile(train_test_split):
        test_split_df = pd.read_csv(train_test_split, index_col=0)
        use_participants = test_split_df.columns.to_list()
        data_files, participants = filter_participants(data_files, use_participants)
        
        new_configs = []
        for f, p in zip(data_files, participants):
            new_config = add_combined_arguments_to_configs(configs, {'data_path': f})
            new_config = add_argument_to_configs(new_config, 'train_test_split', test_split_df[p].tolist())
            new_configs.extend(new_config)

    elif isinstance(train_test_split, list):
        new_configs = add_argument_to_configs(configs, 'data_path', data_files)
        new_configs = add_argument_to_configs(new_configs, 'train_test_split', train_test_split)
        
    else:
        new_configs = add_argument_to_configs(configs, 'data_path', data_files)
        if train_test_split is not None:
            new_configs = add_argument_to_configs(new_configs, 'train_test_split', [train_test_split])

    return new_configs

    
def append_to_config_name(config_name, arg_name, arg_value):
    if arg_name == 'data_path':
        arg_value = os.path.split(arg_value)[1].zfill(2)
    else:
        arg_value = str(arg_value).zfill(2)
    if len(config_name)==0:
        new_name = "_".join([arg_name, arg_value])
    else:
        new_name = "_".join([config_name, arg_name, arg_value])
    return new_name

def filter_participants(data_files: list[str], participants: Optional[list]) -> list:
    filtered_data_files = []
    used_participants = []
    if participants is not None:
        participants = [str(p) for p in participants]
    for file in data_files:        
        participant = data_utils.determine_participant_id(data_path=file)
        if participants is not None:
            if participant in participants:
                filtered_data_files.append(file)
                used_participants.append(participant)
        else:
            filtered_data_files.append(file)
            used_participants.append(participant)
    return filtered_data_files, used_participants


def check_need_to_assign_gpus(config: dict) -> bool:
    '''
    Check if user wants to use GPUs and has not selected device id manually.
    '''
    # if the user specifies device ids themselves, don't bother distributing the tasks.
    if config.get('device_id', None) is not None:
        assign_gpus = False
        print("Device id(s) specified by user -> manual task distribution")
    elif config.get('use_gpu', 0) == 1:
        assert tc.cuda.is_available(),  "CUDA is not available."
        print("Will distribute tasks to GPUs automatically.")
        assign_gpus = True
    else:
        print("Not using GPUs for training.")
        assign_gpus = False
    return assign_gpus


def assign_gpus_to_tasks(configs: list[dict], n_proc_per_gpu: int, n_workers: int) -> Tuple[list, int]:
    '''
    Checks current GPU utilization of the machine, picks out idle devices and distributes them across tasks.
    '''
    def get_current_gpu_utilization() -> dict:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=utilization.gpu',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8'
        )

        # Convert lines into a dictionary
        gpu_util = [int(x) for x in result.strip().split('\n')]
        return dict(zip(range(len(gpu_util)), gpu_util))
    
    util_dict = get_current_gpu_utilization()
    # filter device ids of unused GPUs
    available_device_ids = []
    for id_, util in util_dict.items():
        if util < 75:
            available_device_ids.append(id_)

    if not available_device_ids:
        raise RuntimeError("All GPUs of the machine are in use!")

    # check if there are too many parallel processes spawned by user compared to available GPUs
    device_distribution = np.repeat(available_device_ids, min(n_workers, n_proc_per_gpu))
    n_assigned_workers = device_distribution.size
    if n_assigned_workers < n_workers:
        print(f"There are not enough GPU Resources available to spawn {n_workers} processes. Reducing number of parallel runs to {n_assigned_workers}")
        new_n_workers = n_assigned_workers
    else:
        new_n_workers = n_workers

    # distribute devices across tasks
    new_tasks = []
    idx = 0
    for config in configs:
        config['device_id'] = device_distribution[idx]
        idx += 1
        if idx == new_n_workers:
            idx = 0
    return configs, new_n_workers

def create_batch_configs(project_dict: dict) -> Tuple[List, int]:

    pdict_copy = copy.deepcopy(project_dict)

    hypers = pdict_copy.get('hyperparameters', {})
    combined_hypers = pdict_copy.get('combined_hyperparameters', [])

    pdict_copy['configuration_name'] = ''

    pdict_copy.pop('hyperparameters', None)
    pdict_copy.pop('combined_hyperparameters', None)
    configs = [pdict_copy]

    for hyper_name, hyper_values in hypers.items():
        configs = add_argument_to_configs(configs, hyper_name, hyper_values)
    
    configs = add_combined_arguments_to_configs(configs, combined_hypers)

    for k, config in enumerate(configs):
        if "pbar_descr" in config:
            config['pbar_descr'] = f'{config.get("pbar_descr", "")}; Job {k}/{len(configs)}'
        else:
            config['pbar_descr'] = f'Job {k}/{len(configs)}'
        
    return configs, len(configs)

def prepare_configuration_batch(project_dict: dict):

    assign_gpus = check_need_to_assign_gpus(project_dict)
    configs, n_jobs = create_batch_configs(project_dict)
    if assign_gpus:
        configs, n_workers = assign_gpus_to_tasks(configs, project_dict.get('n_proc_per_gpu', 1), project_dict.get('n_processes', 1))
    else:
        n_workers = project_dict.get('n_processes', 1)
    return configs, n_workers


#logging inspired by https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python

def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)

def logger_init():
    q = mp.Queue()
    # this is the handler for all log records
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, handler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)

    return ql, q


def run_batch(configs: list[dict], n_cpu: int):
    # q_listener, q = logger_init()
    # logging.info('Running %i jobs, %i in parallel.', len(tasks), n_cpu)
    # pool = mp.Pool(processes=n_cpu, initializer=worker_init, initargs=[q])
    mp.set_start_method("spawn")
    pool = mp.Pool(processes=n_cpu)
    pool.map(single_model_trainer, configs, chunksize=1)
    pool.close()
    pool.join()
    # q_listener.stop()

def single_model_trainer(config: dict):
    trainer = select_trainer_function(config['model'])
    trainer(config)