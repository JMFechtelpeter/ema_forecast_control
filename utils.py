import os
import glob
import pickle
import json
import subprocess
import shutil
from tqdm import tqdm
from argparse import Namespace
from typing import Optional, Any
import numpy as np
import pandas as pd
import torch as tc
from torch import nn
from tensorboardX import SummaryWriter
import pynvml
import thop     # for MAC counting
from fvcore.nn import FlopCountAnalysis

from dataset.multimodal_dataset import MultimodalDataset
import data_utils



import logging
log = logging.getLogger(__name__)


def prepare_device(args: Namespace) -> tc.device:
    '''
    Prepare and return a torch.device instance.
    '''
    # check cuda availability
    if args.use_gpu and tc.cuda.is_available():
        id_ = args.device_id
        if id_ == -1:
            id_ = find_gpu_with_lowest_utilization(criterion='load')
        device = tc.device('cuda', id_)
        name = tc.cuda.get_device_name(id_)
        # if args.verbose == 'print':
        #     print(f"Using device {name} for training ({device}).")
        # elif args.verbose == 'log':
        log.info("Using device %s for training (%s).", name, device)
    else:
        device = tc.device('cpu')
        num_threads = tc.get_num_threads()
        # if args.verbose == 'print':
        #     print(f"Training on {device} with {num_threads} threads.")
        # elif args.verbose == 'log':
        log.info("Training on %s with %i threads.", str(device), num_threads)
    return device

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
        

def get_runs(trial_path: str) -> list:
    try:
        run_nrs = [d for d in os.listdir(trial_path) if os.path.isdir(os.path.join(trial_path, d)) and d.isdigit()]
        return run_nrs
    except FileNotFoundError:
        return []

def infer_latest_epoch(run_path: str) -> int:
    '''
    Loop over model_*.pt checkpoints and find the 
    model checkpoint of the latest epoch.
    '''
    # list checkpoints
    chkpts = glob.glob(os.path.join(run_path, 'model_*.pt'))
    assert chkpts, f"No model found in {run_path}"

    # find latest epoch
    latest = 0
    for chkpt in chkpts:
        epoch = int(chkpt.split('_')[-1].strip('.pt'))
        if latest < epoch:
            latest = epoch
    return latest

def next_run(trial_path: str) -> str:
    """increase by one each run, if none exists start at '001' """
    run_nrs = get_runs(trial_path)
    if not run_nrs:
        run_nrs = ['000']
    run = str(int(max(run_nrs)) + 1).zfill(3)
    run_dir = os.path.join(trial_path, run)
    return run_dir

def get_experiment_path(results_folder: str, experiment: str) -> str:
    save_path = os.path.join(results_folder, experiment)
    save_path = data_utils.join_ordinal_bptt_path(save_path)
    return save_path
 
def create_savepath(args: Namespace) -> str:
    save_path = get_experiment_path(args.results_folder, args.experiment)
    trial_path = os.path.join(save_path, args.name)
    if args.run is None:
        run_path = next_run(trial_path)
    else:
        run_path = os.path.join(trial_path, str(args.run).zfill(3))
    os.makedirs(run_path, exist_ok=args.overwrite)
    return run_path

# def init_writer(args, trial_path):
#     writer = None
#     if args.use_tb:
#         # if args.verbose=='print':
#         #     print('Initialized tensorboard writer at {}'.format(trial_path))
#         # elif args.verbose=='log':
#         log.info('Initialized tensorboard writer at %s', trial_path)
#         writer = SummaryWriter(trial_path)
#     return writer

def read_numpy_data(data_path: Optional[str]) -> np.ndarray|None:
    if data_path is None:
        data = None
    else:
        assert os.path.exists(data_path)
        data = np.load(data_path, allow_pickle=True)
    return data

def create_dataset_from_numpy_data(args: Namespace|dict, data_path: Optional[str]=None) -> tuple[Namespace, MultimodalDataset]:
            
    if isinstance(args, dict):
        args = Namespace(**args)  
    if data_path is not None:
        args.data_path = data_path      
    data = read_numpy_data(args.data_path)
    inputs = read_numpy_data(args.inputs_path)
    dataset = MultimodalDataset(args.seq_len, args.batch_size,
                                     args.batches_per_epoch, ignore_leading_nans=True,
                                     return_single_tensor=False,
                                     tolerate_partial_missings=False)
    dataset.add_timeseries(data, name='emas',
                           feature_names=args.data_features)
    dataset.add_timeseries(inputs, name='inputs',
                           feature_names=args.input_features)
    
    args.dim_x = dataset.timeseries['emas'].dim
    if inputs is not None:
        args.dim_s = dataset.timeseries['inputs'].dim
    else:
        args.dim_s = None
    return args, dataset   


def complement_args(args: dict) -> dict:
    if 'dim_y' not in args.keys():
        args['dim_y'] = None
    if 'nonlinear_units' not in args.keys():
        args['nonlinear_units'] = None
    if 'participant' not in args.keys():
        args['participant'] = None
    if 'adaptive_alpha_rate' not in args.keys():
        args['adaptive_alpha_rate'] = 0
    if 'validation_len' not in args.keys():
        args['validation_len'] = 6
    if 'lr_annealing' not in args.keys():
        args['lr_annealing'] = 0
    if 'dim_x_proj' not in args.keys():
        args['dim_x_proj'] = 0
    if 'early_stopping' not in args.keys():
        args['early_stopping'] = 0
    if 'pbar_descr' not in args.keys():
        args['pbar_descr'] = ''
    if 'obs_features' not in args.keys():
        args['obs_features'] = args['data_features']
    if 'random_z0' not in args.keys():
        args['random_z0'] = 0
    return args

def load_args(model_path: str) -> dict:
    args_path = os.path.join(model_path, 'hypers.pkl')
    args = np.load(args_path, allow_pickle=True)
    args = complement_args(args)
    return args

def save_args(args: Namespace|dict, save_path: str, writer: Optional[SummaryWriter]=None):
    """ add hyperparameters to txt file """
    if isinstance(args, Namespace):
        d = args.__dict__
    else:
        d = args
    txt = ''
    for k in d.keys():
        txt += ('{} {}\n'.format(k, d[k]))
    if writer is not None:
        writer.add_text(tag="""hypers""", text_string=str(txt), global_step=None)
    filename = '{}/hypers.txt'.format(save_path)
    with open(filename, 'w') as f:
        f.write(txt)
    filename = '{}/hypers.pkl'.format(save_path)
    with open(filename, 'wb') as f:
        pickle.dump(d, f)

def check_args(args: Namespace|dict):
    def assert_positive_or_none(arg):
        if arg is not None:
            assert arg > 0
    if isinstance(args, dict):
        args = Namespace(**args)
    assert args.data_path is not None
    assert_positive_or_none(args.clip_range)
    assert args.dim_z > 0
    assert args.learning_rate > 0
    assert args.n_epochs > 0
    assert args.dim_z >= args.dim_x or args.dim_z >= args.dim_x_proj
    # assert args.alpha_reg >= 0
    # assert args.n_states_reg >= 0
    # list entries are tuples of floats
    # first entry is between 0 and 1 and sum of all is not higher than one
    # but higher than 0
    # second entry is > 0


def save_files(save_path: str):
    curdir = os.path.abspath('.')
    from distutils.dir_util import copy_tree
    save_path = os.path.join(save_path, 'python_files')
    copy_tree(curdir, save_path)

def save_to_pickle(variable: Any, file_name: str):
    filename = '{}.pkl'.format(file_name)
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)

def load_from_pickle(file_name: str) -> Any:
    filename = '{}.pkl'.format(file_name)
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    return variable

def get_current_gpu_utilization() -> dict:
    """
    From: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are GPU utilization in %.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8'
    )

    # Convert lines into a dictionary
    gpu_util = [int(x) for x in result.strip().split('\n')]
    return dict(zip(range(len(gpu_util)), gpu_util))

def list_files(startpath: Optional[str]=None, jsonify: bool=False):
    # res = ''
    res = {}
    if startpath is None:
        startpath = os.getcwd()
    root_key = startpath.split('/')[-1]
    root_level = len(startpath.split('/'))
    for root, dirs, files in os.walk(startpath):
        current_dict = res
        relative_path = root.split('/')[root_level-1:]
        for d in relative_path:
            try:
                current_dict = current_dict[d]
            except:
                current_dict[d] = {}
                current_dict = current_dict[d]
        for f in files:
            current_dict[f] = ''
    if jsonify:
        res = json.dumps(res, indent=4)
        # level = root.replace(startpath, '').count(os.sep)
        # indent = ' ' * 4 * (level)
        # res += f'{indent}{os.path.basename(root)}/\n'
        # subindent = ' ' * 4 * (level + 1)
        # for f in files:
        #     res += f'{subindent}{f}\n'
    return res

def determine_best_run(runs_path: str) -> str:
    losses = {}
    for model_dir in os.listdir(runs_path):
        try:
            loss = pd.read_csv(os.path.join(runs_path, model_dir,'loss.csv'))
            if (loss['validation_loss']==0).all():
                losses[1000] = model_dir
            else:
                losses[loss['validation_loss'].min()] = model_dir
        except:
            pass
    min_losses = min(losses.keys())
    best_run = losses[min_losses]
    return best_run

def extract_best_runs(experiment_path: str):
    
    models = glob.glob(str(os.path.join(experiment_path, '*/*/*.pt')))
    name_dirs = sorted(set([m.split('/')[-3] for m in models]))
    target_experiment_path = experiment_path.rstrip('/') + '_best_runs'
    for name_dir in tqdm(name_dirs, total=len(name_dirs), desc='Extracting best runs'):
        name_path = os.path.join(experiment_path, name_dir)
        best_run = determine_best_run(name_path)
        best_run_path = os.path.join(experiment_path, name_dir, best_run)
        target_path = os.path.join(target_experiment_path, name_dir, best_run)
        shutil.copytree(best_run_path, target_path, dirs_exist_ok=True)

def get_model_dirs(main_dir: str) -> list:
    model_paths = glob.glob(os.path.join(main_dir, '**/model*.pt'), recursive=True)
    return sorted(set([os.path.split(p)[0] for p in model_paths]))


def filter_model_dirs_by_hyperparameters(main_dir: str, **params) -> list:
    model_dirs = get_model_dirs(main_dir)
    filtered_dirs = []
    for d in model_dirs:
        args = load_args(d)
        include = True
        for p in params:
            if args[p]!=params[p]:
                include = False
        if include:
            filtered_dirs.append(d)
    return filtered_dirs

def available_epochs(folder: str) -> list:
    models = [f for f in os.listdir(folder) if f.endswith('.pt')]
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in models]
    epochs = sorted(set(epochs))
    return epochs

def nanstd(x: tc.Tensor, dim: Optional[tuple|int]=None, keepdim: bool=False) -> tc.Tensor:
    ss = (x - tc.nanmean(x, dim=dim, keepdim=True))**2
    return tc.sqrt(tc.nanmean(ss, dim=dim, keepdim=keepdim))

def compute_total_grad_norm(model: nn.Module, norm_type: int=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)  # Final norm calculation

def count_flops(model: nn.Module, inputs: tuple) -> int:
    """
    Count FLOPs of a model forward pass using fvcore.
    Args:
        model (nn.Module): The model to count FLOPs for.
        inputs (tuple): A tuple of inputs to the model.
    Returns:
        int: The total number of FLOPs.
    """
    flops = FlopCountAnalysis(model, inputs)
    return flops.total()

def count_macs(model: nn.Module, inputs: tuple) -> int:
    """
    Count MACs of a model forward pass using thop.
    Args:
        model (nn.Module): The model to count MACs for.
        inputs (tuple): A tuple of inputs to the model.
    Returns:
        int: The total number of MACs.
    """
    macs, params = thop.profile(model, inputs=inputs, verbose=False)
    return thop.clever_format([macs, params], "%.3f")