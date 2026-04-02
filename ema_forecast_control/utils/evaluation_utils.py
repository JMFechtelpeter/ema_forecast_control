import os
import re
import glob
from typing import Optional, Any
from tqdm import tqdm
import pandas as pd
import torch as tc

from ema_forecast_control.plrnn.plrnn_model import PLRNN
from ema_forecast_control.transformer.autoregressive_transformer_model import AutoregressiveTransformer
from ema_forecast_control.kalman_filter.kalman_filter_model import KalmanFilter
from ema_forecast_control.simple_models.simple_models import SimpleModel
from ema_forecast_control.utils import path_utils, backwards_compatibility
from ema_forecast_control.preprocessing import preprocessing, prepare_timestamp, train_test_split

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

def prepare_data_for_model_evaluation(model_dir: str, allow_test_inputs: bool=False,
                        with_args: Optional[dict]=None,
                        preloaded_data: Optional[pd.DataFrame]=None, preloaded_test_data: Optional[pd.DataFrame]=None,
                ):
    """
    Returns train and test data/input tensors for model evaluation. Usually, it loads the data according to the model args.
    If preloaded train and test data exist, you can pass them as DataFrames. Then the data will not be loaded again.
    This is handy if you want to evaluate several models on the same data, saving data loading time.
    Specify a test_data_dir if you want the test data to be loaded from somewhere else,
    e.g. if you preprocessed differently to the train set.
    """
   
    args = path_utils.load_args(model_dir)
    if with_args is not None:
        args.update(with_args)
    
    if preloaded_data is None:
        preloaded_data = pd.read_csv(args['data_path'])
    
    preloaded_data = prepare_timestamp.prepare_timestamp(preloaded_data, **args['timestamp'])
    train_df, test_df = train_test_split.train_test_split(preloaded_data, args['train_test_split'])
    train_df = preprocessing.preprocessing_pipeline(train_df, args['preprocessing'])

    train_data = tc.tensor(train_df[args['obs_features']].values).float()
    test_data = tc.tensor(test_df[args['obs_features']].values).float()
    if args['input_features'] is not None:
        train_inputs = tc.tensor(train_df[args['input_features']].values).float()
        test_inputs = tc.tensor(test_df[args['input_features']].values).float()
    else:
        train_inputs = None
        test_inputs = None
    
    if not allow_test_inputs:
        if test_inputs is not None:
            test_inputs[:] = 0

    return train_data, train_inputs, test_data, test_inputs


def get_model_class(args):
    if args['model'].upper() == 'PLRNN':
        model_class = PLRNN
    elif args['model'].upper() == 'KALMANFILTER':
        model_class = KalmanFilter
    elif args['model'].upper() == 'TRANSFORMER':
        model_class = AutoregressiveTransformer
    else:
        model_class = simple_models.get_class(args['model'])
    return model_class

def init_model_from_path(model_dir: str, with_args: Optional[dict]=None, select_epoch: Optional[int|str]=None) -> Any:

    def choose_epoch():
        if isinstance(select_epoch, int):
            epoch = select_epoch
        else:
            epochs = available_epochs()
            if len(epochs) == 0:
                raise ValueError(f'No model checkpoints found in {model_dir}.')
            if select_epoch == 'loss':
                if os.path.exists(os.path.join(model_dir, 'loss.csv')):
                    loss = pd.read_csv(os.path.join(model_dir, 'loss.csv'))
                    loss = loss[[(e in epochs) for e in loss['epoch']]]
                    epoch = loss['epoch'][loss['epoch_loss'].argmin()]
                else:
                    raise FileNotFoundError(
                        f'No loss.csv file in model folder {model_dir}, cannot pick epoch by lowest loss.')
            elif select_epoch == 'complete':
                if os.path.exists(os.path.join(model_dir, f"model_{args['n_epochs']}.pt")):
                    epoch = args['n_epochs']
                else:
                    return None, None, None
            elif select_epoch == 'latest':
                epoch = max(epochs)
            else:
                epoch = None
        return epoch

    def available_epochs():
        epochs = []
        for f in os.listdir(model_dir):
            match = re.match(r'model_([0-9]+).pt', f)
            if match is not None:
                epochs.append(int(match.group(1)))
        return sorted(set(epochs))
    
    args = path_utils.load_args(model_dir)
    if with_args is not None:
        args.update(with_args)
    model = get_model_class(args)(args)
    model.init_from_model_path(model_dir, choose_epoch())
    model.args['model_id'] = os.path.split(os.path.split(model_dir)[0])[1]
    return model


def get_Gamma(model: PLRNN|KalmanFilter, model_dir: str) -> Optional[tc.Tensor]:
    if isinstance(model, PLRNN):
        return tc.load(os.path.join(model_dir, 'empirical_covariance.pt'))
    elif isinstance(model, KalmanFilter):
        return model.params['Gamma']
    else:
        return None

def include_exclude_hypers(args: dict, include_hypers: Optional[dict]=None, exclude_hypers: Optional[dict]=None) -> bool:
    if include_hypers is not None:
        for hyper, values in include_hypers.items():
            if args[hyper] not in values:
                return False
    if exclude_hypers is not None:
        for hyper, values in exclude_hypers.items():
            if args[hyper] in values:
                return False
    return True

def complement_args_with_data_info(args: dict, train_data: tc.Tensor) -> dict:
    ''' Complements the model training args with entries 'valid_training_data_points' and 'valid_training_data_ratio' '''
    nans = train_data.isnan().all(dim=1).sum()
    valid = (train_data.shape[0] - nans).item()
    args['valid_time_points'] = valid
    args['valid_ratio'] = valid / train_data.shape[0]
    return args

def get_training_time(model_dir):

    if os.path.exists(os.path.join(model_dir, 'log.txt')):
        with open(os.path.join(model_dir, 'log.txt'), 'r') as file:
            log_record = file.readlines()       
        start_time = pd.Timestamp(re.match('(.*?),', log_record[0]).group(1))
        end_time = pd.Timestamp(re.match('(.*?),', log_record[-1]).group(1))
        training_time = (end_time - start_time).total_seconds()
    elif os.path.exists(os.path.join(model_dir, 'hypers.txt')) and len(glob.glob(os.path.join(model_dir, '*.pt')))>0:
        start_time = os.path.getmtime(os.path.join(model_dir, 'hypers.txt'))
        end_time = max([os.path.getmtime(model_file) for model_file in glob.glob(os.path.join(model_dir, '*.pt'))])
        training_time = end_time - start_time
    else:
        training_time = None
    return training_time

def get_number_of_params(model: PLRNN|AutoregressiveTransformer|KalmanFilter|SimpleModel) -> int:
    """ Returns the total number of parameters of the model. """
    n_params = 0
    for p in model.get_parameters().values():
        n_params += p.numel()
    return n_params

def get_model_folders(project_name: str) -> list[str]:
    """    Returns all subdirs of main_dir which contain at least one *.pt file.    """
    main_dir = path_utils.join_base_path('trained_models', project_name)
    if not os.path.exists(main_dir):
        raise FileNotFoundError(f'{main_dir} not found')
    models = glob.glob(os.path.join(main_dir, '**', '*.pt'), recursive=True)
    folders = [os.path.split(m)[0] for m in models]
    folders = sorted(set(folders))
    return folders

def preload_data(model_dir_paths: list, load_test_data_from: str|None=None, use_tqdm: bool=False, hierarchized: bool=False):
    """
    Returns all subdirs of main_dir which contain at least one *.pt file.
    Loads the csv datasets that the model args refer to, and returns a mapping dict of model folder -> dataframe.
    If load_test_data_from is not None, additionally loads the corresponding csv datasets from there and returns another mapping dict.
    """
    preloaded_dataframes = {}
    test_data_mapping = {}
    if use_tqdm:
        iterator = tqdm(model_dir_paths, desc='Preloading data for models')
    else:
        iterator = model_dir_paths

    for model_dir in iterator:

        if load_test_data_from is not None:
            test_data_path = os.path.join(load_test_data_from, os.path.split(train_data_path)[1])
        else:
            test_data_path = backwards_compatibility.update_data_path(path_utils.load_args(model_dir))['data_path']
        if test_data_path not in preloaded_dataframes.keys():
            preloaded_dataframes[test_data_path] = pd.read_csv(test_data_path)
        test_data_mapping[model_dir] = preloaded_dataframes[test_data_path]

    return test_data_mapping

def create_ensemble_prediction_eval_df(eval_df: pd.DataFrame, outlier_threshold: Optional[float]=None, sort_result: bool=False) -> pd.DataFrame:

    eval_df = eval_df.copy()
    if outlier_threshold is not None:
        exclude_model_ids = eval_df.loc[eval_df['prediction'].abs()>outlier_threshold, 'model_id'].unique()
        eval_df = eval_df.loc[~eval_df['model_id'].isin(exclude_model_ids)]
    ensemble_defining_cols = ['model_id', 'feature', 'steps', 'sample', 'prewarm_steps', 'test_day']
    agg_funcs = {col: 'mean' if pd.api.types.is_numeric_dtype(eval_df[col]) else 'first' for col in eval_df.columns}
    ensemble_eval_df = eval_df.groupby(ensemble_defining_cols, as_index=False, dropna=False, sort=sort_result).agg(agg_funcs)   # CAUTION: sort_result=True results in buggy change prediction
    return ensemble_eval_df