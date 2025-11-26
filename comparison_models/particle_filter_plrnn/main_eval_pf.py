#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:15:47 2022

@author: janik
"""

import logging
import os
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as tc

from pf_plrnn import PF_PLRNN
from dataset.multimodal_dataset import MultimodalDataset
import eval_reallabor_utils
import utils
import data_utils

def configure_logging(verbose: str, path: Optional[str]=None):   
    if path is None:
        path = os.getcwd() 
    logger = logging.getLogger('root')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    if verbose=='log':
        hdlr = logging.FileHandler(os.path.join(path, 'log.txt'))
    elif verbose=='print':
        hdlr = logging.StreamHandler()
    else:
        hdlr = logging.NullHandler()
    hdlr.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger

def _init_worker(threads_per_worker: int = 1):
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads_per_worker)
    try:
        import torch as _tc
        _tc.set_num_threads(threads_per_worker)
        _tc.set_num_interop_threads(1)
    except Exception:
        pass

@dataclass
class EvaluationArgs:

    main_dir: str
    test_data_dir: str
    epoch_criterion: str
    hyperparameters: list[str]=field(default_factory=list)
    random_z0: bool=False
    ahead_prediction_steps: Optional[int]=None
    trajectory_samples: int=5
    prewarm_steps_on_train_set: int|list=2
    allow_test_inputs: bool=False
    test_split_file: Optional[str]=None
    test_split_df: Optional[pd.DataFrame]=None
    test_set_offset_days: int=0
    first_alarms_file: Optional[str]=None
    first_alarms_df: Optional[pd.DataFrame]=None
    include_hypers: dict=field(default_factory=dict)
    exclude_hypers: dict=field(default_factory=dict)
    calculate_max_lyapunov: bool=False
    get_training_time: bool=True
    get_number_of_params: bool=True
    ensemble_predictors: bool=False
    buggy_version_of_ensemble: bool=False
    observation_model_version: str='empirical covariance'   # Options: 'pseudoinverse', 'empirical covariance', 'gaussian obs model'
    label: str=''
    overwrite: bool=False
    file_format: str='csv'
    verbose: str='none'
    preload_data: bool=True
    num_workers: int=1

def evaluate_complete_directory(eval_args: EvaluationArgs):
    """
    Evaluates all models found in main_dir on test data. Saves results in csv file.
    From each run, a model from one epoch ist evaluated, chosen according to epoch_criterion (<epoch number>, "complete", "latest", or "loss")
    - <epoch_number> -> choose model from specific epoch
    - "complete" -> choose epoch that is equal to args.n_epochs, so the run is not evaluated if incomplete
    - "latest" -> choose latest available epoch
    - "loss" -> choose epoch with lowest loss
    Further Arguments:
    - test_data_dir: load the test data from here.
    - hyperparameters, ahead_prediction_steps, trajectory_samples, prewarm_steps_on_train_set: 
        arguments passed to evaluate_model_on_dataset function
    - include_hypers: dict; include only models in evaluation that have specified hyperparameter values (e.g. {'participant':15})
    - exclude_hypers: dict; exclude models from evaluation that have specified hyperparameter values 
    - create_best_runs_file: if True, extracts best run of each set of runs and saves its eval results in a separate file
    - label: append this to the evaluation file name, so you can have multiple eval files for a set of models
    - overwrite: overwrite existing evaluation files with the same name    
    """
    assert eval_args.file_format in ['csv', 'json'], 'Choose either csv or json as file format'
    assert os.path.exists(eval_args.main_dir), f'Directory {eval_args.main_dir} does not exist'
    if eval_args.preload_data:
        model_dir_paths, train_data_mapping, test_data_mapping = eval_reallabor_utils.get_model_folders_and_preload_data(eval_args.main_dir, eval_args.test_data_dir, use_tqdm=True)
    else:
        model_dir_paths = eval_reallabor_utils.get_model_folders(eval_args.main_dir)
        train_data_mapping = {k: None for k in model_dir_paths}
        test_data_mapping = {k: None for k in model_dir_paths}
    assert len(model_dir_paths) > 0,'Given main_dir path does not contain models.'

    # create folder to store summary plots and metrics
    summary_path = os.path.join(eval_args.main_dir, f'00_summary_{eval_args.label}')
    try:
        os.makedirs(summary_path, exist_ok=False)
    except:
        if not eval_args.overwrite:
            decision = input(f'An evaluation path with label "{eval_args.label}" already exists. Overwrite? ')
            if decision not in ['y', 'Y', '1']:
                return None
    log = configure_logging(eval_args.verbose, path=summary_path)
    print(f'Evaluating directory {eval_args.main_dir}, label {eval_args.label}')
    log.info(f'Evaluating directory {eval_args.main_dir}, label {eval_args.label}')
    for arg in eval_args.__dict__.keys():
        log.info(f'{arg}: {eval_args.__dict__[arg]}')

    if eval_args.test_split_file is not None:
        eval_args.test_split_df = pd.read_csv(eval_args.test_split_file, index_col=0)
    elif eval_args.first_alarms_file is not None:
        eval_args.first_alarms_df = pd.read_csv(eval_args.first_alarms_file)
    results = []
    n_errors = 0

    if eval_args.num_workers > 1:
        # Choose backend
        from multiprocessing import get_context
        mp_ctx = get_context('spawn')  # safer with MKL/NumPy
        Executor = ProcessPoolExecutor
        executor_kwargs = dict(max_workers=eval_args.num_workers,
                                mp_context=mp_ctx,
                                initializer=_init_worker,
                                initargs=(1,))

        with Executor(**executor_kwargs) as executor:
            futures = {}
            for folder in model_dir_paths:
                kwargs = {}
                kwargs['preloaded_train_data'] = train_data_mapping[folder]
                kwargs['preloaded_test_data'] = test_data_mapping[folder]
                futures[executor.submit(
                    evaluate_model_folder, folder, eval_args, **kwargs
                )] = folder

            for future in tqdm(as_completed(futures), total=len(model_dir_paths)):
                reserr = future.result()
                if len(reserr[0])>0:
                    results.append(reserr[0])
                if len(reserr[1])>0:
                    log.warning(f'There have been errors evaluating {reserr[2]}.')
                    for e in reserr[1]:
                        log.warning(f'{reserr[2]}: {e}')
                    n_errors += len(reserr[1])

    else:
        for folder in tqdm(model_dir_paths):        
            res, err, _ = evaluate_model_folder(folder, eval_args,
                                            preloaded_train_data=train_data_mapping[folder],
                                            preloaded_test_data=test_data_mapping[folder])
            if len(res)>0:
                results.append(res)
            if len(err)>0:
                log.warning(f'There have been errors evaluating {folder}.')
                for e in err:
                    log.warning(f'{folder}: {e}')
                n_errors += len(err)

    if len(results) > 0:
        results = pd.concat(results, ignore_index=True)
        if eval_args.ensemble_predictors:
            results = eval_reallabor_utils.create_ensemble_prediction_eval_df(results, sort_result=eval_args.buggy_version_of_ensemble)
        results.index.name = 'idx'
        if eval_args.file_format=='csv':
            results.to_csv(os.path.join(summary_path, 'evaluation.csv'))
        elif eval_args.file_format=='json':
            results.to_json(os.path.join(summary_path, 'evaluation.json'))
    else:
        log.error('No models were evaluated.')
    
    log.info(f'{n_errors} errors during data/model loading...')
    
    return results


def evaluate_model_folder(folder: str, eval_args: EvaluationArgs, preloaded_train_data: pd.DataFrame|None=None, preloaded_test_data: pd.DataFrame|None=None):
    
    log = logging.getLogger('evaluate_model_folder')
    errors = []
    results = []
    args = eval_reallabor_utils.update_data_path(utils.load_args(folder))
    test_args = eval_reallabor_utils.update_data_path(utils.load_args(folder))
    if eval_args.test_split_df is not None:    
        end_train_set = eval_args.test_split_df[args['participant']].dropna().to_numpy()
        begin_test_set = eval_args.test_split_df[args['participant']].dropna().to_numpy()
    elif eval_args.first_alarms_df is not None:
        train_dataset_name = os.path.split(os.path.split(args['data_path'])[0])[1]
        test_dataset_name = os.path.split(eval_args.test_data_dir)[1]
        participant = int(args['participant'])
        validation_split = int(float(args['train_on_data_until_timestep']))
        tf = eval_args.first_alarms_df[eval_args.first_alarms_df['participant']==participant]
        day_of_val_split = tf.loc[tf[train_dataset_name]==validation_split, 'day']
        if len(day_of_val_split)==1:
            day_of_val_split = day_of_val_split.item()
        else:
            errors.append(f'Validation index {day_of_val_split} is not among first alarms.')
            return [], errors
        if (tf['day']==day_of_val_split+eval_args.test_set_offset_days).any():
            end_train_set = [tf.loc[tf['day']==day_of_val_split+eval_args.test_set_offset_days, train_dataset_name].item()]
            begin_test_set = [tf.loc[tf['day']==day_of_val_split+eval_args.test_set_offset_days, test_dataset_name].item()]
        else:  # day_of_val_split + test_set_offset_days doesn't exist
            errors.append(f'Test index {day_of_val_split+eval_args.test_set_offset_days} is not among first alarms.')
            return [], errors
    else:
        end_train_set = [None]
        begin_test_set = [None]
    for ets, bts in zip(end_train_set, begin_test_set):
        if ets is not None:
            args['train_on_data_until_timestep'] = ets
            args['train_on_data_until_datetime'] = None
        if bts is not None:
            test_args['train_on_data_until_timestep'] = bts
            test_args['train_on_data_until_datetime'] = None
        # if preloaded_train_data is not None or preloaded_test_data is not None:
        try:
            model, train_dataset, test_dataset = eval_reallabor_utils.load_model_and_data(
                folder,
                epoch_criterion=eval_args.epoch_criterion, 
                allow_test_inputs=eval_args.allow_test_inputs,
                with_args=args,
                alternate_test_args=test_args,
                load_test_data_from=eval_args.test_data_dir,
                preloaded_train_data=preloaded_train_data, preloaded_test_data=preloaded_test_data,
                hierarchized=False)
        except Exception as e:
            errors.append(f'Test set beginning at {test_args["train_on_data_until_timestep"]}: {str(e)}')
            continue
        continue_ = False
        if test_dataset is None:
            continue_ = True
        for hyper, values in eval_args.include_hypers.items():
            if model.args[hyper] not in values:
                continue_ = True
                break
        if continue_:
            continue
        for hyper, values in eval_args.exclude_hypers.items():
            if model.args[hyper] in values:
                continue_ = True
                break
        if continue_:
            continue
        model.args = eval_reallabor_utils.complement_args_with_data_info(model.args, test_data_dir=eval_args.test_data_dir,
                                                                         preloaded_test_data=preloaded_test_data)  # Adds 'valid_training_data_points' and 'valid_training_data_ratio' to args
        try:
            res = evaluate_pf_plrnn_on_dataset(model, train_dataset, test_dataset, eval_args.hyperparameters, eval_args.ahead_prediction_steps,
                                            eval_args.trajectory_samples)
        except:
            errors.append(f'Test set beginning at {test_args["train_on_data_until_timestep"]}: Error during evaluation.')
            continue
        res['test_day'] = test_args['train_on_data_until_timestep']
        if eval_args.get_training_time:
            res['training_time'] = eval_reallabor_utils.get_training_time(folder)
        if eval_args.get_number_of_params:
            res['n_params'] = eval_reallabor_utils.get_number_of_params(model)
        res.reset_index(inplace=True)
        results.append(res)
    
    if len(results)>0:
        results = pd.concat(results, ignore_index=True)
        results.index.name = 'idx'
        
    return results, errors, folder


def evaluate_pf_plrnn_on_dataset(model: PF_PLRNN, train_dataset: MultimodalDataset, test_dataset: MultimodalDataset, 
                              hyperparameters: list, ahead_prediction_steps: Optional[int], trajectory_samples: int):
    """
    Evaluate single model on test set of dataset, saves results in a pd.DataFrame with columns:
    - [prewarm_steps, sample, steps, feature, model_id, run] (these hyperparameters are always saved)
    - [ground_truth, train_mean, train_var, prediction]
    - [*hyperparameters] (user-defined)
    Arguments:
    - model
    - test_dataset: MultimodalDataset (on which to evaluate the data)
    - hyperparameters: List, which additional hyperparameters to save in results
    - ahead_prediction_steps: how many steps to predict (cannot be more than the test set length)
    - trajectory_samples: how many prediction samples to draw from the model
    - prewarm_steps_on_train_set: how many steps are drawn from end of train set for prewarming
    """
    log = logging.getLogger('evaluate_model_on_dataset')
    if ahead_prediction_steps is not None:
        ahead_prediction_steps = min(ahead_prediction_steps, test_dataset.T-1)
    else:
        ahead_prediction_steps = test_dataset.T-1
    res = prepare_evaluation_df(model.args, train_dataset, test_dataset, ahead_prediction_steps, trajectory_samples, 0)

    if model is not None:
        predictions = []
        predictions_wo_inputs = []
        train_emas, train_inputs = train_dataset.data()
        test_emas, test_inputs = test_dataset.data()
        if test_inputs is not None:
            zero_inputs = tc.zeros_like(test_inputs)
        else:
            test_inputs = None
            zero_inputs = FileNotFoundError

        if 'x_mean' not in model.params or 'x_std' not in model.params:
            model.params['x_mean'] = train_emas.nanmean(0)
            model.params['x_std'] = ((train_emas - model.params['x_mean'])**2).nanmean(0)**0.5

        for k in range(trajectory_samples):            
            forecast = model.forecast_pf(train_emas, train_inputs, test_emas, test_inputs, n_particles=100)
            generated = forecast['x_fore_mean'][:ahead_prediction_steps]
            generated = tc.cat([tc.full((1, generated.shape[1]), tc.nan), generated], dim=0)
            predictions.append(generated.flatten())
            if test_inputs is not None:
                forecast_wo_inputs = model.forecast_pf(train_emas, train_inputs, test_emas, zero_inputs, n_particles=1000)
                generated_wo_inputs = forecast_wo_inputs['x_fore_mean'][:ahead_prediction_steps]
                generated_wo_inputs = tc.cat([tc.full((1, generated_wo_inputs.shape[1]), tc.nan), generated_wo_inputs], dim=0)
                predictions_wo_inputs.append(generated_wo_inputs.flatten())
        if len(predictions)>0:
            res['prediction'] = tc.cat(predictions, axis=0)
            if len(predictions_wo_inputs)>0:
                res['prediction_without_inputs'] = tc.cat(predictions_wo_inputs, axis=0)
        else:
            log.warning(f'{model.args["model_id"]} did not generate any predictions.')
            
        for hyper in hyperparameters:
            if hyper == 'data_path':
                res[hyper] = os.path.split(model.args[hyper])[1]
            elif 'preprocessing' in hyper and isinstance(model.args[hyper], list):
                res[hyper] = '-'.join(model.args[hyper])
            elif hyper == 'intervention':
                res[hyper] = 0
                if test_inputs is not None:
                    for k in res.index.get_level_values('steps').unique():
                        if k>0:
                            res.loc[(slice(None),slice(None),k), 'intervention'] = (test_inputs[k-1].nansum()>0).item() * 1
                        # if gt_inputs[0].nansum()>0:
                        #     res['intervention'] = 1
            elif hyper in model.args.keys():
                res[hyper] = model.args[hyper]
            else:
                raise ValueError(f'Hyperparameter {hyper} not found')
                # res[hyper] = eval_reallabor_utils.add_hyper(hyper, model)
        
    return res


def prepare_evaluation_df(args, train_dataset: MultimodalDataset, test_dataset: MultimodalDataset,
                          ahead_prediction_steps: int, trajectory_samples: int, prewarm_steps_on_train_set: int|list):  
    if isinstance(prewarm_steps_on_train_set, int):
        prewarm_steps_on_train_set = [prewarm_steps_on_train_set]
    n_prewarm_options = len(prewarm_steps_on_train_set)

    feature_names = train_dataset.timeseries['emas'].feature_names
    df_index = [prewarm_steps_on_train_set, range(trajectory_samples), range(ahead_prediction_steps+1), feature_names]
    df_index_names = ['prewarm_steps', 'sample', 'steps', 'feature']   
    res = pd.DataFrame(index=pd.MultiIndex.from_product(df_index, names=df_index_names), columns=['model_id', 'run'])
    res['model_id'] = args['model_id']
    res['participant'] = args['participant']
    date_identifier = 'train_until' if 'train_until' in args else ('train_on_data_until_timestep' if 'train_on_data_until_timestep' in args else 'train_on_data_until_datetime')    ####
    res[date_identifier] = args[date_identifier]
    res['run'] = args['run']
    n_feat = train_dataset.timeseries['emas'].data.shape[1]
    train_mean = train_dataset.timeseries['emas'].data.nanmean(0)
    train_var = ((train_dataset.timeseries['emas'].data - train_mean.unsqueeze(0))**2).nanmean(0)
    ground_truth = test_dataset.timeseries['emas'].data[:ahead_prediction_steps+1]
    ground_truth_mean = ground_truth[1:].nanmean(0)
    ground_truth_var = ((ground_truth[1:] - ground_truth_mean.unsqueeze(0))**2).nanmean(0)
    res['ground_truth'] = ground_truth.flatten().repeat(n_prewarm_options*trajectory_samples)
    res['gt_mean'] = ground_truth_mean.repeat(n_prewarm_options*(ahead_prediction_steps+1)*trajectory_samples)
    res['gt_var'] = ground_truth_var.repeat(n_prewarm_options*(ahead_prediction_steps+1)*trajectory_samples)
    res['train_mean'] = train_mean.repeat(n_prewarm_options*(ahead_prediction_steps+1)*trajectory_samples)
    res['train_var'] = train_var.repeat(n_prewarm_options*(ahead_prediction_steps+1)*trajectory_samples)
    res['prediction'] = np.nan
    res['prediction_without_inputs'] = np.nan

    return res


if __name__ == '__main__':

    MRT = 2

    evaluate_main_dirs = [  
                            data_utils.join_base_path(f'ordinal-bptt/results/v3_MRT{MRT}_PF'),
                         ]

    for main_dir in evaluate_main_dirs:
        if not os.path.exists(main_dir) and main_dir.endswith('_best_runs'):
            utils.extract_best_runs(main_dir.removesuffix('_best_runs'))
        eval_args = EvaluationArgs(main_dir,
                                    test_data_dir=data_utils.dataset_path(MRT, 'processed_csv_no_con'),
                                    hyperparameters=['latent_model', 'intervention', 'valid_training_data_points', 'valid_training_data_ratio', 'train_on_data_until_timestep', 'participant'],
                                    trajectory_samples=1,
                                    ahead_prediction_steps=7,
                                    epoch_criterion='latest',
                                    get_training_time=False,
                                    get_number_of_params=False,
                                    allow_test_inputs=False,
                                    # USE TEST SPLIT FILE TO EVALUATE EACH MODEL ON SEVERAL DAYS
                                    # test_split_file='/home/janik.fechtelpeter/Documents/reallaborai4u/data_management/train_test_splits/valid_first_alarms_no_con_smoothed.csv',
                                    # USE FIRST ALARMS FILE & TEST SET OFFSET DAYS TO EVALUATE MODEL ON DAY NO. (VALIDATION DAY + N)
                                    # test_set_offset_days=1,                                    
                                    # first_alarms_file=data_utils.train_test_split_path(MRT, 'first_alarms.csv'),
                                    # DO NOT SPECIFY ANY TEST FILE TO EVALUATE ON VALIDATION SET
                                    # exclude_hypers={'train_on_data_until_timestep':[3]},
                                    # include_hypers={'participant':'140.0'},
                                    ensemble_predictors=True,
                                    buggy_version_of_ensemble=False,
                                    label = 'ensemble',
                                    overwrite=True,
                                    verbose='print',
                                    preload_data=False,
                                    num_workers=10,
                                    )
        evaluate_complete_directory(eval_args)