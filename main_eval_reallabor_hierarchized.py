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
from hierarchized_bptt.hierarchized_models import HierarchizedPLRNN
from dataset.multimodal_dataset import  DatasetWrapper
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as tc
import eval_reallabor_utils
import utils
import data_utils

def configure_logging(verbose: str, path: Optional[str]=None, level: int=logging.INFO):   
    if path is None:
        path = os.getcwd() 
    logger = logging.getLogger('root')
    logger.handlers = []
    logger.setLevel(level)
    if verbose=='log':
        hdlr = logging.FileHandler(os.path.join(path, 'log.txt'))
    elif verbose=='print':
        hdlr = logging.StreamHandler()
    else:
        hdlr = logging.NullHandler()
    hdlr.setLevel(level)
    formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger

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
    get_final_epoch: bool=True
    ensemble_classifiers: bool=False
    label: str=''
    overwrite: bool=False
    file_format: str='csv'
    verbose: str='none'
    num_workers: int=1
    logging_level: int=logging.INFO


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
    model_dir_paths, train_data_mapping, test_data_mapping = eval_reallabor_utils.get_model_folders_and_preload_data(eval_args.main_dir, eval_args.test_data_dir, 
                                                                                                                     hierarchized=True, use_tqdm=True)
    # train_data_mapping = {k: None for k in train_data_mapping.keys()} # FOR DEBUGGING
    # test_data_mapping = {k: None for k in test_data_mapping.keys()} # FOR DEBUGGING
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
    log = configure_logging(eval_args.verbose, path=summary_path, level=eval_args.logging_level)
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
        with ThreadPoolExecutor(eval_args.num_workers) as executor:
            futures = {executor.submit(evaluate_model_folder, folder, eval_args,
                                       **{'preloaded_train_data': train_data_mapping[folder]}, **{'preloaded_test_data': test_data_mapping[folder]}, 
                                       ): folder for folder in model_dir_paths}
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
        if eval_args.ensemble_classifiers:
            results = eval_reallabor_utils.create_ensemble_prediction_eval_df(results)
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
    args = utils.load_args(folder)
    if eval_args.test_split_df is not None:
        test_splits = [row.to_dict() for i, row in eval_args.test_split_df.iterrows()]
    elif eval_args.first_alarms_df is not None:
        train_dataset_name = os.path.split(args['data_path'])[1]
        test_dataset_name = os.path.split(eval_args.test_data_dir)[1]
        test_splits = {}
        for subject_idx in args['subject_indices']:
            subj_first_alarms = eval_args.first_alarms_df[eval_args.first_alarms_df['participant']==subject_idx]
            val_split = data_utils.determine_test_index(subject_idx, args['train_until'], args['train_test_split_row'])
            day_of_val_split = subj_first_alarms.loc[(subj_first_alarms[train_dataset_name]==val_split), 'day']
            if len(day_of_val_split)==1:
                day_of_val_split = day_of_val_split.item()
            else:
                errors.append(f'Validation index {day_of_val_split} is not among first alarms.')
                continue
            if (subj_first_alarms['day']==day_of_val_split+eval_args.test_set_offset_days).any():
                test_splits[subject_idx] = int(subj_first_alarms.loc[subj_first_alarms['day']==day_of_val_split+eval_args.test_set_offset_days, test_dataset_name].item())
            else:  # day_of_val_split + test_set_offset_days doesn't exist
                errors.append(f'Test index {day_of_val_split+eval_args.test_set_offset_days} is not among first alarms.')
                continue
        test_splits = [test_splits]
    else:
        test_splits = [None]
    
    for splt in test_splits:
        if splt is not None:
            with_args = {'train_until': splt}
        else:
            with_args = {}
        try:
            model, train_dataset, test_dataset = eval_reallabor_utils.load_model_and_data(
                folder,
                epoch_criterion=eval_args.epoch_criterion, 
                allow_test_inputs=eval_args.allow_test_inputs,
                with_args=with_args,
                load_test_data_from=eval_args.test_data_dir,
                preloaded_train_data=preloaded_train_data, preloaded_test_data=preloaded_test_data,
                hierarchized=True)
        except Exception as e:
            errors.append(f'Test set {splt}: {str(e)}')
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
        try:
            res = evaluate_model_on_dataset(model, train_dataset, test_dataset, eval_args.hyperparameters, eval_args.ahead_prediction_steps,
                                            eval_args.trajectory_samples, eval_args.prewarm_steps_on_train_set, eval_args.random_z0)
        except Exception as e:
            log.error(e)
            continue
        if eval_args.get_training_time:
            res['training_time'] = eval_reallabor_utils.get_training_time(folder)
        if eval_args.get_number_of_params:
            res['n_params'] = eval_reallabor_utils.get_number_of_params(model)
        if eval_args.get_final_epoch:
            res['final_epoch'] = utils.available_epochs(folder)[-1]
        res.reset_index(inplace=True)
        results.append(res)

    if len(results)>0:
        results = pd.concat(results, ignore_index=True)
        results.index.name = 'idx'
        
    return results, errors, folder


def evaluate_model_on_dataset(model: HierarchizedPLRNN, train_wrapper: DatasetWrapper, test_wrapper: DatasetWrapper, 
                            hyperparameters: list, ahead_prediction_steps: int, trajectory_samples: int,
                            prewarm_steps_on_train_set: int|list, random_z0: bool):
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
    # if calculate_max_lyapunov:
    #     max_lyapunov = eval_reallabor_utils.lyapunov_spectrum(model, 500, 10).max().detach().numpy()
    #     res['max_lyapunov'] = max_lyapunov
    res = prepare_evaluation_df(model.args, train_wrapper, test_wrapper, ahead_prediction_steps, trajectory_samples, prewarm_steps_on_train_set)
    predictions = []
    predictions_wo_inputs = []   
    gt_emas, gt_inputs = test_wrapper.global_data(slice(0, ahead_prediction_steps+1))
    zero_inputs = tc.zeros_like(gt_inputs)
    subject_idx = tc.tensor(test_wrapper.datasets.index)
    x0 = gt_emas[:, 0]
    for p in res.index.get_level_values('prewarm_steps').unique():        
        if p > 0:
            prewarm_data, prewarm_inputs = train_wrapper.global_data(slice(-p-1,-1), global_indices=test_wrapper.dataset_indices)
        else:
            prewarm_data = prewarm_inputs = None
        for k in range(trajectory_samples):
            if random_z0:
                z0 = tc.randn((len(subject_idx), model.args['dim_z']))
            else:
                z0 = tc.zeros((len(subject_idx), model.args['dim_z']))
            generated = model.generate_free_trajectory(
                subject_idx, x0, ahead_prediction_steps, inputs=gt_inputs, z0=z0,
                prewarm_data=prewarm_data, prewarm_inputs=prewarm_inputs,
                return_hidden=False
            )
            predictions.append(generated.flatten())
            if gt_inputs is not None:
                generated_wo_inputs = model.generate_free_trajectory(
                    subject_idx, x0, ahead_prediction_steps, inputs=zero_inputs, z0=z0,
                    prewarm_data=prewarm_data, prewarm_inputs=prewarm_inputs,
                    return_hidden=False
                )
                predictions_wo_inputs.append(generated_wo_inputs.flatten())
            else:
                predictions_wo_inputs = None
    if len(predictions)>0:
        res['prediction'] = tc.cat(predictions, axis=0)
        if predictions_wo_inputs is not None:
            res['prediction_without_inputs'] = tc.cat(predictions_wo_inputs, axis=0)
    else:
        log.warning(f'{model.args["model_id"]} did not generate any predictions.')
        
    for hyper in hyperparameters:
        if hyper == 'data_path':
            res[hyper] = os.path.split(model.args[hyper])[1]
        elif 'preprocessing' in hyper and isinstance(model.args[hyper], list):
            res[hyper] = '-'.join(model.args[hyper])
        elif hyper == 'latent_model':
            res[hyper] = 'hierarchized-' + model.args[hyper]
        elif hyper == 'intervention':
            res[hyper] = 0
            if gt_inputs is not None:
                for p in res.index.get_level_values('participant').unique():
                    for k in res.index.get_level_values('steps').unique():
                        if k>1:
                            participant_int_idx = model.subject_index_map[p]
                            res.loc[(slice(None),slice(None),p,k)] = (gt_inputs[participant_int_idx,k-2].nansum()>0).item() * 1
        elif hyper in model.args.keys():
            res[hyper] = model.args[hyper]
        else:
            raise ValueError(f'Hyperparameter {hyper} not found')
        
    return res


def prepare_evaluation_df(args, train_wrapper: DatasetWrapper, test_wrapper: DatasetWrapper,
                        ahead_prediction_steps: int, trajectory_samples: int, prewarm_steps_on_train_set: int|list):
    if isinstance(prewarm_steps_on_train_set, int):
        prewarm_steps_on_train_set = [prewarm_steps_on_train_set]
    n_prewarm_options = len(prewarm_steps_on_train_set)
    feature_names = train_wrapper.datasets.iloc[0].timeseries['emas'].feature_names    
    total_res = []
    df_index_names = ['participant', 'steps', 'feature']
    for participant in test_wrapper.datasets.index:
        participant_n_steps = ahead_prediction_steps
        df_index = [[participant], range(1, participant_n_steps+1), feature_names]
        res = pd.DataFrame(index=pd.MultiIndex.from_product(df_index, names=df_index_names), 
                           columns=['model_id', 'run', 'train_until', 'ground_truth'])
        res['model_id'] = args['model_id']
        res['run'] = args['run']
        test_index = data_utils.determine_test_index(participant, args['train_until'], args['train_test_split_row'])
        res['train_until'] = test_index
        ground_truth = test_wrapper.pad_array_on_axis_with_nans_to_size(test_wrapper.datasets[participant].timeseries['emas'].data[1:participant_n_steps+1], 0, ahead_prediction_steps)
        res['ground_truth'] = ground_truth.flatten().numpy()
        gt_mean = ground_truth.nanmean(0)
        res['gt_mean'] = gt_mean.repeat(participant_n_steps).numpy()
        gt_var = ((ground_truth - ground_truth.nanmean(0, keepdim=True))**2).nanmean(0)
        res['gt_var'] = gt_var.repeat(participant_n_steps).numpy()
        train_mean = train_wrapper.datasets[participant].timeseries['emas'].data.nanmean(0)
        res['train_mean'] = train_mean.repeat(participant_n_steps).numpy()
        train_var = ((train_wrapper.datasets[participant].timeseries['emas'].data - train_mean.unsqueeze(0))**2).nanmean(0)
        res['train_var'] = train_var.repeat(participant_n_steps).numpy()
        res['prediction'] = np.nan
        res['prediction_without_inputs'] = np.nan
        total_res.append(res)
    total_res = pd.concat(total_res, axis=0, ignore_index=False)
    total_res = pd.concat([total_res]*trajectory_samples, axis=0, keys=range(trajectory_samples), names=['sample'])
    total_res = pd.concat([total_res]*n_prewarm_options, axis=0, keys=prewarm_steps_on_train_set, names=['prewarm_steps'])
    return total_res



if __name__ == '__main__':

    MRT = 3

    evaluate_main_dirs = [  
                            data_utils.join_base_path(f'ordinal-bptt/results/hiera_MRT{MRT}_finetuned'),
                            # data_utils.join_base_path('ordinal-bptt/results/hiera_MRT3_5splits_p28_best_runs')
                         ]    

    for main_dir in evaluate_main_dirs:
        if not os.path.exists(main_dir) and main_dir.endswith('_best_runs'):
            utils.extract_best_runs(main_dir.removesuffix('_best_runs'))
        eval_args = EvaluationArgs(main_dir=main_dir,
                                    test_data_dir=data_utils.dataset_path(MRT, 'processed_csv_no_con'),
                                    # hyperparameters=['valid_training_data_points', 'valid_training_data_ratio', 'train_on_data_until_timestep', 'participant',
                                    #                  'dim_z', 'dim_y'],
                                    # hyperparameters=['participant', 'latent_model'],
                                    hyperparameters=['train_until', 'latent_model', 'intervention'],
                                    random_z0=False,
                                    trajectory_samples=1,
                                    ahead_prediction_steps=7,
                                    epoch_criterion='latest',
                                    prewarm_steps_on_train_set=[0],
                                    calculate_max_lyapunov=False,
                                    get_training_time=True,
                                    get_number_of_params=False,
                                    get_final_epoch=True,
                                    allow_test_inputs=True,
                                    # USE TEST SPLIT FILE TO EVALUATE EACH MODEL ON SEVERAL DAYS
                                    # test_split_file='/home/janik.fechtelpeter/Documents/reallaborai4u/data_management/train_test_splits/valid_first_alarms_no_con_smoothed.csv',
                                    # USE FIRST ALARMS FILE & TEST SET OFFSET DAYS TO EVALUATE MODEL ON DAY NO. (VALIDATION DAY + N)
                                    # test_set_offset_days=1,                                    
                                    # first_alarms_file=data_utils.train_test_split_path(MRT, 'first_alarms.csv'),
                                    # DO NOT SPECIFY ANY TEST FILE TO EVALUATE ON VALIDATION SET
                                    # exclude_hypers={'train_on_data_until_timestep':[3]},
                                    ensemble_classifiers=True,
                                    label = '7stepsahead_interv', 
                                    overwrite=False,
                                    verbose='log',
                                    num_workers=1,
                                    logging_level=logging.WARNING
                                    )
        evaluate_complete_directory(eval_args)