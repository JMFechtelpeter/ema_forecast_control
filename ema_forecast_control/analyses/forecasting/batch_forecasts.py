import logging
import ast
import os
from typing import Optional
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as tc

from ema_forecast_control.dataset.time_series_dataset import TimeSeriesDataset
from ema_forecast_control.utils import evaluation_utils, path_utils, logging_utils
from ema_forecast_control.utils import backwards_compatibility


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
    assert os.path.exists(path_utils.get_project_model_root(eval_args.project_name)), f'Directory {eval_args.project_name} does not exist'

    model_dir_paths = evaluation_utils.get_model_folders(eval_args.project_name)
    assert len(model_dir_paths) > 0,'Given project_name path does not contain models.'

    if eval_args.preload_data:
        test_data_mapping = evaluation_utils.preload_data(model_dir_paths, eval_args.test_data_dir, use_tqdm=True)
    else:
        test_data_mapping = {k: None for k in model_dir_paths}    

    # create folder to store summary plots and metrics
    summary_path = path_utils.join_base_path('forecasts')
    forecast_file_name = os.path.join(summary_path, '_'.join([eval_args.project_name, eval_args.label, f'forecasts.{eval_args.file_format}']))
    if os.path.exists(forecast_file_name) and not eval_args.overwrite:
        decision = input(f'An evaluation path with label "{eval_args.label}" already exists. Overwrite? ')
        if decision not in ['y', 'Y', '1']:
            return None
    log = logging_utils.configure_logging(summary_path, eval_args.verbose)
    print(f'Evaluating project {eval_args.project_name}, label {eval_args.label}')
    log.info(f'Evaluating project {eval_args.project_name}, label {eval_args.label}')
    for arg in eval_args.__dict__.keys():
        log.info(f'{arg}: {eval_args.__dict__[arg]}')

    if eval_args.test_split_file is not None:
        eval_args.test_split_df = pd.read_csv(eval_args.test_split_file, index_col=0)
    else:
        eval_args.test_split_df = None
    results = []
    n_errors = 0

    if eval_args.n_processes > 1:
        from multiprocessing import get_context
        mp_ctx = get_context('spawn')  # safer with MKL/NumPy
        Executor = ProcessPoolExecutor
        executor_kwargs = dict(max_workers=eval_args.n_processes,
                                mp_context=mp_ctx,
                                initializer=_init_worker,
                                initargs=(1,))

        with Executor(**executor_kwargs) as executor:
            futures = {}
            for folder in model_dir_paths:
                kwargs = {}
                kwargs['preloaded_data'] = test_data_mapping[folder]
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
                                            preloaded_data=test_data_mapping[folder])
            if len(res)>0:
                results.append(res)
            if len(err)>0:
                log.warning(f'There have been errors evaluating {folder}.')
                for e in err:
                    log.warning(f'{folder}: {e}')
                n_errors += len(err)

    if len(results) > 0:
        results = pd.concat(results, ignore_index=True)
        results.index.name = 'idx'
        if eval_args.file_format=='csv':
            results.to_csv(forecast_file_name)
        elif eval_args.file_format=='json':
            results.to_json(forecast_file_name)
    else:
        log.error('No models were evaluated.')
    
    log.info(f'{n_errors} errors during data/model loading...')
    
    return results


def evaluate_model_folder(folder: str, eval_args: EvaluationArgs, preloaded_data: pd.DataFrame|None=None):
    
    log = logging.getLogger('evaluate_model_folder')
    errors = []
    results = []
    args = backwards_compatibility.update_data_path(path_utils.load_args(folder))
    if eval_args.test_data_dir is not None:
        args['data_path'] = eval_args.test_data_dir
    if eval_args.test_split_df is not None:    
        begin_test_set = eval_args.test_split_df[args['participant']].dropna().to_numpy()
    else:
        begin_test_set = [None]
    for bts in begin_test_set:
        if bts is not None:
            args['train_test_split'] = bts
        try:
            train_data, train_inputs, test_data, test_inputs = evaluation_utils.prepare_data_for_model_evaluation(
                folder, allow_test_inputs=eval_args.allow_test_inputs, with_args=args,
                preloaded_data=preloaded_data
            )
            log.info(f"Loaded data with {train_data.shape[0]} time steps.")
        except Exception as e:
            errors.append(f'Test set beginning at {args["train_test_split"]}: Error loading data {str(e)}')
            continue
        try:
            model = evaluation_utils.init_model_from_path(folder, with_args=args, select_epoch=eval_args.epoch_criterion)
        except Exception as e:
            errors.append(f'Test set beginning at {args["train_test_split"]}: Could not initialize model ({str(e)})')
            continue
        if not eval_args.use_pseudoinverse:
            try:
                Gamma = evaluation_utils.get_Gamma(model, folder)
            except Exception as e:
                errors.append(f'Test set beginning at {args["train_on_data_until_timestep"]}: Could not get recognition model ({str(e)})')
                continue
        else:
            Gamma = None
        if len(test_data) == 0:
            continue
        if not evaluation_utils.include_exclude_hypers(model.args, eval_args.include_hypers, eval_args.exclude_hypers):
            continue
        model.args = evaluation_utils.complement_args_with_data_info(model.args, train_data)
        res = evaluate_model_on_data(model, train_data, train_inputs, test_data, test_inputs, Gamma, eval_args.hyperparameters, eval_args.ahead_prediction_steps,
                                        eval_args.trajectory_samples, eval_args.prewarm_steps, eval_args.random_z0)
        res['train_test_split'] = args['train_test_split']
        res['training_time'] = evaluation_utils.get_training_time(folder)
        res['n_params'] = evaluation_utils.get_number_of_params(model)
        res.reset_index(inplace=True)
        results.append(res)
    
    if len(results)>0:
        results = pd.concat(results, ignore_index=True)
        results.index.name = 'idx'
        
    return results, errors, folder


def evaluate_model_on_data(model, train_data: tc.Tensor, train_inputs: tc.Tensor,
                            test_data: Optional[tc.Tensor], test_inputs: Optional[tc.Tensor],
                              Gamma: Optional[tc.Tensor],
                              hyperparameters: list, ahead_prediction_steps: Optional[int], trajectory_samples: int,
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
    if ahead_prediction_steps is not None:
        ahead_prediction_steps = min(ahead_prediction_steps, test_data.shape[0]-1)
    else:
        ahead_prediction_steps = test_data.shape[0]-1

    feature_names = model.args['obs_features']
    res = prepare_evaluation_df(model.args, train_data, test_data, feature_names, ahead_prediction_steps, trajectory_samples, prewarm_steps_on_train_set)

    if model is not None:
        predictions = []
        predictions_wo_inputs = []
        if test_inputs is not None:
            zero_inputs = tc.zeros_like(test_inputs)
        else:
            test_inputs = None
            zero_inputs = None
        x0 = test_data[0]
        recognition_matrix = model.get_recognition_model(Gamma=Gamma)
        for p in res.index.get_level_values('prewarm_steps').unique():
            if p > 0:
                prewarm_data = train_data[-p-1:-1]
                prewarm_inputs = train_inputs[-p-1:-1] if train_inputs is not None else None
            else:
                prewarm_data = prewarm_inputs = None
            for k in range(trajectory_samples):
                if random_z0:
                    z0 = tc.randn(model.args['dim_z'])
                else:
                    z0 = None
                generated, latent_traj = model.generate_free_trajectory(
                    x0, ahead_prediction_steps, inputs=test_inputs, z0=z0,
                    prewarm_data=prewarm_data, prewarm_inputs=prewarm_inputs,
                    recognition_matrix=recognition_matrix,
                    return_hidden=True, 
                )
                generated = tc.cat([tc.full((1, generated.shape[1]), tc.nan), generated], dim=0)
                predictions.append(generated.flatten())
                if test_inputs is not None:
                    generated_wo_inputs, latent_traj = model.generate_free_trajectory(
                        x0, ahead_prediction_steps, inputs=zero_inputs, z0=z0,
                        prewarm_data=prewarm_data, prewarm_inputs=prewarm_inputs,
                        recognition_matrix=recognition_matrix,
                        return_hidden=True
                    )
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
            elif hyper in model.args.keys():
                res[hyper] = model.args[hyper]
            else:
                raise ValueError(f'Hyperparameter {hyper} not found')
        
    return res


def prepare_evaluation_df(args, train_data: tc.Tensor, test_data: tc.Tensor, feature_names: list[str],
                          ahead_prediction_steps: int, trajectory_samples: int, prewarm_steps_on_train_set: int|list):  
    if isinstance(prewarm_steps_on_train_set, int):
        prewarm_steps_on_train_set = [prewarm_steps_on_train_set]
    n_prewarm_options = len(set(prewarm_steps_on_train_set))

    df_index = [sorted(set(prewarm_steps_on_train_set)), range(trajectory_samples), range(ahead_prediction_steps+1), feature_names]
    df_index_names = ['prewarm_steps', 'sample', 'steps', 'feature']   
    res = pd.DataFrame(index=pd.MultiIndex.from_product(df_index, names=df_index_names), columns=['model_id', 'run'])
    res['model_id'] = args['model_id']
    res['participant'] = args['participant']
    date_identifier = 'train_until' if 'train_until' in args else ('train_on_data_until_timestep' if 'train_on_data_until_timestep' in args else 'train_on_data_until_datetime')    ####
    res[date_identifier] = args[date_identifier]
    res['run'] = args['run']
    n_feat = train_data.shape[1]
    train_mean = train_data.nanmean(0)
    train_var = ((train_data - train_mean.unsqueeze(0))**2).nanmean(0)
    ground_truth = test_data[:ahead_prediction_steps+1]
    ground_truth_mean = ground_truth[1:].nanmean(0)
    ground_truth_var = ((ground_truth[1:] - ground_truth_mean.unsqueeze(0))**2).nanmean(0)
    res['ground_truth'] = ground_truth.flatten().repeat(n_prewarm_options*trajectory_samples)
    res['prediction'] = np.nan
    res['prediction_without_inputs'] = np.nan

    return res


if __name__ == '__main__':

    parser = ArgumentParser(description='Batch model evaluation')
    parser.add_argument('evaluate_projects', nargs='*', type=str, default=[f'v3_MRT3_every_day'], help='List of project directories to evaluate')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Evaluate on data from this directory')
    parser.add_argument('--epoch_criterion', type=str, default='latest', help='Epoch selection criterion')
    parser.add_argument('--hyperparameters', type=str, nargs='*', default=['dim_z'], help='Additional hyperparameters to save')
    parser.add_argument('--random_z0', action='store_true', help='Sample a random latent initial state')
    parser.add_argument('--ahead_prediction_steps', type=int, default=10, help='Number of prediction steps ahead')
    parser.add_argument('--trajectory_samples', type=int, default=1, help='Number of forecast samples per model')
    parser.add_argument('--prewarm_steps', type=int, nargs='+', default=[0], help='Prewarm steps on the training set')
    parser.add_argument('--allow_test_inputs', action='store_true', help='Allow using test inputs during evaluation')
    parser.add_argument('--test_split_file', type=str, default=None, help='CSV file with test split information')
    parser.add_argument('--include_hypers', type=ast.literal_eval, default={}, help='Dict of hyperparameters to include')
    parser.add_argument('--exclude_hypers', type=ast.literal_eval, default={}, help='Dict of hyperparameters to exclude')
    parser.add_argument('--use_pseudoinverse', action='store_true', help='Use the pseudoinverse observation model instead of pseudo-Kalman gain')
    parser.add_argument('--label', type=str, default='', help='Label appended to the output file name')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--file_format', type=str, choices=['csv', 'json'], default='json', help='Output file format')
    parser.add_argument('--verbose', type=str, choices=['none', 'print', 'log'], default='none', help='Logging verbosity')
    parser.add_argument('--preload_data', action='store_true', help='Preload data before evaluation')
    parser.add_argument('--n_processes', type=int, default=4, help='Number of worker processes')

    eval_args = parser.parse_args()

    for project in eval_args.evaluate_projects:
        project_args = copy.copy(eval_args)
        project_args.project_name = project        
        evaluate_complete_directory(project_args)