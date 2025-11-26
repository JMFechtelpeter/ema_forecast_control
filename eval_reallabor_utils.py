import os
import sys
import glob
from typing import Callable, Optional, Any
import warnings
import pandas as pd
import yaml
import hashlib
import re
import time
from tqdm import tqdm
from operator import itemgetter
import numpy as np
from numpy.typing import ArrayLike
import torch as tc
import matplotlib.pyplot as plt
from scipy import stats, linalg
import statsmodels.api as sm_api
import statsmodels.formula.api as sm_formula_api
import data_utils
import utils
from bptt.plrnn import PLRNN
from hierarchized_bptt.hierarchized_models import HierarchizedPLRNN
from comparison_models.simple_models import models as simple_models
from comparison_models.simple_models.models import KalmanFilter, VAR1
from comparison_models.transformer.time_series_transformer import AutoregressiveTransformer
from comparison_models.particle_filter_plrnn.pf_plrnn import PF_PLRNN
from concurrent.futures import ThreadPoolExecutor

import time
from joblib import Parallel, delayed

# 1. bootstrapping in metrics einbauen
# 2. bootstrapping macht groupby und führt metrics für jede Gruppe aus
# 3. 

def rmse(eval_df: pd.DataFrame, hyperparameters: list, bootstrap_samples: int=0, relative_ci: bool=False, confidence: float=95, use_tqdm: bool=False):
    ''' bootstrap between-subject variability '''
    levels = ['participant', identify_test_split_argument(eval_df), 'steps', 'feature']
    for lvl in levels:
        if lvl in hyperparameters:
            levels.remove(lvl)
    working_df = eval_df.copy(deep=True)
    working_df['sq_residuals'] = (working_df['prediction'] - working_df['ground_truth'])**2
    for l in range(len(levels)):
        working_df = working_df.groupby(levels + hyperparameters).mean(numeric_only=True)
        levels = levels[:-1]
    if len(hyperparameters)>0:
        result = working_df.groupby(hyperparameters).mean(numeric_only=True)
    else:
        result = working_df.mean(numeric_only=True)
    result = np.sqrt(result['sq_residuals']).rename('rmse')
    if bootstrap_samples > 0:
        ci_low, ci_high = (100-confidence)/2, (100+confidence)/2
        rmse_samples = []
        if use_tqdm:
            iterator = tqdm(range(bootstrap_samples), desc=f'Bootstrapping rmse...')
        else:
            iterator = range(bootstrap_samples)
        for _ in iterator:
            if len(hyperparameters)>0:
                sample = working_df.groupby(hyperparameters).sample(frac=1, replace=True).groupby(hyperparameters).mean(numeric_only=True)
            else:
                sample = working_df.sample(frac=1, replace=True).groupby(hyperparameters).mean(numeric_only=True)
            rmse_samples.append(np.sqrt(sample['sq_residuals']).rename('rmse'))

        rmse_samples = pd.concat(rmse_samples, axis=1)
        ci_lower = rmse_samples.quantile(ci_low / 100, axis=1)
        ci_upper = rmse_samples.quantile(ci_high / 100, axis=1)
        if relative_ci:
            ci_lower = result - ci_lower
            ci_upper = ci_upper - result
        ci = pd.concat([ci_lower, ci_upper], axis=1, keys=['ci_lower', 'ci_upper'])
        if relative_ci:
            ci = ci.clip(lower=0)

        return result, ci
    else:

        return result


def mae(eval_df: pd.DataFrame, hyperparameters: list):
    levels = ['participant', identify_test_split_argument(eval_df), 'steps', 'feature']
    for lvl in levels:
        if lvl in hyperparameters:
            levels.remove(lvl)
    working_df = eval_df.copy(deep=True)
    working_df['abs_residuals'] = (working_df['prediction'] - working_df['ground_truth']).abs()
    for l in range(len(levels)):
        working_df = working_df.groupby(levels + hyperparameters).mean(numeric_only=True)
        levels = levels[:-1]
    if len(hyperparameters)>0:
        working_df = working_df.groupby(hyperparameters).mean(numeric_only=True)
    else:
        working_df = working_df.mean(numeric_only=True)
    result = working_df['abs_residuals'].rename('mae')
    return result


def average_semi_variance(df: pd.DataFrame):
    ''' calculates theta(ASV) according to Piepho (2019) '''
    cov_matrix = df.cov()
    n = cov_matrix.shape[0]
    return np.trace(cov_matrix - 1/n * np.tile(cov_matrix.sum(axis=0), (n, 1)))


def pseudo_r2_piepho(eval_df: pd.DataFrame, hyperparameters: list, 
                     compare_to_eval_df: pd.DataFrame|None=None,
                     individual_r2: bool=False, per_day: bool=False,
                     hypers_that_apply_only_to_residual_variance: Optional[list]=None):
    
    # def get_asv_for_all_levels_and_hypers_old(df, levels, hypers):
    #     working_df = df.copy().set_index(['participant', 'train_on_data_until_timestep', 'steps', 'feature', 'model_id', 'run'] + hypers, append=False)
    #     working_df = working_df[['ground_truth', 'prediction']]
    #     working_df['residuals'] = working_df['prediction'] - working_df['ground_truth']
    #     working_df = working_df.set_index(np.arange(len(working_df)), append=True).unstack('feature')

    #     grouper = working_df.groupby(levels + hypers)
    #     index = grouper.mean().index
    #     asv_residuals = pd.Series(index=index, dtype=float)
    #     asv_total = pd.Series(index=index, dtype=float)
    #     with warnings.catch_warnings():
    #         # warnings.filterwarnings("ignore", category=FutureWarning)
    #         for index, group in grouper:
    #             asv_residuals.loc[index] = average_semi_variance(group['residuals'])
    #             asv_total.loc[index] = average_semi_variance(group['ground_truth'])
    #     return asv_residuals, asv_total
    
    def get_asv_for_all_levels_and_hypers(df, levels, hypers):
        working_df = df.copy()
        working_df['aux_index'] = working_df.groupby('feature').cumcount()
        working_df['residuals'] = working_df['prediction'] - working_df['ground_truth']
        residuals = working_df.pivot(index=levels + hypers + ['aux_index'], columns='feature', values='residuals')
        gt = working_df.pivot(index=levels + hypers + ['aux_index'], columns='feature', values='ground_truth')

        asv_residuals = residuals.groupby(levels + hypers).apply(average_semi_variance)
        asv_total = gt.groupby(levels + hypers).apply(average_semi_variance)
        return asv_residuals, asv_total

    calc_asv_on_levels = ['participant']
    if per_day:
        calc_asv_on_levels.append(identify_test_split_argument(eval_df))
    hyper_without_levels = hyperparameters.copy()
    if hypers_that_apply_only_to_residual_variance is not None:
        hypers_that_apply_to_total_var = [h for h in hyperparameters if h not in hypers_that_apply_only_to_residual_variance]
        hypers_without_levels_that_apply_to_total_var = [h for h in hyperparameters if h not in hypers_that_apply_only_to_residual_variance]
    else:
        hypers_that_apply_to_total_var = hyperparameters.copy()
        hypers_without_levels_that_apply_to_total_var = hyperparameters.copy()
    for lvl in calc_asv_on_levels:
        if lvl in hyperparameters:
            hyper_without_levels.remove(lvl)
        if lvl in hypers_without_levels_that_apply_to_total_var:
            hypers_without_levels_that_apply_to_total_var.remove(lvl)

    asv_residuals, asv_total = get_asv_for_all_levels_and_hypers(eval_df, calc_asv_on_levels, hyper_without_levels)
    if compare_to_eval_df is not None:
        asv_total, _ = get_asv_for_all_levels_and_hypers(compare_to_eval_df, calc_asv_on_levels, hypers_without_levels_that_apply_to_total_var)

    if individual_r2:
        r2 = 1 - asv_residuals/asv_total
        r2 = r2.groupby(hyperparameters).mean()
    else:
        if len(hyperparameters) > 0:
            asv_residuals = asv_residuals.groupby(hyperparameters).mean()
            if len(hypers_that_apply_to_total_var) > 0:
                asv_total = asv_total.groupby(hypers_that_apply_to_total_var).mean()
            else:
                asv_total = asv_total.mean()
        else:
            asv_residuals = asv_residuals.mean()
            asv_total = asv_total.mean()
        r2 = 1 - asv_residuals/asv_total

    return r2


def pseudo_r2_janik(eval_df: pd.DataFrame, hyperparameters: list,
                    compare_to_eval_df: pd.DataFrame|None=None,
                    per_day=False,
                    use_train_mean_as_baseline_predictor=False):
    
    levels = ['participant', 'feature']
    if per_day:
        levels.append(identify_test_split_argument(eval_df))
    for lvl in levels:
        if lvl in hyperparameters:
            levels.remove(lvl)
    working_df = eval_df.copy()[levels + hyperparameters]
    working_df['sq_residuals'] = (eval_df['prediction'] - eval_df['ground_truth'])**2
    if compare_to_eval_df is not None:
        working_compare = compare_to_eval_df.copy()[levels + hyperparameters]
        working_compare['sq_mean_residuals'] = (compare_to_eval_df['prediction'] - compare_to_eval_df['ground_truth'])**2
    elif use_train_mean_as_baseline_predictor:
        working_compare = working_df
        working_compare['sq_mean_residuals'] = (working_df['ground_truth'] - working_df['train_mean'])**2
    else:
        working_compare = working_df
        working_compare['sq_mean_residuals'] = np.nan
        grouper = working_compare.groupby(levels + hyperparameters)
        ngroup = grouper.ngroup()
        for g in range(grouper.ngroups):
            working_compare.loc[ngroup==g, 'sq_mean_residuals'] = (working_df.loc[ngroup==g, 'ground_truth'] - working_df.loc[ngroup==g, 'ground_truth'].mean())**2

    
    for l in range(len(levels)):
        working_df = working_df.groupby(levels + hyperparameters).mean()
        working_compare = working_compare.groupby(levels + hyperparameters).mean()
        levels = levels[:-1]
    if len(hyperparameters)>0:
        working_df = working_df.groupby(hyperparameters).mean()
        working_compare = working_compare.groupby(hyperparameters).mean()
    else:
        working_df = working_df.mean()
        working_compare = working_compare.groupby(hyperparameters).mean()

    r2 = 1 - working_df['sq_residuals']/working_compare['sq_mean_residuals']

    return r2


def training_time(eval_df: pd.DataFrame, hyperparameters: list):
    levels = ['participant', identify_test_split_argument(eval_df), 'steps', 'feature']
    for lvl in levels:
        if lvl in hyperparameters:
            levels.remove(lvl)
    working_df = eval_df.copy(deep=True)
    for l in range(len(levels)):
        working_df = working_df.groupby(levels + hyperparameters).mean(numeric_only=True)
        levels = levels[:-1]
    if len(hyperparameters)>0:
        working_df = working_df.groupby(hyperparameters).mean(numeric_only=True)
    else:
        working_df = working_df.mean(numeric_only=True)
    result = working_df['training_time']
    return result

def final_epoch(eval_df: pd.DataFrame, hyperparameters: list):
    levels = ['participant', identify_test_split_argument(eval_df), 'steps', 'feature']
    for lvl in levels:
        if lvl in hyperparameters:
            levels.remove(lvl)
    working_df = eval_df.copy(deep=True)
    for l in range(len(levels)):
        working_df = working_df.groupby(levels + hyperparameters).mean(numeric_only=True)
        levels = levels[:-1]
    if len(hyperparameters)>0:
        working_df = working_df.groupby(hyperparameters).mean(numeric_only=True)
    else:
        working_df = working_df.mean(numeric_only=True)
    result = working_df['final_epoch']
    return result



def rmse2(eval_df: pd.DataFrame, hyperparameters: list, bootstrap_samples: int=0, relative_ci: bool=False, confidence: float=95, use_tqdm: bool=False):
    ''' bootstrap within-subject variability '''
    levels = ['participant', identify_test_split_argument(eval_df), 'steps', 'feature']
    for lvl in levels:
        if lvl in hyperparameters:
            levels.remove(lvl)

    rmse_samples = []
    ci_low, ci_high = (100-confidence)/2, (100+confidence)/2
    if use_tqdm and bootstrap_samples > 0:
        iterator = tqdm(range(bootstrap_samples+1), desc=f'Bootstrapping rmse...')
    else:
        iterator = range(bootstrap_samples+1)
    for n in iterator:
        working_df = eval_df.copy(deep=True)
        working_df['sq_residuals'] = (working_df['prediction'] - working_df['ground_truth'])**2
        for l in range(len(levels)):
            working_df = working_df.groupby(levels + hyperparameters).mean(numeric_only=True)
            levels = levels[:-1]
        if len(hyperparameters)>0:
            get_sample_from = working_df.groupby(hyperparameters)
        else:
            get_sample_from = working_df
        if n==0:
            result = get_sample_from.mean(numeric_only=True)
        else:
            result = get_sample_from.sample(frac=1, replace=True).groupby(hyperparameters).mean(numeric_only=True)
        rmse_samples.append(np.sqrt(result['sq_residuals']).rename('rmse'))
    
    result = rmse_samples[0]
    if bootstrap_samples > 0:
        rmse_samples = pd.concat(rmse_samples[1:], axis=1)
        ci_lower = rmse_samples.quantile(ci_low / 100, axis=1)
        ci_upper = rmse_samples.quantile(ci_high / 100, axis=1)
        if relative_ci:
            ci_lower = result - ci_lower
            ci_upper = ci_upper - result
        ci = pd.concat([ci_lower, ci_upper], axis=1, keys=['ci_lower', 'ci_upper'])
        if relative_ci:
            ci = ci.clip(lower=0)

        return result, ci
    else:

        return result


def bootstrap(statistic_fn: Callable, eval_df: pd.DataFrame, hyperparameters: list, 
              samples: int=1000, interval: float=95.0, relative_values: bool=False, 
              use_tqdm=False, workers: int=1, **statistic_kwargs):
    ci_low, ci_high = (100-interval)/2, (100+interval)/2
    bootstrap_samples = []

    if len(hyperparameters)>0:
        compute_sample = lambda _: statistic_fn(eval_df.groupby(hyperparameters).sample(frac=1, replace=True), hyperparameters, **statistic_kwargs)
    else:
        compute_sample = lambda _: statistic_fn(eval_df.sample(frac=1, replace=True), hyperparameters, **statistic_kwargs)

    if workers == 1:
        if use_tqdm:
            iterator = tqdm(range(samples), desc=f'Bootstrapping {statistic_fn.__name__}...')
        else:
            iterator = range(samples)
        for _ in iterator:
            bootstrap_samples.append(compute_sample(_))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            if use_tqdm:
                iterator = tqdm(executor.map(compute_sample, range(samples)), total=samples, desc=f'Bootstrapping {statistic_fn.__name__}...')
            else:
                iterator = executor.map(compute_sample, range(samples))
            bootstrap_samples.extend(iterator)

    bootstrap_samples = pd.concat(bootstrap_samples, axis=1)
    ci_lower = bootstrap_samples.quantile(ci_low / 100, axis=1)
    ci_upper = bootstrap_samples.quantile(ci_high / 100, axis=1)
    if relative_values:
        abs_statistic = statistic_fn(eval_df, hyperparameters, **statistic_kwargs)
        ci_lower = abs_statistic - ci_lower
        ci_upper = ci_upper - abs_statistic
    ci = pd.concat([ci_lower, ci_upper], axis=1, keys=['ci_lower', 'ci_upper'])
    if relative_values:
        ci = ci.clip(lower=0)

    return ci


def load_eval_df(path: str, add_step_0: bool=True):
    eval_df = pd.read_csv(path, index_col=0)
    if add_step_0 and (0 not in eval_df['steps']):
        model_ids = eval_df['model_id'].unique()
        for mid in model_ids:
            assume_model_path = os.path.join(os.path.dirname(os.path.dirname(path)), mid)
            if os.path.exists(assume_model_path) and len(os.listdir(assume_model_path))>0:
                args = utils.load_args(os.path.join(assume_model_path, os.listdir(assume_model_path)[0]))
                if args['latent_model'].startswith('hierarchized'):
                    raise NotImplementedError('')
                


def create_difference_eval_df(eval_df: pd.DataFrame, only_changes: bool=False, only_interventions: bool=False, 
                              use_ground_truth_for_predicted_difference: bool=False, dropna: bool=False):
    diff_eval_df = eval_df.copy(deep=True)
    n_features = len(eval_df['feature'].unique())
    diff_eval_df['ground_truth'] = eval_df['ground_truth'].diff(n_features)
    if 0 in eval_df['steps']:
        diff_eval_df.loc[eval_df['steps']==0, 'prediction'] = eval_df.loc[eval_df['steps']==0, 'ground_truth']
        if 'prediction_without_inputs' in eval_df.columns:
            diff_eval_df.loc[eval_df['steps']==0, 'prediction_without_inputs'] = eval_df.loc[eval_df['steps']==0, 'ground_truth']
        diff_eval_df.loc[eval_df['steps']==0, 'ground_truth'] = np.nan
    else:
        diff_eval_df.loc[eval_df['steps']==1, 'ground_truth'] = np.nan
    if use_ground_truth_for_predicted_difference:
        diff_eval_df['prediction'] = eval_df['prediction'] - eval_df['ground_truth'].shift(n_features)
    else:
        diff_eval_df['prediction'] = eval_df['prediction'].diff(n_features)
    if 'prediction_without_inputs' in eval_df.columns:
        if use_ground_truth_for_predicted_difference:
            diff_eval_df['prediction_without_inputs'] = eval_df['prediction_without_inputs'] - eval_df['ground_truth'].shift(n_features)
        else:
            diff_eval_df['prediction_without_inputs'] = eval_df['prediction_without_inputs'].diff(n_features)
    if only_changes:
        if 'prediction_without_inputs' in eval_df.columns:
            diff_eval_df.loc[diff_eval_df['ground_truth']==0, 'prediction_without_inputs'] = np.nan
        diff_eval_df.loc[diff_eval_df['ground_truth']==0, ['ground_truth', 'prediction']] = np.nan        
    if only_interventions:
        if 'prediction_without_inputs' in eval_df.columns:
            diff_eval_df.loc[eval_df['intervention']==0, 'prediction_without_inputs'] = np.nan    
        diff_eval_df.loc[eval_df['intervention']==0, ['ground_truth', 'prediction']] = np.nan           
    if dropna:
        diff_eval_df = diff_eval_df.loc[diff_eval_df['ground_truth'].notna()]
    return diff_eval_df

def create_intervention_eval_df(eval_df: pd.DataFrame):
    intervention_eval_df = eval_df.copy(deep=True)
    n_features = len(eval_df['feature'].unique())
    intervention_eval_df.loc[intervention_eval_df['intervention']==0, 'prediction'] = np.nan
    intervention_eval_df.loc[intervention_eval_df['steps']==1, 'prediction'] = np.nan
    if 'prediction_without_inputs' in eval_df.columns:
        intervention_eval_df.loc[intervention_eval_df['intervention']==0, 'prediction_without_inputs'] = np.nan
        intervention_eval_df.loc[intervention_eval_df['steps']==1, 'prediction_without_inputs'] = np.nan
    return intervention_eval_df


def calculate_metrics(eval_df: pd.DataFrame, hyperparameters: list[str], metrics: list['str'], 
                      sources_of_variance: Optional[list]=None, ci: Optional[str]='sem', outlier_threshold: Optional[int|float]=None,
                      mean_over: list[str]=['feature', 'steps']):
    hp = hyperparameters
    working_df = eval_df.copy()
    results_df = pd.DataFrame()
    errorbar_df = pd.DataFrame()

    def ci_fn(key: str):
        if ci is not None:
            ci_df = working_df
            if sources_of_variance is not None and ci!='bootstrap':
                ci_df = working_df.groupby(hp + sources_of_variance)[key].mean().reset_index()
            if ci=='sem':            
                ci_df = ci_df.groupby(hp)[key].sem()
            elif ci=='std':
                ci_df = ci_df.groupby(hp)[key].std()
            elif ci=='bootstrap':
                ci_df = bootstrap_ci(key)
        else:
            ci_df = None
        return ci_df
    
    def bootstrap_ci(key: str):
        ''' Calculate bootstrap confidence intervals of metrics according to the column key'''
        n_iterations = 1000
        ci_low, ci_high = 2.5, 97.5
        bootstrap_samples = []
        for _ in range(n_iterations):
            sample = working_df.sample(frac=1, replace=True)
            sample_mean = sample.groupby(hp)[key].mean()
            bootstrap_samples.append(sample_mean)
        bootstrap_samples = pd.concat(bootstrap_samples, axis=1)
        ci_lower = bootstrap_samples.quantile(ci_low / 100, axis=1)
        ci_upper = bootstrap_samples.quantile(ci_high / 100, axis=1)
        ci_df = (ci_upper - ci_lower)/2

        return ci_df

    if outlier_threshold is not None:
        exclude_model_ids = working_df.loc[working_df['prediction'].abs()>outlier_threshold, 'model_id'].unique()
        working_df = working_df.loc[~working_df['model_id'].isin(exclude_model_ids)]
    for metric in metrics:
        if metric == 'mse':
            working_df['sq_residuals'] = (working_df['ground_truth'] - working_df['prediction'])**2
            results = working_df.groupby(hp)['sq_residuals'].mean()
            errorbar = ci_fn('sq_residuals')
        elif metric == 'mae':
            working_df['abs_residuals'] = (working_df['ground_truth'] - working_df['prediction']).abs()
            results = working_df.groupby(hp, sort=False)['abs_residuals'].mean()
            errorbar = ci_fn('abs_residuals')
        elif metric == 'explained_var':
            working_df['sq_residuals_by_var'] = (working_df['ground_truth'] - working_df['prediction'])**2 / working_df['train_var']
            results = working_df.groupby(hp, sort=False)['sq_residuals_by_var'].mean()
            errorbar = ci_fn('sq_residuals_by_var')
        elif metric == 'diff_mae':
            n_features = len(working_df['feature'].unique())
            working_df['gt_diff'] = working_df['ground_truth'].diff(n_features)
            working_df.loc[working_df['steps']==1, 'gt_diff'] = np.nan
            working_df['pred_diff'] = working_df['prediction'].diff(n_features)
            working_df.loc[working_df['steps']==1, 'pred_diff'] = np.nan
            working_df['abs_diff_residuals'] = (working_df['gt_diff'] - working_df['pred_diff']).abs()
            results = working_df.groupby(hp, sort=False)['abs_diff_residuals'].mean()
            errorbar = ci_fn('abs_diff_residuals')
        elif metric == 'change_mae':
            n_features = len(working_df['feature'].unique())
            working_df['gt_diff'] = working_df['ground_truth'].diff(n_features)
            working_df.loc[working_df['steps']==1, 'gt_diff'] = np.nan
            working_df.loc[working_df['gt_diff']==0, 'gt_diff'] = np.nan
            working_df['pred_diff'] = working_df['prediction'].diff(n_features)
            working_df.loc[working_df['steps']==1, 'pred_diff'] = np.nan
            working_df['abs_diff_residuals'] = (working_df['gt_diff'] - working_df['pred_diff']).abs()
            results = working_df.groupby(hp, sort=False)['abs_diff_residuals'].mean()
            errorbar = ci_fn('abs_diff_residuals')
        elif metric == 'training_time':
            results = working_df.groupby(hp, sort=False)['training_time'].mean()
            errorbar = ci_fn('training_time')
        elif metric == 'n_params':
            results = working_df.groupby(hp, sort=False)['n_params'].mean()
            errorbar = ci_fn('n_params')
        elif metric == '%_completed':
            results = working_df.groupby(hp, sort=False)['prediction'].count() / working_df.groupby(hp, sort=False)['prediction'].size() * 100
            errorbar = 0
        elif metric == 'final_epoch':
            results = working_df.groupby(hp, sort=False)['final_epoch'].mean()
            errorbar = ci_fn('final_epoch')
        elif metric == 'max_lyapunov':
            results = working_df.groupby(hp, sort=False)['max_lyapunov'].mean()
            errorbar = ci_fn('max_lyapunov')
        else:
            raise NotImplementedError(f'Metric {metric} not implemented.')
        results_df[metric] = results
        errorbar_df[metric] = errorbar
    return results_df, errorbar_df

def anova_metrics(eval_df: pd.DataFrame, hyperparameters: list, metrics: list, sources_of_variance: list):
    assert 1 <= len(hyperparameters) <= 3, 'Minimum 1, maximum 3 hyperparameters for ANOVA on metrics'
    hp = hyperparameters + sources_of_variance
    working_df = eval_df.copy()
    results_list = []
    for metric in metrics:
        if metric == 'mse':
            working_df['sq_residuals'] = (working_df['ground_truth'] - working_df['prediction'])**2
            results = working_df.groupby(hp)['sq_residuals'].mean()
            metric_proxy = 'sq_residuals'
        elif metric == 'mae':
            working_df['abs_residuals'] = (working_df['ground_truth'] - working_df['prediction']).abs()
            results = working_df.groupby(hp, sort=False)['abs_residuals'].mean()
            metric_proxy = 'abs_residuals'
        elif metric == 'explained_var':
            working_df['sq_residuals_by_var'] = (working_df['ground_truth'] - working_df['prediction'])**2 / working_df['train_var']
            results = working_df.groupby(hp, sort=False)['sq_residuals_by_var'].mean()
            metric_proxy = 'sq_residuals_by_var'
        elif metric == 'diff_mae':
            working_df['gt_diff'] = working_df['ground_truth'].diff()
            working_df.loc[working_df['steps']==1, 'gt_diff'] = np.nan
            working_df['pred_diff'] = working_df['prediction'].diff()
            working_df.loc[working_df['steps']==1, 'pred_diff'] = np.nan
            working_df['abs_diff_residuals'] = (working_df['gt_diff'] - working_df['pred_diff']).abs()
            results = working_df.groupby(hp, sort=False)['abs_diff_residuals'].mean()
            metric_proxy = 'abs_diff_residuals'
        # results = results.unstack(hyperparameters)
        results = results.reset_index()
        if len(hyperparameters) == 1:
            formula = f'{metric_proxy} ~ C({hyperparameters[0]})'
        elif len(hyperparameters) == 2:
            formula = f'{metric_proxy} ~ C({hyperparameters[0]}) + C({hyperparameters[1]}) + C({hyperparameters[0]}):C({hyperparameters[1]})'
        elif len(hyperparameters) == 3:
            formula = (f'{metric_proxy} ~ C({hyperparameters[0]}) + C({hyperparameters[1]}) + C({hyperparameters[2]}) + '
                       f'C({hyperparameters[0]}):C({hyperparameters[1]}) + C({hyperparameters[0]}):C({hyperparameters[2]}) + '
                       f'C({hyperparameters[1]}):C({hyperparameters[2]})')
        lsm = sm_formula_api.ols(formula, data=results).fit()
        anova_table = sm_api.stats.anova_lm(lsm, typ=2)
        results_list.append(anova_table)
    return results_list
    

def get_model_folders(main_dir):
    """
    Returns all subdirs of main_dir which contain at least one *.pt file.
    """
    if not os.path.exists(main_dir):
        raise FileNotFoundError(f'{main_dir} not found')
    models = glob.glob(os.path.join(main_dir, '**', '*.pt'), recursive=True)
    folders = [os.path.split(m)[0] for m in models]
    folders = sorted(set(folders))
    return folders

def get_model_folders_and_preload_data(main_dir: str, load_test_data_from: str|None=None, use_tqdm: bool=False, hierarchized: bool=False):
    """
    Returns all subdirs of main_dir which contain at least one *.pt file.
    Loads the csv datasets that the model args refer to, and returns a mapping dict of model folder -> dataframe.
    If load_test_data_from is not None, additionally loads the corresponding csv datasets from there and returns another mapping dict.
    """

    model_dir_paths = get_model_folders(main_dir)
    preloaded_dataframes = {}
    train_data_mapping = {}
    test_data_mapping = {}
    if use_tqdm:
        iterator = tqdm(model_dir_paths, desc='Preloading data for models')
    else:
        iterator = model_dir_paths

    for model_dir in iterator:

        train_data_path = update_data_path(utils.load_args(model_dir))['data_path']
        if train_data_path not in preloaded_dataframes.keys():
            if hierarchized:
                preloaded_dataframes[train_data_path] = data_utils.read_data_files(train_data_path)
            else:
                preloaded_dataframes[train_data_path] = pd.read_csv(train_data_path)
        train_data_mapping[model_dir] = preloaded_dataframes[train_data_path]

        if load_test_data_from is not None:
            if hierarchized:
                test_data_path = load_test_data_from
            else:
                test_data_path = os.path.join(load_test_data_from, os.path.split(train_data_path)[1])
        else:
            test_data_path = train_data_path
        if test_data_path not in preloaded_dataframes.keys():
            if hierarchized:
                preloaded_dataframes[test_data_path] = data_utils.read_data_files(test_data_path)
            else:
                preloaded_dataframes[test_data_path] = pd.read_csv(test_data_path)
        test_data_mapping[model_dir] = preloaded_dataframes[test_data_path]

    return model_dir_paths, train_data_mapping, test_data_mapping
            

def load_model_and_data(model_dir: str, epoch_criterion: str='latest', allow_test_inputs: bool=False,
                        with_args: Optional[dict]=None, 
                        data_path: Optional[str]=None, load_test_data_from: Optional[str]=None, 
                        alternate_test_args: Optional[dict]=None, test_index: Optional[Any]=None,
                        preloaded_train_data: Optional[pd.DataFrame|list[pd.DataFrame]]=None, preloaded_test_data: Optional[pd.DataFrame|list[pd.DataFrame]]=None,
                        hierarchized: bool=False):
    """
    Returns the model with attached train dataset, and the test dataset. Usually, it loads the data according to the model args.
    If preloaded train and test data exist, you can pass them as DataFrames. Then the data will not be loaded again.
    This is handy if you want to evaluate several models on the same data, saving data loading time.
    epoch_criterion: how to choose the epoch. Options are:
        'complete':   args['n_epochs'], else None
        'latest':     latest available epoch
        'loss':       epoch with lowest train loss
        int:          specific epoch
    Specify a test_data_dir if you want the test data to be loaded from somewhere else,
    e.g. if you preprocessed differently to the train set.
    """
    def available_epochs(folder):
        epochs = []
        for f in os.listdir(folder):
            match = re.match(r'model_([0-9]+).pt', f)
            if match is not None:
                epochs.append(int(match.group(1)))
        if len(epochs) == 0:
            epochs.append(-1)
        return sorted(set(epochs))
    
    args = utils.load_args(model_dir)
    if with_args is not None:
        args.update(with_args)

    if hierarchized and not isinstance(preloaded_train_data, pd.DataFrame):
        dataset_creator = data_utils.create_dataset_for_hierarchized_model
    elif not hierarchized and not isinstance(preloaded_train_data, list):
        dataset_creator = data_utils.create_dataset_reallabor
    else:
        raise ValueError('If provided, preloaded data must be a DataFrame for hierarchized models, or a list of DataFrames for standard models.')
    
    args, train_dataset, _ = dataset_creator(args, preloaded_data=preloaded_train_data, 
                                             min_valid_training_timesteps=0, verbose='none', omit_subjects_without_indices=True)
    if test_index is not None:
        raise DeprecationWarning('The use of the test_index argument is deprecated. '
                                 'Instead, use alternate_test_args["train_on_data_until_timestep"] or alternate_test_args["train_until"].')
    test_args = args.copy()
    if alternate_test_args is not None:
        test_args.update(alternate_test_args)
    if load_test_data_from is not None:
        if hierarchized:
            test_args['data_path'] = load_test_data_from
        else:
            test_args['data_path'] = os.path.join(load_test_data_from, os.path.split(args['data_path'])[1])
    if load_test_data_from is None and preloaded_test_data is None:     # no special test data or path to load them from is specified. Load or attach train data.
        _, _, test_dataset = dataset_creator(test_args, preloaded_data=preloaded_train_data, 
                                                min_valid_training_timesteps=0, verbose='none', omit_subjects_without_indices=True)
    else:   # Load or attach test data.
        _, _, test_dataset = dataset_creator(test_args, preloaded_data=preloaded_test_data, 
                                                min_valid_training_timesteps=0, verbose='none', omit_subjects_without_indices=True)

    if not allow_test_inputs:
        if hierarchized:
            for dataset in test_dataset.datasets:
                if dataset.timeseries['inputs'] is not None:
                    dataset.timeseries['inputs'].data[:] = 0
        else:
            if test_dataset.timeseries['inputs'] is not None:
                test_dataset.timeseries['inputs'].data[:] = 0

    if isinstance(epoch_criterion, int):
        epoch = epoch_criterion
    else:
        epochs = available_epochs(model_dir)
        if epoch_criterion == 'loss':
            if os.path.exists(os.path.join(model_dir, 'loss.csv')):
                loss = pd.read_csv(os.path.join(model_dir, 'loss.csv'))
                loss = loss[[(e in epochs) for e in loss['epoch']]]
                epoch = loss['epoch'][loss['epoch_loss'].argmin()]
            else:
                raise FileNotFoundError(
                    f'No loss.csv file in model folder {model_dir}, cannot pick epoch by lowest loss.')
        elif epoch_criterion == 'complete':
            if os.path.exists(os.path.join(model_dir, f"model_{args['n_epochs']}.pt")):
                epoch = args['n_epochs']
            else:
                return None, None, None
        elif epoch_criterion == 'latest':
            epoch = max(available_epochs(model_dir))
        else:
            epoch = None
    model_class = determine_model_class(args)
    model = model_class(args)
    model.init_from_model_path(model_dir, epoch)
    model.args['model_id'] = os.path.split(os.path.split(model_dir)[0])[1]
    ###HOTFIX
    try:
        model.args['n_heads']
    except:
        pass
    else:
        model.args['latent_model'] = 'Transformer'
    ###END HOTFIX
    return model, train_dataset, test_dataset

def update_data_path(args: dict):
    """
    Updates the data path in the args dictionary to the current server's path.
    """
    if 'data_path' in args.keys():
        data_path_rest = args['data_path'].split('reallaborai4u/')[1]
        args['data_path'] = data_utils.join_base_path('reallaborai4u', data_path_rest)
    return args

def include_exclude_hypers(evaluation_df: pd.DataFrame, include_hyper: dict, exclude_hyper: dict):
    '''
    Filters rows from evaluation_df. 
    For each hyperparameter specified in <exclude_hyper>, removes all rows where this parameter has the specified value(s).
    For each hyperparameter specified in <include_hyper>, removes all rows where this parameter has values different to the specified value(s).
    '''
    for hp, hp_value in include_hyper.items():
        if isinstance(hp_value, list):
            evaluation_df = evaluation_df[np.any(
                [evaluation_df[hp] == x for x in hp_value], axis=0)]
        else:
            evaluation_df = evaluation_df[evaluation_df[hp] == hp_value]
    for hp, hp_value in exclude_hyper.items():
        if isinstance(hp_value, list):
            evaluation_df = evaluation_df[np.all(
                [evaluation_df[hp] != x for x in hp_value], axis=0)]
        else:
            evaluation_df = evaluation_df[evaluation_df[hp] != hp_value]
    return evaluation_df

def get_hypers(main_dir: str, only_varying: bool=True,
               include_hypers: Optional[list]=None, exclude_hypers: Optional[list]=None, use_tqdm: bool=True):
    '''
    Retrieves all used hyperparameter values (args) of models in main_dir and subdirs.
    If only_varying, retrieves only the hyperparameters that differ between models.
    '''
    model_folders = get_model_folders(main_dir)
    if len(model_folders)==0:
        raise FileNotFoundError(f'No models found in {main_dir}')
    args = None
    if use_tqdm:
        iterator = tqdm(model_folders)
    else:
        iterator = model_folders
    for f in iterator:
        try:
            new_args = utils.load_args(f)
        except:
            continue
        if include_hypers is not None:
            new_args = {k: new_args[k] for k in include_hypers}
        if exclude_hypers is not None:
            new_args = {k: new_args[k] for k in new_args.keys() if k not in exclude_hypers}
        if args is None:
            args = {key: [value] for key, value in new_args.items()}
        else:
            for hyper, value in new_args.items():
                if hyper in args.keys():
                    if isinstance(value, list):
                        append = True
                        for vlist in args[hyper]:
                            if all([vx in vlist for vx in value]):
                                append = False
                                break
                        if append:
                            args[hyper].append(value)
                    else:
                        if value not in args[hyper]:
                            args[hyper].append(value)
                else:
                    args[hyper] = [np.nan, value]
    if args is not None:
        if only_varying:
            args = {k:args[k] for k in args if len(args[k])>1}
        else:
            for hyper, value in args.items():
                if len(value) == 1:
                    args[hyper] = value[0]    

    return args

def summarize_runs(main_dir, leading_columns=['participant','name']):

    model_folders = get_model_folders(main_dir)
    if len(model_folders)==0:
        raise FileNotFoundError(f'No models found in {main_dir}')
    args = []
    for f in model_folders:
        new_args = utils.load_args(f)
        

# def add_hyper(hyper: str, model: Model):
#     possible_hypers = ['valid_training_data_points', 'valid_training_data_ratio']
#     if hyper==possible_hypers[0]:
#         nans = model.dataset.timeseries['emas'].data.isnan().all(dim=0).sum()
#         res = (model.dataset.timeseries['emas'].T - nans).item()
#     elif hyper==possible_hypers[1]:
#         nans = model.dataset.timeseries['emas'].data.isnan().all(dim=0).sum()
#         res = 1 - (nans/model.dataset.timeseries['emas'].T).item() 
#     else:
#         raise ValueError(f'The new hyperparameter has to be one of {possible_hypers}.')
#     return res

def complement_args_with_data_info(args: dict, test_data_dir: str|None=None, preloaded_test_data: pd.DataFrame|None=None):
    ''' Complements the model training args with entries 'valid_training_data_points', 'valid_training_data_ratio',
        and 'split', which indicates the first time step of the test set '''
    if test_data_dir is not None:
        test_data_path = os.path.join(test_data_dir, os.path.split(args['data_path'])[1])
    else:
        test_data_path = None
    args, hypo_train, _ = data_utils.create_dataset_reallabor(args, data_path=test_data_path, preloaded_data=preloaded_test_data, 
                                                              verbose='none', min_valid_training_timesteps=0)
    nans = hypo_train.timeseries['emas'].data.isnan().all(dim=1).sum()
    valid = (hypo_train.timeseries['emas'].T - nans).item()
    args['valid_training_data_points'] = valid
    args['valid_training_data_ratio'] = valid / hypo_train.timeseries['emas'].T
    args['split'] = hypo_train.n_timesteps() - 1
    return args

def load_plot_random_model(main_dir, save_path):

    model_dir_paths = get_model_folders(main_dir)
    if len(model_dir_paths) == 0:
        raise NameError('Given main_dir path does not contain models.')
    i = np.random.randint(len(model_dir_paths))
    model, train_dataset, test_dataset = load_model_and_data(model_dir_paths[i], 'latest')
    test_data, test_inputs = test_dataset.data()
    train_data, train_inputs = train_dataset.data()
    model.plot_generated_against_obs(train_data, train_inputs, 
                                     #prewarm_data=train_data[-4:], prewarm_inputs=train_inputs[-4:],
                                     plot_mean=False, ylim=(0.5, 7.5))
    plt.savefig(save_path)

def determine_model_class(args):
    if 'dim_model' in args.keys() and 'n_heads' in args.keys():
        model_class = AutoregressiveTransformer
        args['latent_model'] = 'AutoregressiveTransformer'
    elif isinstance(args['data_path'], str) and 'dim_p' in args.keys():
        model_class = HierarchizedPLRNN
    elif 'n_particles' in args.keys():
        model_class = PF_PLRNN
    elif 'PLRNN' in args['latent_model']:
        model_class = PLRNN    
    else:
        model_class = simple_models.get_class(args['latent_model'])
    return model_class

def combine_evaluation_files(paths: list, save_path: Optional[str], specifier_name: str, specifier_values: Optional[list]=None):
    dfs = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p, index_col=0)
        if specifier_values is not None:
            df[specifier_name] = specifier_values[i]
        elif specifier_name not in df.columns:
            raise ValueError(f'{specifier_name} is not in columns of {p}. You must provide specifier values.')
        dfs.append(df)
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    if save_path is not None:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        combined.to_csv(save_path)
    return combined


class ModelCatalogue:

    def __init__(self, main_dir: str, num_workers: int=1, force_new_catalogue: bool=False, only_valid_days: bool=False, **hyperparameters):
        self.main_dir = main_dir
        self.hyperparameters = hyperparameters
        self.only_valid_days = only_valid_days
        catalogue_loaded = self._load_catalogue_if_exists()
        if not catalogue_loaded or force_new_catalogue:
            if len(hyperparameters)==0:
                model_dirs = utils.get_model_dirs(main_dir)
            else:
                model_dirs = utils.filter_model_dirs_by_hyperparameters(main_dir, **hyperparameters)
            catalogue = []
            def process_model_dir(d):
                args = utils.load_args(d)
                props = pd.DataFrame(index=[0])
                props['model_dir'] = d
                props['participant'] = int(float(args['participant']))
                props['model_timestep'] = int(float(args['train_on_data_until_timestep']))
                props['model_datetime'] = 'NotImplemented'
                props['train_on_last_n_steps'] = args['train_on_last_n_steps']
                return props
            catalogue = Parallel(n_jobs=num_workers)(delayed(process_model_dir)(d) for d in model_dirs)
            self.catalogue = pd.concat(catalogue, ignore_index=True)
            self.catalogue = self.catalogue.sort_values('model_dir')
            self._save_catalogue()
        if self.only_valid_days:
            def check_if_row_is_valid(row):
                participant = row['participant']
                model_timestep = row['model_timestep']
                if participant in valid_days.columns:
                    if model_timestep in valid_days[participant].values:
                        return True
                    else:
                        return False
                else:
                    return False
            try:
                mrt = int(re.search(r'MRT([1-3])', self.main_dir).group(1))
            except:
                raise RuntimeError('Cannot determine MRT from main_dir path. Therefore, cannot filter invalid days.')
            use_days_from_file = data_utils.train_test_split_path(mrt, 'valid_first_alarms_no_con_smoothed.csv')
            valid_days = pd.read_csv(use_days_from_file, index_col=0)
            valid_days.columns = [int(c) for c in valid_days.columns]
            is_valid = self.catalogue.apply(check_if_row_is_valid, axis=1)
            self.catalogue = self.catalogue.loc[is_valid]


    def _save_catalogue(self):
        hypers_as_yaml = yaml.dump(self.hyperparameters, sort_keys=True)
        hypers_hash = hashlib.sha256(hypers_as_yaml.encode()).hexdigest()[:6]
        path = data_utils.join_ordinal_bptt_path(f'eval_reallabor/model_catalogues/{os.path.basename(self.main_dir)}/{hypers_hash}')
        os.makedirs(path, exist_ok=True)        
        with open(os.path.join(path, 'hyperparameters.yml'), 'w') as file:
            yaml.safe_dump(self.hyperparameters, file, sort_keys=True)
        self.catalogue.to_csv(os.path.join(path, 'model_catalogue.csv'))

    def _load_catalogue_if_exists(self):
        hypers_as_yaml = yaml.dump(self.hyperparameters, sort_keys=True)
        hypers_hash = hashlib.sha256(hypers_as_yaml.encode()).hexdigest()[:6]
        catalogue_path = data_utils.join_ordinal_bptt_path(f'eval_reallabor/model_catalogues/{os.path.basename(self.main_dir)}/{hypers_hash}/model_catalogue.csv')
        if os.path.exists(catalogue_path):
            self.catalogue = pd.read_csv(catalogue_path, index_col=0)
            return True
        else:
            return False

    def get_all_model_dirs(self, participant: int):
        models = self.catalogue[self.catalogue['participant']==participant].sort_values('model_timestep')
        choose_dirs = models['model_dir'].to_list()
        return choose_dirs

    def get_latest_model_dirs(self, participant: int, timestep: int|None=None, datetime: int|None=None) -> list[str]:
        if datetime is not None:
            raise NotImplementedError('Choose model dir by datetime')
        if timestep is None:
            timestep = np.inf
        models = self.catalogue[self.catalogue['participant']==participant].sort_values('model_timestep')
        if (models['model_timestep'] < timestep).any():
            models = models.loc[models['model_timestep']<timestep]
            models = models.loc[models['model_timestep']==models['model_timestep'].max()]
            choose_dirs = models['model_dir'].to_list()
        else:
            choose_dirs = []
        return choose_dirs
    
    def get_best_latest_model_dir(self, participant: int, timestep: int|None=None, datetime: int|None=None):
        model_dirs = self.get_latest_model_dirs(participant, timestep, datetime)
        if len(model_dirs) > 0:
            if len(model_dirs)>1:
                run_dirs = set([os.path.dirname(m) for m in model_dirs])
                if len(run_dirs)==1:
                    run_dir = list(run_dirs)[0]
                    best_run = utils.determine_best_run(run_dir)
                    model_dir = os.path.join(run_dir, best_run)
                else:
                    raise RuntimeError("In latest model dirs, there seem to be more than 1 model configuration. You can pick a specific configuration by specifying hyperparameters in ModelCatalogue.")
            else:
                model_dir = model_dirs[0]
        else:
            model_dir = None
        return model_dir

    def get_best_model_dirs(self, participant: int):
        models = self.catalogue.loc[self.catalogue['participant']==participant].sort_values('model_timestep')
        dirs = []
        for timestep in models['model_timestep'].unique():
            current_models = models.loc[models['model_timestep']==timestep]
            if len(current_models)>1:
                run_dirs = set([os.path.dirname(m) for m in current_models['model_dir']])
                if len(run_dirs)==1:
                    run_dir = list(run_dirs)[0]
                    best_run = utils.determine_best_run(run_dir)
                    model_dir = os.path.join(run_dir, best_run)
                else:
                    raise RuntimeError("In latest model dirs, there seem to be more than 1 model configuration. You can pick a specific configuration by specifying hyperparameters in ModelCatalogue.")
            else:
                model_dir = current_models['model_dir'].values[0]
            dirs.append(model_dir)
        return dirs
            

    
    def get_current_model_dir(self, *args, **kwargs):
        raise DeprecationWarning('get_current_model_dir is deprecated because it returns only a single dir even if there are multiple runs.'
                                 ' Use "get_latest_model_dirs" instead, then you will get all available runs.')



def summarize_training_run(main_dir, summary_file='results/training_summary.yml', ignore_hypers=['pbar_desc']):
    try:
        model_dirs = get_model_folders(main_dir)
    except Exception as e:
        print(e)
        return
    model_dates = [os.path.getmtime(d) for d in model_dirs]
    summary = {}
    name = os.path.split(main_dir)[1]
    try:
        args = get_hypers(main_dir, ignore_hypers=ignore_hypers)
    except Exception as e:
        print(e)
        return
    if isinstance(args['data_path'], str):
        args['data'] = [args['data']]
    summary['datetime'] = time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(min(model_dates)))
    summary['data_dir'] = os.path.split(args['data_path'][0])[0]
    # summary['data_version'] = time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(os.path.getmtime(args['data_path'][0])))
    summary['data_files'] = [os.path.split(v)[1] for v in args['data_path']]
    
    is_varied_hyper = lambda h: ((isinstance(args[h], list)) and (len(args[h])>1) and (h not in ['obs_features', 'input_features']))
    ignore_hyper = ['name', 'data_path', 'run']
    varied_hypers = sorted([h for h in args if ((is_varied_hyper(h)) and (h not in ignore_hyper))])
    constant_hypers = sorted([h for h in args if ((h not in varied_hypers) and (h not in ignore_hyper))])
    summary['varied_hypers'] = dict([(h, args[h]) for h in varied_hypers])    
    summary['constant_hypers'] = dict([(h, args[h]) for h in constant_hypers])
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as file:
            complete_summary = yaml.safe_load(file)
        complete_summary[name] = summary
        sorted_names = sorted(complete_summary.keys())
        complete_summary = dict([(h, complete_summary[h]) for h in sorted_names])
    else:
        complete_summary = dict([(name, summary)])
    with open(summary_file, 'w') as file:
        yaml.safe_dump(complete_summary, file, sort_keys=False)

    print(f'Successfully added {name} to training run summary')

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

def get_number_of_params(model):
    n_params = 0
    for p in model.parameters():
        if p.requires_grad:
            n_params += p.numel()
    return n_params    

def lyapunov_spectrum(model: PLRNN, x0: tc.Tensor, T: int, T_trans: int=100, ons: int=5):

    # evolve for transient time Tₜᵣ
    tmp, latent = model.generate_free_trajectory(x0, T_trans, return_hidden=True)
    if latent.ndim==2:
        latent = latent.unsqueeze(0)
    # initialize
    z = latent[:, -1]
    y = tc.zeros_like(z)
    # initialize as Identity matrix
    Q = tc.eye(z.shape[-1]).unsqueeze(0).repeat(z.shape[0], 1, 1)

    for t in range(T):        
        z = model.latent_step(z)
        try:
            jacobians = model.jacobian(z)
        except:
            states = (z, )
            jacobians = tc.autograd.functional.jacobian(model.latent_step, states)[0]
        # compute jacobian
        Q = jacobians @ Q
        if (t%ons == 0):
            # reorthogonalize
            Q, R = tc.linalg.qr(Q)
            # accumulate lyapunov exponents
            y += tc.log(tc.abs(tc.diagonal(R, dim1=-2, dim2=-1)))
    return y / T

def max_lyapunov_exponent(model: PLRNN, x0: tc.Tensor, T: int, T_trans: int=100, stop_at_convergence: bool=False, tol: float=1e-4):

    # evolve for transient time Tₜᵣ
    tmp, latent = model.generate_free_trajectory(x0, T_trans, return_hidden=True)
    if latent.ndim==2:
        latent = latent.unsqueeze(0)
    # initialize
    z = latent[:, -1]
    y = tc.zeros((z.shape[0], 1))
    # initialize as Identity matrix
    q = tc.randn_like(z)
    w = tc.zeros(z.shape[0])

    for t in range(T):        
        z = model.latent_step(z)
        try:
            jacobians = model.jacobian(z)
        except:
            states = (z, )
            jacobians = tc.autograd.functional.jacobian(model.latent_step, states)[0]
        # compute jacobian
        q = tc.einsum('bxy,by->bx', jacobians, q)
        # compute stretch
        old_w = w * 1.0
        w = tc.norm(q, dim=-1, keepdim=True)
        # accumulate lyapunov exponents
        y += tc.log(w)
        if stop_at_convergence:
            if tc.abs(tc.log(w/old_w).max()) < tol:
                break
        q = q / w
        
    return tc.squeeze(y / (t+1))


def create_ensemble_prediction_eval_df(eval_df: pd.DataFrame, outlier_threshold: Optional[float]=None, sort_result: bool=False) -> pd.DataFrame:

    eval_df = eval_df.copy()
    if outlier_threshold is not None:
        exclude_model_ids = eval_df.loc[eval_df['prediction'].abs()>outlier_threshold, 'model_id'].unique()
        eval_df = eval_df.loc[~eval_df['model_id'].isin(exclude_model_ids)]
    # ensemble_defining_cols = [col for col in eval_df.columns if col not in 
    #                           [
    #                            'ground_truth', 'gt_mean', 'gt_var', 'train_mean', 'train_var', 'prediction',
    #                            'training_time', 'n_params', 'run', 'intervention'
    #                           ]
    #                          ]
    ensemble_defining_cols = ['model_id', 'feature', 'steps', 'sample', 'prewarm_steps', 'test_day']
    agg_funcs = {col: 'mean' if pd.api.types.is_numeric_dtype(eval_df[col]) else 'first' for col in eval_df.columns}
    ensemble_eval_df = eval_df.groupby(ensemble_defining_cols, as_index=False, dropna=False, sort=sort_result).agg(agg_funcs)   # CAUTION: sort_result=True results in buggy change prediction
    return ensemble_eval_df


def identify_test_split_argument(eval_df: pd.DataFrame):
    if 'train_until' in eval_df.columns:
        return 'train_until'
    elif 'train_on_data_until_timestep' in eval_df.columns:
        return 'train_on_data_until_timestep'
    elif 'train_on_data_until_datetime' in eval_df.columns:
        return 'train_on_data_until_datetime'
    else:
        return None    
    

def clear_line_and_print(msg: Any):
    string = str(msg)
    sys.stdout.write('\r'+msg+" "*(200-len(msg)))
    sys.stdout.flush()


def gamma_weighted_pinv(B: tc.Tensor, Gamma: Optional[tc.Tensor]):
    if Gamma is None:
        B_inv = tc.pinverse(B)
    else:
        B_inv = tc.inverse(B.T @ tc.inverse(Gamma) @ B) @ B.T @ tc.inverse(Gamma)
    return B_inv

def one_shot_kalman_gain(A: tc.Tensor, B: tc.Tensor, Sigma: tc.Tensor, Gamma: tc.Tensor):
    Sigma_zz = tc.tensor(linalg.solve_discrete_lyapunov(A, Sigma), dtype=A.dtype)
    Sigma_zx = Sigma_zz @ B.T
    Sigma_xx = B @ Sigma_zz @ B.T + Gamma
    return Sigma_zx @ tc.inverse(Sigma_xx)

def get_recognition_matrix(model: PLRNN|KalmanFilter|VAR1|AutoregressiveTransformer, Gamma: Optional[tc.Tensor]=None, B: Optional[tc.Tensor]=None):
    if isinstance(model, PLRNN):
        if B is None:
            B = model.get_parameters()['B']
        recognition_model = gamma_weighted_pinv(B, Gamma)
    elif isinstance(model, KalmanFilter):
        A = model.params['A']
        B = model.params['B']
        Sigma = model.params['Sigma']
        Gamma = model.params['Gamma']
        recognition_model = one_shot_kalman_gain(A, B, Sigma, Gamma)
    else:
        recognition_model = None
    return recognition_model

def get_network_matrix(model: PLRNN|KalmanFilter|VAR1, x: Optional[tc.Tensor]=None, Gamma: Optional[tc.Tensor]=None, B: Optional[tc.Tensor]=None):

    if isinstance(model, PLRNN):
        if x is not None:
            squeeze=False
            if x.ndim==1:
                squeeze=True
                x = x.unsqueeze(0)
            if model.args['mean_centering']:
                x = x - model.data_mean
            if B is None:
                B = model.get_parameters()['B']
            B_inv = gamma_weighted_pinv(B, Gamma)
            z = tc.einsum('lo,bo->bl', B_inv, x)
            if 0 < model.args['dim_x_proj'] < model.args['dim_z']:
                z = tc.cat((z, tc.zeros((z.shape[0], model.args['dim_z']-z.shape[1]))), dim=1)
            J = model.jacobian(z)
            if 0 < model.args['dim_x_proj'] < model.args['dim_z']:
                J = J.transpose(-2, 0).transpose(-1, 1)[:3, :3].transpose(1, -1).transpose(0, -2)
            network = tc.einsum('bok,kp->bop', tc.einsum('ol,blk->bok', B, J), B_inv).detach()
            if squeeze:
                network = network.squeeze(0)
        else:
            raise ValueError('PLRNN jacobians require z')
    elif isinstance(model, KalmanFilter):
        A = model.params['A'].to(tc.float64)
        B = model.params['B'].to(tc.float64)
        Sigma = model.params['Sigma'].to(tc.float64)
        Gamma = model.params['Gamma'].to(tc.float64)
        network = B @ A @ one_shot_kalman_gain(A, B, Sigma, Gamma)
        # network = B @ A @ tc.pinverse(B)
    elif isinstance(model, VAR1):
        network = model.params['A']
    return network


def split_basis_into_kernel_and_orthogonal_complement(M: tc.Tensor):
    
    m, n = M.shape
    U, s, Vh = tc.linalg.svd(M, full_matrices=True)
    
    # Using a tolerance to account for numerical precision
    tol = max(m, n) * tc.finfo(tc.float).eps
    
    rank = tc.sum(s > tol)
    kernel_basis = Vh[rank:].T
    compl_basis = Vh[:rank].T
    
    # If the kernel is empty, return an empty array with correct shape
    if kernel_basis.size == 0:
        kernel_basis = tc.zeros((n, 0))
    if compl_basis.size == 0:
        compl_basis = tc.zeros((n, 0))
    
    return kernel_basis, compl_basis


def target_input_on_node(B: tc.Tensor, C: tc.Tensor, target_node: int, eps: float = 1e-8):
    """
    Returns u (||u||=1) maximizing alignment of Au with e_j via
    argmax_u (u^T A^T e_j e_j^T A u) / (u^T A^T A u).
    Also returns c* = e_j^T A u and Au.
    """
    A = B @ C
    G = A.T @ A
    a_j = A[target_node]
    u = tc.linalg.solve(G, a_j)
    u = u / (tc.norm(u) + eps)
    return u

def impulse_selectivity_score(B: tc.Tensor, C: tc.Tensor, u: tc.Tensor, target_node: int):
    delta_x = (B @ C @ u).abs()
    return delta_x[target_node] / tc.sum(delta_x)


def impulse_response(model: PLRNN|KalmanFilter|VAR1|AutoregressiveTransformer, u: tc.Tensor, T: int,
                     Gamma: Optional[tc.Tensor]=None, B: Optional[tc.Tensor]=None,
                     x0: Optional[tc.Tensor]=None, cumulative: bool=False, relative: bool=False):
    ''' IR is of shape (batch * T * dim_x), or (batch * dim_x) if cumulative. If x0 has no batch dimension, it is omitted.'''
    if x0 is None:
        x0 = tc.zeros(model.args['dim_x'])
    inputs = tc.zeros((T, model.args['dim_s']))
    inputs[0] = u
    recognition_matrix = get_recognition_matrix(model, Gamma=Gamma)
    ir = model.generate_free_trajectory(x0, T, inputs, 
                                        recognition_matrix=recognition_matrix,
                                        observation_matrix=B) - x0.unsqueeze(-2)
    if relative:
        ir0 = model.generate_free_trajectory(x0, T, tc.zeros((T, model.args['dim_s'])),
                                             recognition_matrix=recognition_matrix,
                                             observation_matrix=B) - x0.unsqueeze(-2)
        ir = ir - ir0
    if cumulative:
        ir = tc.sum(ir, dim=-2)
    return ir


def get_Gamma_and_B(model: PLRNN|KalmanFilter, model_dir: str, version: str):
    if isinstance(model, PLRNN):
        if version.lower() == 'pseudoinverse':
            B = model.get_parameters()['B']
            Gamma = None
        elif version.lower() == 'gaussian obs model':
            obs_model = tc.load(data_utils.join_ordinal_bptt_path(model_dir, 'gaussian_obs_model.pt'))
            B = obs_model['B']
            Gamma = obs_model['Gamma']
        elif version.lower() == 'regularized gaussian obs model':
            obs_model = tc.load(data_utils.join_ordinal_bptt_path(model_dir, 'gaussian_obs_model_B_penalty.pt'))
            B = obs_model['B']
            Gamma = obs_model['Gamma']
        elif version.lower() == 'empirical covariance':
            B = model.get_parameters()['B']
            Gamma = tc.load(data_utils.join_ordinal_bptt_path(model_dir, 'empirical_covariance.pt'))
        else:
            raise ValueError(f'Unknown version {version}')
    elif isinstance(model, KalmanFilter):
        B = model.params['B']
        Gamma = model.params['Gamma']
    else:
        B = None
        Gamma = None
    return Gamma, B


def weighted_degree_centrality(network: tc.Tensor, mode: str='out', absolute: bool=True):
    if mode=='in':
        network = network.transpose(-2, -1)
    if absolute:
        network = tc.abs(network)
    hubness = tc.sum(network, dim=-1)
    return hubness

def weighted_eigenvector_centrality(network: tc.Tensor, mode: str='out', absolute: bool=True):
    if mode=='in':
        network = network.transpose(-2, -1)
    eigvals, eigvecs = tc.linalg.eig(network)
    max_eigval_idx = tc.argmax(tc.abs(eigvals.real), dim=-1)
    if network.ndim==3:
        hubness = tc.abs(eigvecs[tc.arange(eigvecs.shape[0]), :, max_eigval_idx].real)
    else:
        hubness = tc.abs(eigvecs[:, max_eigval_idx].real)
    return hubness

def network_wasserstein_distance(network1: tc.Tensor, network2: tc.Tensor):
    return stats.wasserstein_distance(network1.flatten(), network2.flatten())

def network_spectrum_distance(network1: tc.Tensor, network2: tc.Tensor, mode: str='correlation'):
    eig1 = tc.sort(tc.linalg.eigvals(network1).real).values
    eig2 = tc.sort(tc.linalg.eigvals(network2).real).values
    if mode=='correlation':
        return 1 - tc.corrcoef(tc.stack((eig1, eig2)))[0, 1]
    elif mode=='cosine':
        return 1 - tc.dot(eig1, eig2) / (tc.norm(eig1) * tc.norm(eig2))
    elif mode=='l1':
        return tc.linalg.vector_norm(eig1 - eig2, 1)
    elif mode=='l2':
        return tc.linalg.vector_norm(eig1 - eig2, 2)
    else:
        raise ValueError('mode must be one of ["correlation", "cosine", "l1", "l2"]')

def bonferroni_holm_correction(p_values: ArrayLike, alpha: float=0.05):
    """
    Performs the Bonferroni-Holm correction for multiple hypothesis testing.
    p_values: list of p-values to be corrected
    alpha: significance level
    """
    p_values = np.asarray(p_values)
    ranks = p_values.argsort().argsort()
    m = len(ranks)
    corrected_p_values = np.array([min(p_values[i] * (m - ranks[i]), 1) for i in range(m)])
    return corrected_p_values

if __name__=='__main__':

    eval_df = pd.read_csv('/home/janik.fechtelpeter/Documents/ordinal-bptt/results/v2_MRT2_10splits_valid_ratio/00_summary_7stepsahead_ensemble/evaluation.csv')
    print(pseudo_r2_piepho(eval_df.loc[eval_df['data_dropout_to_level']>0.3], ['data_dropout_to_level'],
                           compare_to_eval_df=None, per_day=True,
                           hypers_that_apply_only_to_residual_variance=['data_dropout_to_level']))