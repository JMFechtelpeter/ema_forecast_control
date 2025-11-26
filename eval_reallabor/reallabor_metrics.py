import sys
from typing import Optional, Callable
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import eval_reallabor_utils
from joblib import Parallel, delayed
# from multiprocessing import Pool

# @average_semi_variance_numba.njit
# def average_semi_variance_numba(data: np.ndarray):
#     ''' Calculates theta(ASV) according to Piepho (2019) with Numba optimization '''
#     cov_matrix = np.cov(data, rowvar=False)  # Compute covariance matrix
#     n = cov_matrix.shape[0]
#     cov_sum = np.sum(cov_matrix, axis=0)
#     return np.trace(cov_matrix - (1 / n) * np.tile(cov_sum, (n, 1)))

def average_semi_variance(df: pd.DataFrame):
    ''' calculates theta(ASV) according to Piepho (2019) '''
    cov_matrix = df.cov()
    n = cov_matrix.shape[0]
    return np.trace(cov_matrix - 1/n * np.tile(cov_matrix.sum(axis=0), (n, 1)))

class MetricsManager:
    
    metrics_cols = ['abs_residuals', 'sq_residuals','diff_abs_residuals', 
                    'interv_abs_residuals', 'delta_interv_abs_residuals', 
                    'interv_diff_abs_residuals', 'delta_interv_diff_abs_residuals',
                    'change_sign_correct', 'interv_change_sign_correct', 
                    'asv_residuals', 'asv_total']

    def __init__(self, eval_df: pd.DataFrame, hyperparameters: list, include_r2: bool=True, 
                 use_gt_for_predicted_difference: bool=False, only_nonzero_differences: bool=False):
        self.eval_df = eval_df
        self.use_gt_for_predicted_difference = use_gt_for_predicted_difference
        self.only_nonzero_differences = only_nonzero_differences
        self.levels = ['participant', eval_reallabor_utils.identify_test_split_argument(eval_df), 'steps', 'feature']
        self.include_r2 = include_r2
        self._build_metrics_columns()
        self.hypers = hyperparameters
        self.levels = self._remove_hyperparameters_from_levels(hyperparameters)
        self.raw_metrics = self._calculate_raw_metrics(eval_df)

    def _build_metrics_columns(self):
        self.eval_df['residuals'] = (self.eval_df['prediction'] - self.eval_df['ground_truth'])
        self.eval_df['abs_residuals'] = (self.eval_df['prediction'] - self.eval_df['ground_truth']).abs()
        self.eval_df['sq_residuals'] = (self.eval_df['prediction'] - self.eval_df['ground_truth'])**2
        diff_df = eval_reallabor_utils.create_difference_eval_df(self.eval_df, only_changes=self.only_nonzero_differences)
        self.eval_df['diff_abs_residuals'] = (diff_df['prediction'] - diff_df['ground_truth']).abs()
        self.eval_df['diff_sq_residuals'] = (diff_df['prediction'] - diff_df['ground_truth'])**2
        self.eval_df['change_sign_correct'] = (diff_df['prediction'] * diff_df['ground_truth']).apply(np.sign).replace(-1, 0)
        try:
            interv_change_df = eval_reallabor_utils.create_difference_eval_df(self.eval_df, only_interventions=True, 
                                                                              only_changes=self.only_nonzero_differences, 
                                                                              use_ground_truth_for_predicted_difference=self.use_gt_for_predicted_difference)
            self.eval_df['interv_diff_abs_residuals'] = (interv_change_df['prediction'] - interv_change_df['ground_truth']).abs()
            self.eval_df['no_interv_diff_abs_residuals'] = (interv_change_df['prediction_without_inputs'] - interv_change_df['ground_truth']).abs()
            self.eval_df['delta_interv_diff_abs_residuals'] = ((interv_change_df['prediction'] - interv_change_df['ground_truth']).abs() 
                                                          - (interv_change_df['prediction_without_inputs'] - interv_change_df['ground_truth']).abs())
            self.eval_df['interv_change_sign_correct'] = (interv_change_df['prediction'] * interv_change_df['ground_truth']).apply(np.sign).replace(-1, 0)
            interv_df = eval_reallabor_utils.create_intervention_eval_df(self.eval_df)
            self.eval_df['interv_abs_residuals'] = (interv_df['prediction'] - interv_df['ground_truth']).abs()
            self.eval_df['no_interv_abs_residuals'] = (interv_df['prediction_without_inputs'] - interv_df['ground_truth']).abs()
            self.eval_df['delta_interv_abs_residuals'] = ((interv_df['prediction'] - interv_df['ground_truth']).abs() 
                                                          - (interv_df['prediction_without_inputs'] - interv_df['ground_truth']).abs())
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"Error in creating intervention metrics: {exc_tb.tb_lineno}:{e}")
        self.eval_df['asv_residuals'] = np.nan
        self.eval_df['asv_total'] = np.nan
        
    def _calculate_raw_metrics(self, df: pd.DataFrame):
        raw_metrics = self._take_level_means_successively(df)
        if self.include_r2:
            asv_r, asv_t = self._get_asv(df)
            raw_metrics['asv_residuals'] = asv_r
            raw_metrics['asv_total'] = asv_t
        return raw_metrics

    def _get_asv(self, df: pd.DataFrame):
        levels = ['participant']
        hypers = self.hypers
        levels = [lvl for lvl in levels if lvl not in hypers]

        working_df = df.copy()
        working_df['aux_index'] = working_df.groupby('feature').cumcount()
        working_df['residuals'] = working_df['prediction'] - working_df['ground_truth']
        residuals = working_df.pivot(index=levels + hypers + ['aux_index'], columns='feature', values='residuals')
        gt = working_df.pivot(index=levels + hypers + ['aux_index'], columns='feature', values='ground_truth')
        asv_residuals = residuals.groupby(levels + hypers).apply(average_semi_variance)
        asv_total = gt.groupby(levels + hypers).apply(average_semi_variance)
        return asv_residuals, asv_total

    def _remove_hyperparameters_from_levels(self, hyperparameters: list):
        return [lvl for lvl in self.levels if lvl not in hyperparameters]
    
    def _for_each_hyperparameter(self, df: pd.DataFrame):
        if len(self.hypers)>0:
            return df.groupby(self.hypers)
        else:
            return df

    def _take_level_means_successively(self, df: pd.DataFrame):
        for l in range(len(self.levels)):
            df = df.groupby(self.levels[:len(self.levels)-l] + self.hypers).mean(numeric_only=True)
        return df
    
    # def _cohens_d(self, series: pd.Series, paired=False, correct=False, nan_policy='propagate'):
    #     df = series.unstack(self.hyperparameters)
    #     obs_difference = (df.to_numpy()[:, None] - df.to_numpy())[np.newaxis, :]
    #     # nx = (~np.isnan(x)).sum()
    #     # ny = (~np.isnan(y)).sum()
    #     if paired:
    #         d = np.nanmean(x - y, axis) / np.nanstd(x - y, ddof=1)
    #     else:
    #         raise NotImplementedError("Cohen's d for unpaired samples is not yet implemented.")
    #         dof = nx + ny - 2
    #         d = ((np.nanmean(x, axis) - np.nanmean(y, axis)) / 
    #             np.sqrt(((nx-1)*np.nanstd(x, ddof=1) ** 2 + (ny-1)*np.nanstd(y, ddof=1) ** 2) / dof))
    #         if correct:
    #             d *= (1 - 3 / (4*(nx + ny) - 9))
    #     return d

    def get_metric(self, metric: str):
        return getattr(self, metric)

    def wilcoxon(self, column: str, separately_for_each: Optional[list]=None, **kwargs):
        def perform_wilcoxon_test(df):
            local_hypers = [h for h in self.hypers if h not in separately_for_each]
            df_unstacked = df.unstack(local_hypers)
            statistic = pd.DataFrame(columns=df_unstacked.columns, index=df_unstacked.columns)
            pvalue = pd.DataFrame(columns=df_unstacked.columns, index=df_unstacked.columns)
            for c1, c2 in itertools.product(df_unstacked.columns, repeat=2):
                test_result = stats.wilcoxon(df_unstacked[c1], df_unstacked[c2], nan_policy='omit', zero_method='zsplit', **kwargs)
                statistic.loc[c1, c2] = test_result.statistic
                pvalue.loc[c1, c2] = test_result.pvalue
            res = pd.concat([statistic, pvalue], axis=1, keys=['statistic', 'pvalue'])
            return res         
        df = self.raw_metrics.copy()[column]
        if separately_for_each is not None:
            res = df.groupby(separately_for_each).apply(perform_wilcoxon_test)
        else:
            separately_for_each = []
            res = perform_wilcoxon_test(df)

        return res
    
    def ttest(self, column: str, separately_for_each: Optional[list]=None, **kwargs):
        def perform_ttest(df):
            local_hypers = [h for h in self.hypers if h not in separately_for_each]
            df_unstacked = df.unstack(local_hypers)
            deg_freedom = pd.DataFrame(columns=df_unstacked.columns, index=df_unstacked.columns)
            statistic = pd.DataFrame(columns=df_unstacked.columns, index=df_unstacked.columns)
            pvalue = pd.DataFrame(columns=df_unstacked.columns, index=df_unstacked.columns)
            for c1, c2 in itertools.product(df_unstacked.columns, repeat=2):
                test_result = stats.ttest_rel(df_unstacked[c1], df_unstacked[c2], nan_policy='omit', **kwargs)
                deg_freedom.loc[c1, c2] = test_result.df
                statistic.loc[c1, c2] = test_result.statistic
                pvalue.loc[c1, c2] = test_result.pvalue
            res = pd.concat([deg_freedom, statistic, pvalue], axis=1, keys=['df', 'statistic', 'pvalue'])
            return res         
        df = self.raw_metrics.copy()[column]
        if separately_for_each is not None:
            res = df.groupby(separately_for_each).apply(perform_ttest)
        else:
            separately_for_each = []
            res = perform_ttest(df)

        return res
    
    
    def agg(self, df: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        if df is None:
            df = self.raw_metrics.copy()
        aggregated = self._for_each_hyperparameter(df).agg(func, **kwargs)
        if isinstance(aggregated, pd.Series):
            aggregated = pd.DataFrame(aggregated).T
        return aggregated
    
    def transform(self, df: Optional[pd.DataFrame], func: Callable|str, **kwargs):
        if df is None:
            df = self.raw_metrics.copy()
        transformed = self._for_each_hyperparameter(df).transform(func, **kwargs)
        if isinstance(transformed, pd.Series):
            transformed = pd.DataFrame(transformed).T
        return transformed
    
    def r2(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', individual=False, relative_to: Optional[dict]=None, **kwargs):
        if not self.include_r2:
            raise ValueError("Metrics Manager was created with include_r2=False.")
        if raw_metrics is None:
            raw_metrics = self.raw_metrics.copy()
        if relative_to is not None:
            if 'aux_index' in raw_metrics.index.names:
                raw_metrics.index = raw_metrics.index.droplevel('aux_index')
            raw_metrics['aux_index'] = raw_metrics.groupby(list(relative_to.keys())).cumcount()
            raw_metrics.set_index('aux_index', append=True, inplace=True)
            filtered_values = raw_metrics.xs(tuple(relative_to.values()), level=tuple(relative_to.keys()))['asv_residuals']
            raw_metrics['asv_total'] = raw_metrics['asv_total'].groupby(list(relative_to.keys())).apply(lambda x: filtered_values).reorder_levels(raw_metrics.index.names)
            raw_metrics = raw_metrics.droplevel('aux_index')
        if individual:
            raw_metrics['indi_r2'] = 1 - raw_metrics['asv_residuals']/raw_metrics['asv_total']
            return self.agg(raw_metrics, func=func, **kwargs)['indi_r2'].rename('r2')
        else:
            aggregated = self.agg(raw_metrics, func=func, **kwargs)[['asv_residuals', 'asv_total']]
            return (1 - aggregated['asv_residuals']/aggregated['asv_total']).rename('r2')
    
    def rmse(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        return self.agg(self.transform(raw_metrics, np.sqrt), func=func, **kwargs)['sq_residuals'].rename('rmse')
    
    def mae(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        return self.agg(raw_metrics, func=func, **kwargs)['abs_residuals'].rename('mae')
    
    def diff_mae(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):       
        return self.agg(raw_metrics, func=func, **kwargs)['diff_abs_residuals'].rename('diff mae')
    
    def interv_mae(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        return self.agg(raw_metrics, func=func, **kwargs)['interv_abs_residuals'].rename('interv mae')
    
    def interv_diff_mae(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        return self.agg(raw_metrics, func=func, **kwargs)['interv_diff_abs_residuals'].rename('interv diff mae')
    
    def change_sign_correct(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        return self.agg(raw_metrics, func=func, **kwargs)['change_sign_correct'].rename('change sign correct')

    def interv_change_sign_correct(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        return self.agg(raw_metrics, func=func, **kwargs)['interv_change_sign_correct'].rename('interv change sign correct')
    
    def delta_interv_mae(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        return self.agg(raw_metrics, func=func, **kwargs)['delta_interv_abs_residuals'].rename('delta interv mae')
    
    def delta_interv_diff_mae(self, raw_metrics: Optional[pd.DataFrame]=None, func: Callable|str='mean', **kwargs):
        return self.agg(raw_metrics, func=func, **kwargs)['delta_interv_diff_abs_residuals'].rename('delta interv diff mae')


class BootstrapConfidenceIntervals(MetricsManager):


    def __init__(self, eval_df: pd.DataFrame, hyperparameters: list, samples: int, agg_func: str='mean', within_subjects: bool=False, 
                 use_tqdm: bool=True, first_seed: Optional[int]=None, num_workers: int=1, include_r2: bool=True, **agg_func_kwargs):
        super().__init__(eval_df, hyperparameters, include_r2)
        self.use_tqdm = use_tqdm
        self.seed = first_seed
        self.within_subjects = within_subjects
        self.bootstrap_samples = self._generate_bootstrap_samples(samples, within_subjects, use_tqdm, num_workers)
    
    def _sample(self, df: pd.DataFrame):
        if self.seed is not None:
            self.seed += 1
        return self._for_each_hyperparameter(df).sample(frac=1, replace=True, random_state=self.seed)

    def _generate_bootstrap_samples(self, samples: int, within_subjects: bool, use_tqdm: bool, num_workers: int):

        def process_sample(seed_offset):
            working_df = self.eval_df.copy()
            if self.seed is not None:
                np.random.seed(self.seed + seed_offset)
            if within_subjects:
                working_df = self._sample(working_df)
                working_df = self._calculate_raw_metrics(working_df)
            else:
                working_df = self._calculate_raw_metrics(working_df)
                working_df = self._sample(working_df)
            return working_df[self.metrics_cols]

        if use_tqdm and samples > 0:
            iterator = tqdm(range(samples), desc=f'Generating bootstrap samples...')
        else:
            iterator = range(samples)

        bootstrap_samples = Parallel(n_jobs=num_workers)(
            delayed(process_sample)(n) for n in iterator
        )
        return bootstrap_samples
    
    def _get_ci(self, df: pd.DataFrame, confidence: float, relative_to: Optional[pd.Series]):
        ci_low, ci_high = (100-confidence)/2, (100+confidence)/2
        ci_lower = df.quantile(ci_low / 100, axis=1)
        ci_upper = df.quantile(ci_high / 100, axis=1)
        if relative_to is not None:
            ci_lower = relative_to - ci_lower
            ci_upper = ci_upper - relative_to
        ci = pd.concat([ci_lower, ci_upper], axis=1, keys=['ci_lower', 'ci_upper'])
        if relative_to is not None:
            ci = ci.clip(lower=0)
        return ci
    
    def get_metric_ci(self, metric: str, confidence: float=95.0, relative_ci: bool=True, **kwargs):
        fn = self.get_metric(metric)
        statistic = fn(self.raw_metrics, **kwargs)
        metric_samples = [fn(bs, **kwargs) for bs in self.bootstrap_samples]
        metric_samples = pd.concat(metric_samples, axis=1)
        relative_to = statistic if relative_ci else None
        ci = self._get_ci(metric_samples, confidence, relative_to)
        return statistic, ci


class BootstrapHypothesisTester(MetricsManager):

    def __init__(self, eval_df: pd.DataFrame, hyperparameters: list, samples: int, 
                 agg_func: str='mean', within_subjects: bool=False, 
                 use_tqdm: bool=True, first_seed: Optional[int]=None, num_workers: int=1, 
                 separately_for_each: Optional[list]=None, include_r2: bool=True, **agg_func_kwargs):
        super().__init__(eval_df, hyperparameters, include_r2)
        self.use_tqdm = use_tqdm
        self.seed = first_seed
        self.agg_func = agg_func
        self.agg_func_kwargs = agg_func_kwargs
        self.within_subjects = within_subjects
        self.separate = separately_for_each
        self.null_samples = self._generate_samples_for_null_distribution(samples, within_subjects, use_tqdm, num_workers)

    def _resample_with_reassigned_hyperparameters(self, df: pd.DataFrame):
        if self.seed is not None:
            self.seed += 1
        resampled = df.copy()
        # non_hyper_cols = [col for col in resampled.columns if col not in self.hypers]
        if self.separate is not None:
            resampled[self.metrics_cols] = resampled.groupby(self.separate)[self.metrics_cols].sample(frac=1, replace=True, random_state=self.seed).to_numpy(dtype=float)
        else:
            resampled[self.metrics_cols] = resampled[self.metrics_cols].sample(frac=1, replace=True, random_state=self.seed).to_numpy(dtype=float)
        return resampled
    
    def _generate_samples_for_null_distribution(self, samples: int, within_subjects: bool, use_tqdm: bool, num_workers: int):

        def process_sample(seed_offset):
            working_df = self.eval_df.copy()
            if self.seed is not None:
                np.random.seed(self.seed + seed_offset)
            if within_subjects: 
                working_df = self._resample_with_reassigned_hyperparameters(working_df)
                working_df = self._take_level_means_successively(working_df)
            else:
                working_df = self._take_level_means_successively(working_df)
                working_df = self._resample_with_reassigned_hyperparameters(working_df)
            return working_df[self.metrics_cols]

        if use_tqdm and samples > 0:
            iterator = tqdm(range(samples), desc=f'Generating null distribution samples...')
        else:
            iterator = range(samples)

        bootstrap_samples = Parallel(n_jobs=num_workers)(
            delayed(process_sample)(n) for n in iterator
        )
        return bootstrap_samples

    def _bootstrap_hypothesis_test(self, obs_statistic: pd.Series, statistic_samples: pd.DataFrame, alternative: str):
        obs_difference = (obs_statistic.to_numpy()[:, None] - obs_statistic.to_numpy())[np.newaxis, :]
        difference_samples = statistic_samples.to_numpy()[:, None, :] - statistic_samples.to_numpy()[None, :, :]  # Broadcasting
        difference_samples = np.transpose(difference_samples, (2, 0, 1))
        if alternative=='two-sided':
            more_extreme_samples = (np.abs(difference_samples) >= np.abs(obs_difference)).mean(axis=0)
        elif alternative=='greater':
            more_extreme_samples = (difference_samples >= obs_difference).mean(axis=0)
        elif alternative=='less':
            more_extreme_samples = (difference_samples <= obs_difference).mean(axis=0)
        else:
            raise ValueError("Bootstrap Hypothesis Test: Alternative must be one of ['two-sided', 'greater', or 'less'].")
        p = pd.DataFrame(index=obs_statistic.index, columns=obs_statistic.index, data=more_extreme_samples)
        if self.separate is not None:
            for sep in self.separate:
                p = p.groupby(sep, group_keys=False).apply(lambda x: x.xs(x.name, level=sep, axis=1))
        return p
    
    def get_metric_test(self, metric: str, alternative: str='two-sided', **kwargs):
        fn = self.get_metric(metric)
        statistic = fn(self.raw_metrics, self.agg_func, **kwargs, **self.agg_func_kwargs)
        metric_samples = [fn(bs, func=self.agg_func, **kwargs, **self.agg_func_kwargs) for bs in self.null_samples]
        metric_samples = pd.concat(metric_samples, axis=1)
        p = self._bootstrap_hypothesis_test(statistic, metric_samples, alternative)
        return statistic, p