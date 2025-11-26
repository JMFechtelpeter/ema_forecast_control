import os
import numpy as np
import torch as tc
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

import utils
from evaluation import mse
from evaluation import klx
from bptt import plrnn
from evaluation.pse import power_spectrum_error, power_spectrum_error_per_dim

EPOCH = None
SKIP_TRANS = 0
DATA_GENERATED = None
PRINT = False
N_STEPS = 50
# INV_PP = True

# IMPLEMENTED_METRICS = {'mse': EvaluateMSE, 'rmse': EvaluateRMSE, 'rel_mse': EvaluateRelativeMSE,
#                        'rel_rmse': EvaluateRelativeRMSE, 'feat_mse': EvaluateFeatureMSE,
#                        'feat_rel_mse': EvaluateFeatureRelativeMSE}


def get_generated_data(model, data, inputs):
    """
    Use global variable as a way to draw trajectories only once for evaluating several metrics, for speed.
    :param model:
    :return:
    """
    global DATA_GENERATED
    # Problem: if block is only entered once per training,
    # to trajectory is never updated with better models.
    if DATA_GENERATED is None:
        X, Z = model.generate_free_trajectory(data, inputs, len(data))
        DATA_GENERATED = X[SKIP_TRANS:]
    return DATA_GENERATED


def printf(x):
    if PRINT:
        print(x)


class Evaluator(object):
    def __init__(self, init_data):
        model_ids, data, inputs, save_path = init_data
        self.model_ids = model_ids
        self.save_path = save_path

        if isinstance(data, np.ndarray):
            self.data = tc.tensor(data[SKIP_TRANS:], dtype=tc.float)
        else:
            self.data = data[SKIP_TRANS:].clone().detach()
        if inputs is not None:
            self.inputs = inputs[SKIP_TRANS:].clone().detach()
        else:
            self.inputs = None

        self.name = NotImplementedError
        self.dataframe_columns = NotImplementedError

    def metric(self, model):
        return NotImplementedError

    def evaluate_metric(self):
        metric_dict = dict()
        assert self.model_ids is not None
        for model_id in self.model_ids:
            model = self.load_model(model_id)
            metric_dict[model_id] = self.metric(model)
        self.save_dict(metric_dict)

    def load_model(self, model_id):
        model = plrnn.Model()
        model.init_from_model_path(model_id, EPOCH)
        model.eval()
        return model

    def save_dict(self, metric_dict):
        df = pd.DataFrame.from_dict(data=metric_dict, orient='index')
        df.columns = self.dataframe_columns
        utils.make_dir(self.save_path)
        df.to_csv('{}/{}.csv'.format(self.save_path, self.name), sep='\t')


class EvaluateKLx(Evaluator):
    def __init__(self, init_data):
        super(EvaluateKLx, self).__init__(init_data)
        self.name = 'klx'
        self.dataframe_columns = ('klx',)

    def metric(self, model):
        self.data = self.data.to(model.device)
        if self.inputs is not None:
            self.inputs = self.inputs.to(model.device)
        data_gen = get_generated_data(model, self.data, self.inputs)
        T, dx = data_gen.size()
        if dx > 5:
            klx_value = klx.klx_metric(*self.pca(data_gen, self.data)).cpu()
        else:
            klx_value = klx.klx_metric(data_gen, self.data).cpu()

        printf('\tKLx {}'.format(klx_value.item()))
        return [np.array(klx_value.numpy())]

    @staticmethod
    def pca(x_gen, x_true):
        '''
        perform pca for to make KLx-Bin feasible for
        high dimensional data. Computes the first 5 principal
        components.
        '''
        try:
            U, S, V = tc.pca_lowrank(x_true, q=5, center=False, niter=10)
        except:
            raise
        x_pca = x_true @ V[:, :5]
        x_gen_pca = x_gen @ V[:, :5]
        return x_gen_pca, x_pca
    
    
class EvaluateMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'mse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        mse_results = mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, self.n_steps, from_step=0,
                                                 invert_preprocessing=invert_preprocessing,
                                                 prewarm=prewarm)
        non_missing_idx = np.arange(len(mse_results))[~np.isnan(mse_results)]
        for step in non_missing_idx[:3]:
            printf('\tAbsolute MSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results
    
class EvaluateRMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'rmse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        mse_results = np.sqrt(mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, self.n_steps, from_step=0,
                                                         invert_preprocessing=invert_preprocessing,
                                                         prewarm=prewarm))
        non_missing_idx = np.arange(len(mse_results))[~np.isnan(mse_results)]
        for step in non_missing_idx[:3]:
            printf('\tAbsolute RMSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results
    
class EvaluateRelativeMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'rel_mse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        train_set_mean = tc.tensor(np.nanmean(model.dataset.timeseries['emas'].train_data, axis=0))
        time_series_mean_mse = mse.const_prediction_mse(model, self.data, train_set_mean, self.n_steps,
                                                        invert_preprocessing=invert_preprocessing,
                                                        prewarm=prewarm)
        model_mse = mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, self.n_steps, from_step=0,
                                               invert_preprocessing=invert_preprocessing,
                                               prewarm=prewarm)
        mse_results = model_mse - time_series_mean_mse
        non_missing_idx = np.arange(len(mse_results))[~np.isnan(mse_results)]
        
        for step in non_missing_idx[:3]:
            printf('\tRelative MSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results
    
class EvaluateMeanMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'mean_mse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        train_set_mean = tc.tensor(np.nanmean(model.dataset.timeseries['emas'].train_data, axis=0))
        time_series_mean_mse = mse.const_prediction_mse(model, self.data, train_set_mean, self.n_steps,
                                                        invert_preprocessing=invert_preprocessing,
                                                        prewarm=prewarm)
        mse_results = time_series_mean_mse
        non_missing_idx = np.arange(len(mse_results))[~np.isnan(mse_results)]
        
        for step in non_missing_idx[:3]:
            printf('\tMean Predictor MSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results
    
class EvaluateRelativeRMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'rel_rmse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        train_set_mean = tc.tensor(np.nanmean(model.dataset.timeseries['emas'].train_data, axis=0))
        time_series_mean_mse = np.sqrt(mse.const_prediction_mse(model, self.data, train_set_mean, self.n_steps,
                                                                invert_preprocessing=invert_preprocessing,
                                                                prewarm=prewarm))
        model_mse = np.sqrt(mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, self.n_steps, from_step=0,
                                                       invert_preprocessing=invert_preprocessing,
                                                       prewarm=prewarm))
        mse_results = model_mse - time_series_mean_mse
        non_missing_idx = np.arange(len(mse_results))[~np.isnan(mse_results)]
        
        for step in non_missing_idx[:3]:
            printf('\tRelative RMSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results
    
class EvaluateFeatureMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'feat_mse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        mse_results = mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, 
                                                 self.n_steps, from_step=0, feature_mean=False,
                                                 invert_preprocessing=invert_preprocessing,
                                                 prewarm=prewarm)
        non_missing_idx = np.arange(mse_results.shape[0])[~np.isnan(mse_results.mean(1))]
        for step in non_missing_idx[:3]:
            printf('\tFeature MSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results
    
class EvaluateFeatureRelativeMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'feat_rel_mse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        train_set_mean = tc.tensor(np.nanmean(model.dataset.timeseries['emas'].train_data, axis=0))
        time_series_mean_mse = mse.const_prediction_mse(model, self.data, train_set_mean, 
                                                        self.n_steps, feature_mean=False,
                                                        invert_preprocessing=invert_preprocessing,
                                                        prewarm=prewarm)
        model_mse = mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, 
                                               self.n_steps, from_step=0, feature_mean=False,
                                               invert_preprocessing=invert_preprocessing,
                                               prewarm=prewarm)
        mse_results = model_mse - time_series_mean_mse
        non_missing_idx = np.arange(mse_results.shape[0])[~np.isnan(mse_results.mean(1))]
        for step in non_missing_idx[:3]:
            printf('\tFeature Relative MSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results
    
class EvaluateFeatureRMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'feat_rmse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        mse_results = np.sqrt(mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, 
                                                 self.n_steps, from_step=0, feature_mean=False,
                                                 invert_preprocessing=invert_preprocessing,
                                                 prewarm=prewarm))
        non_missing_idx = np.arange(mse_results.shape[0])[~np.isnan(mse_results.mean(1))]
        for step in non_missing_idx[:3]:
            printf('\tFeature RMSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results
    
class EvaluateFeatureRelativeRMSE(Evaluator):
    def __init__(self, init_data):
        super().__init__(init_data)
        self.name = 'feat_rel_rmse'
        self.n_steps = min(N_STEPS, self.data.shape[0]-1)
        self.dataframe_columns = tuple([f'{i}' for i in range(1, 1 + self.n_steps)])

    def metric(self, model, invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
        train_set_mean = tc.tensor(np.nanmean(model.dataset.timeseries['emas'].train_data, axis=0))
        time_series_mean_mse = np.sqrt(mse.const_prediction_mse(model, self.data, train_set_mean, 
                                                        self.n_steps, feature_mean=False,
                                                        invert_preprocessing=invert_preprocessing,
                                                        prewarm=prewarm))
        model_mse = np.sqrt(mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, 
                                               self.n_steps, from_step=0, feature_mean=False,
                                               invert_preprocessing=invert_preprocessing,
                                               prewarm=prewarm))
        mse_results = model_mse - time_series_mean_mse
        non_missing_idx = np.arange(mse_results.shape[0])[~np.isnan(mse_results.mean(1))]
        for step in non_missing_idx[:3]:
            printf('\tFeature Relative RMSE-{} {}'.format(step+1, mse_results[step]))
        return mse_results

    

class EvaluatePSE(Evaluator):
    def __init__(self, init_data):
        super(EvaluatePSE, self).__init__(init_data)
        self.name = 'pse'
        n_dim = self.data.shape[1]
        self.dataframe_columns = tuple(['mean'] + ['dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        self.data = self.data.to(model.device)
        data_gen = get_generated_data(model, self.data, self.inputs)

        x_gen = data_gen.cpu().numpy()
        x_true = self.data.cpu().numpy()
        pse, pse_per_dim = power_spectrum_error(x_gen=x_gen, x_true=x_true)

        printf('\tPSE {}'.format(pse))
        printf('\tPSE per dim {}'.format(pse_per_dim))
        return [pse] + pse_per_dim


class SaveArgs(Evaluator):
    def __init__(self, init_data):
        super(SaveArgs, self).__init__(init_data)
        self.name = 'args'
        self.dataframe_columns = ('dim_x', 'dim_z', 'dim_s', 'n_bases')

    def metric(self, model):
        args = model.args
        return [args['dim_x'], args['dim_z'], args['dim_s'], args['n_bases']]


def choose_evaluator_from_metric(metric_name, init_data):
    if metric_name == 'mse':
        EvaluateMetric = EvaluateMSE(init_data)
    elif metric_name == 'rmse':
        EvaluateMetric = EvaluateRMSE(init_data)
    elif metric_name == 'rel_mse':
        EvaluateMetric = EvaluateRelativeMSE(init_data)
    elif metric_name == 'rel_rmse':
        EvaluateMetric = EvaluateRelativeRMSE(init_data)
    elif metric_name == 'mean_mse':
        EvaluateMetric = EvaluateMeanMSE(init_data)
    elif metric_name == 'feat_mse':
        EvaluateMetric = EvaluateFeatureMSE(init_data)
    elif metric_name == 'feat_rel_mse':
        EvaluateMetric = EvaluateFeatureRelativeMSE(init_data)
    elif metric_name == 'feat_rmse':
        EvaluateMetric = EvaluateFeatureRMSE(init_data)
    elif metric_name == 'feat_rel_rmse':
        EvaluateMetric = EvaluateFeatureRelativeRMSE(init_data)
    elif metric_name == 'klx':
        EvaluateMetric = EvaluateKLx(init_data)
    elif metric_name == 'pse':
        EvaluateMetric = EvaluatePSE(init_data)
    else:
        raise NotImplementedError
    return EvaluateMetric


def eval_model_on_data_with_metric(model, data, metric, **kwargs):
    obs = data[0]
    inputs = data[1]
    init_data = (None, obs, inputs, None)
    EvaluateMetric = choose_evaluator_from_metric(metric, init_data)
    #EvaluateMetric.data = data
    metric_value = EvaluateMetric.metric(model, **kwargs)
    return metric_value


def is_model_id(path):
    """Check if path ends with a three digit, e.g. save/test/001 """
    run_nr = path.split('/')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers


def get_model_ids(path):
    """
    Get model ids from a directory by recursively searching all subdirectories for files ending with a number
    """
    assert os.path.exists(path)
    if is_model_id(path):
        model_ids = [path]
    else:
        all_subfolders = glob(os.path.join(path, '**/*'), recursive=True)
        model_ids = [file for file in all_subfolders if is_model_id(file)]
    assert model_ids, 'could not load from path: {}'.format(path)
    return model_ids



def evaluate_model_path(data_path, inputs_path=None, model_path=None, metrics=None):
    """Evaluate a single model in directory model_path w.r.t. metrics and save results in csv file in model_path"""
    model_ids = [model_path]
    data = utils.read_data(data_path)
    if len(data.shape)==3 and data.shape[0]==1:
        data = data.squeeze(0)
    if inputs_path is not None:
        inputs = utils.read_data(inputs_path)
        if len(inputs.shape)==3 and inputs.shape[0]==1:
            inputs = inputs.squeeze(0)
    else:
        inputs = None
    init_data = (model_ids, data, inputs, model_path)
    Save = SaveArgs(init_data)
    Save.evaluate_metric()
    global DATA_GENERATED
    DATA_GENERATED = None

    for metric_name in metrics:
        EvaluateMetric = choose_evaluator_from_metric(metric_name=metric_name, init_data=(model_ids, data, inputs, model_path))
        EvaluateMetric.evaluate_metric()

def evaluate_all_models(eval_dir, data_path, inputs_path, metrics):
    model_paths = get_model_ids(eval_dir)
    n_models = len(model_paths)
    print('Evaluating {} models'.format(n_models))
    for i, model_path in enumerate(model_paths):
        print('{} of {}'.format(i+1, n_models))
        # try:
        evaluate_model_path(data_path=data_path, model_path=model_path, metrics=metrics)
        # except:
        #     print('Error in model evaluation {}'.format(model_path))
    return

def gather_eval_results(eval_dir='save', save_path='save_eval', metrics=None, save_plots=True):
    """Pre-calculated metrics in individual model directories are gathered in one csv file"""
    if metrics is None:
        metrics = ['klx', 'pse']
    metrics.append('args')
    model_ids = get_model_ids(eval_dir)
    for metric in metrics:
        paths = [os.path.join(model_id, '{}.csv'.format(metric)) for model_id in model_ids]
        data_frames = []
        for path in paths:
            try:
                individual_results = pd.read_csv(path, sep='\t', index_col=0)
                data_frames.append(individual_results)
            except:
                print('Warning: Missing model at path: {}'.format(path))
        data_gathered = pd.concat(data_frames)
        utils.make_dir(save_path)
        metric_save_path = '{}/{}.csv'.format(save_path, metric)
        data_gathered.to_csv(metric_save_path, sep='\t')
        
        if save_plots:
            data_gathered['model'] = ['/'.join(f.split('/')[-2:]) for f in model_ids]
            data_gathered = data_gathered.set_index('model')
            data_gathered.transpose().plot()
            plt.savefig(os.path.join(save_path, f'{metric}.png'), dpi=300)
            plt.close()


if __name__ == '__main__':
    eval_dir = '/home/janik/Storage/ZI Mannheim/KI Reallabor/Identity Teacher Forcing/BPTT/results/MockData'
    data_path = '/home/janik/Storage/ZI Mannheim/KI Reallabor/data_mock/plrnn_gaussian.npy'
    inputs_path = '/home/janik/Storage/ZI Mannheim/KI Reallabor/data_mock/plrnn_inputs.npy'
    metrics = ['mse']
    evaluate_all_models(eval_dir, data_path, inputs_path, metrics)
    gather_eval_results(eval_dir=eval_dir, save_path=eval_dir, metrics=metrics)
