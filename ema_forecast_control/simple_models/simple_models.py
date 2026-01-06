import os
from typing import Optional
from collections import OrderedDict
from operator import itemgetter
import re
import torch as tc
import torch.nn as nn
import pandas as pd

from ema_forecast_control.dataset.time_series_dataset import TimeSeriesDataset
from ema_forecast_control.utils import training_utils

import logging
log = logging.getLogger(__name__)

def get_class(name: str):
    if name=='MeanPredictor':
        model = MeanPredictor
    elif name=='InputsRegression':
        model = InputsRegression
    elif name.startswith('MovingAverage'):
        match = re.match(r'MovingAverage\(([0-9]+)\)', name)
        if match is not None:
            p = match.group(1)
        else:
            raise NotImplementedError(name)
        model = MovingAverage(int(p))
    elif name=='VAR1':
        model = VAR1
    else:
        raise NotImplementedError(name)
    return model

def impute(data) -> tc.Tensor:
        df = pd.DataFrame(data)
        df.ffill(inplace=True, axis=0)
        df.bfill(inplace=True, axis=0)
        df.fillna(value=0, inplace=True)
        array = df.to_numpy()
        return tc.tensor(array)  

class SimpleModel(nn.Module):

    deterministic = True

    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.params = OrderedDict()
        self.loss_fn = tc.nn.MSELoss()
        self.optimized = False

    def init_from_model_path(self, model_path, *args, **kwargs):
        self.args = training_utils.load_args(model_path)
        self.params = tc.load(os.path.join(model_path, 'model.pt'))

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, model_path):
        tc.save(self.params, os.path.join(model_path, 'model.pt'))

    def get_parameters(self) -> OrderedDict:
        return self.params

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def generate_free_trajectory(self, x0: tc.Tensor, T: int, inputs: Optional[tc.Tensor]=None, return_hidden: bool=False, *args, **kwargs) -> tuple[tc.Tensor, str]:
        data = x0.unsqueeze(0)
        result = self.forward(data, inputs, T, *args, **kwargs)
        if return_hidden:
            return result, 'placeholder for latent traj'
        else:
            return result

    def validate(self, validation_data: Optional[tc.Tensor], validation_inputs: Optional[tc.Tensor]) -> tc.Tensor:
        if validation_data is not None:
            generated = self.generate_free_trajectory(validation_data[0], validation_data.shape[0],
                                                        inputs=validation_inputs, return_hidden=False)
            validation_target = validation_data[1:self.args['validation_len']+1]
            generated = generated[:len(validation_target)]                
            val_loss = self.loss_fn(generated[~validation_target.isnan()], validation_target[~validation_target.isnan()]) 
        else:
            val_loss = tc.tensor(0.)
        return val_loss
    

def MovingAverage(p: int):

    class MovingAverageModel(SimpleModel):

        def fit(self, dataset: TimeSeriesDataset, *args, **kwargs):
            self.params['mean'] = tc.nanmean(dataset.data[-p:], dim=0)
            self.loss = self.loss_fn(dataset.data[dataset.valid_indices], self.params['mean'].repeat(dataset.n_valid, 1))
            self.optimized = True

        def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor]=None, steps: Optional[int]=None, *args, **kwargs) -> tc.Tensor:
            if steps is None:
                steps = data.shape[0]
            result = tc.zeros((steps, data.shape[1]))
            for t in range(steps):
                result[t] = self.params['mean']
            return result
        
        def get_p(self) -> int: 
            return p
    
    return MovingAverageModel

    
class MeanPredictor(SimpleModel):

    def fit(self, dataset: TimeSeriesDataset, *args, **kwargs):
        self.params['mean'] = tc.nanmean(dataset.data, dim=0)
        self.loss = self.loss_fn(dataset.data[dataset.valid_indices], self.params['mean'].repeat(dataset.n_valid, 1))
        self.optimized = True

    def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor]=None, steps: Optional[int]=None, *args, **kwargs) -> tc.Tensor:
        if steps is None:
            steps = data.shape[0]
        result = tc.zeros((steps, data.shape[1]))
        for t in range(steps):
            result[t] = self.params['mean']
        return result
    
    
class InputsRegression(SimpleModel):

    def fit(self, dataset: TimeSeriesDataset, *args, **kwargs):
        emas = dataset.data
        if dataset.inputs is not None:
            inputs = dataset.inputs
        else:
            raise ValueError('Dataset has no inputs. Without inputs, InputRegression is equivalent to MeanPredictor.')
        self.fit_to_data(emas, inputs)
        self.optimized = True

    def fit_to_data(self, data: tc.Tensor, inputs: Optional[tc.Tensor]):
        if self.args['mean_centering']:
            data_mean = data.nanmean(dim=0, keepdim=True).nan_to_num(nan=0)
        else:
            data_mean = tc.zeros((1, data.shape[1]))
        data = data - data_mean
        if inputs is None:
            inputs = tc.zeros((data.shape[0], 0))
        else:
            inputs = tc.nan_to_num(inputs, nan=0)
        if self.args['impute_missing_values']:
            data = impute(data)
        if self.args['intercept']:
            combined_predictor = tc.hstack((tc.ones((data.shape[0], 1)), inputs))[:-1]
        else:
            combined_predictor = inputs[:-1]
        target = data[1:]
        valid = (~tc.isnan(combined_predictor).any(dim=1)) & (~tc.isnan(target).any(dim=1))
        combined_predictor = combined_predictor[valid]
        target = target[valid]
        moment_matrix = combined_predictor.T @ combined_predictor
        regression_weights = tc.linalg.pinv(moment_matrix) @ combined_predictor.T @ target
        if self.args['intercept']:
            intercept = regression_weights[0]
            C = regression_weights[1:].T
        else:
            intercept = tc.zeros(data.shape[1])
            C = regression_weights.T
        self.params['C'] = C
        self.params['intercept'] = intercept
        self.params['mean'] = data_mean
        self.loss = self.loss_fn(target, combined_predictor @ regression_weights)

    def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor]=None, steps: Optional[int]=None, *args, **kwargs) -> tc.Tensor:
        if steps is None:
            steps = data.shape[0]
        mean, C, intercept = itemgetter('mean', 'C', 'intercept')(self.params)
        if inputs is None:
            inputs = tc.zeros((steps, C.shape[1]))        
        result = tc.einsum('ij,tj->ti', C, inputs) + intercept
        result = result[:steps]
        result = result + mean
        return result    

        
class VAR1(SimpleModel):    

    def __init__(self, args: dict):
        super().__init__(args)
        self.params['A'] = tc.zeros((self.args['dim_x'], self.args['dim_x']))
        self.params['B'] = tc.zeros((self.args['dim_x'], self.args['dim_s']))
        self.params['intercept'] = tc.zeros(self.args['dim_x'])
        self.params['mean'] = tc.zeros((1, self.args['dim_x']))
        self.params['lmbda'] = tc.nan

    def fit_to_data(self, data: tc.Tensor, inputs: Optional[tc.Tensor], *args, **kwargs):
        if self.args['mean_centering']:
            data_mean = data.nanmean(dim=0, keepdim=True).nan_to_num(nan=0)
        else:
            data_mean = tc.zeros((1, data.shape[1]))
        data = data - data_mean
        if inputs is None:
            inputs = tc.zeros((data.shape[0], 0))
        else:
            inputs = tc.nan_to_num(inputs, nan=0)
        if self.args['impute_missing_values']:
            data = impute(data)
        
        if self.args['intercept']:
            combined_predictor = tc.hstack((tc.ones((data.shape[0], 1)), data, inputs))[:-1]
        else:
            combined_predictor = tc.hstack((data, inputs))[:-1]
        target = data[1:]
        valid = (~tc.isnan(combined_predictor).any(dim=1)) & (~tc.isnan(target).any(dim=1))
        combined_predictor = combined_predictor[valid]
        target = target[valid]        
        size = combined_predictor.shape[1]
        stabilized = False
        for lmbda in tc.arange(0,10.5,0.01):
            moment_matrix = combined_predictor.T @ combined_predictor + lmbda * tc.eye(size)
            regression_weights = tc.linalg.pinv(moment_matrix) @ combined_predictor.T @ target
            if self.args['intercept']:
                intercept = regression_weights[0]
                A = regression_weights[1:data.shape[1]+1].T
                B = regression_weights[data.shape[1]+1:].T
            else:
                intercept = tc.zeros(data.shape[1])
                A = regression_weights[:data.shape[1]].T
                B = regression_weights[data.shape[1]:].T
            lmbda = lmbda
            if tc.abs(tc.linalg.eig(A)[0]).max() < self.args['max_A_eigval']:
                self.params['mean'] = data_mean
                self.params['A'] = A
                self.params['B'] = B
                self.params['intercept'] = intercept
                self.params['lmbda'] = lmbda
                self.loss = self.loss_fn(target, combined_predictor @ regression_weights)
                stabilized = True
                break
        if not stabilized:
            log.error(f'Ridge Regression did not converge to a model with maximum eigenvalue < {self.args["max_A_eigval"]} for lambda < 10.5.')
            self.optimized = False
        else:
            if tc.numel(inputs) == 0:
                self.params['B'] = None
            if self.args['pbar_descr'] is not None:
                print(self.args['pbar_descr'])
            self.optimized = True

    def fit(self, dataset: TimeSeriesDataset, *args, **kwargs):
        emas = dataset.data
        if dataset.inputs is not None:
            inputs = dataset.inputs
        else:
            inputs = None
        self.fit_to_data(emas, inputs)
        
    def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor]=None, steps: Optional[int]=None, *args, **kwargs) -> tc.Tensor:
        if steps is None:
            steps = data.shape[0]
        result = tc.zeros((steps, data.shape[1]))
        if len(self.params)== 5:
            mean, A, B, intercept = itemgetter('mean', 'A', 'B', 'intercept')(self.params)
        elif len(self.params)== 4:
            A, B, intercept = itemgetter('A', 'B', 'intercept')(self.params)
            mean = tc.zeros(data.shape[1])
        else:
            raise ValueError('Not all model parameters exist. Please fit the model first.')

        data = data - mean
        step = data[0]
        for t in range(steps):
            step = tc.einsum('ij,j->i', A, step) + intercept
            if inputs is not None:
                step += tc.einsum('ij,j->i', B, inputs[t])
            result[t] = step
        result = result + mean
        return result