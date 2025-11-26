import os
import sys
import re   
from typing import Optional, Any
from collections import OrderedDict
from operator import itemgetter
from argparse import Namespace
try:
    sys.path.append('../..')
    from dataset.multimodal_dataset import MultimodalDataset
except:
    sys.path.append(os.getcwd())
    from dataset.multimodal_dataset import MultimodalDataset 
import torch as tc
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import comparison_models.simple_models.expectation_maximization as expectation_maximization
import utils

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
    elif name=='KalmanFilter':
        model = KalmanFilter
    else:
        raise NotImplementedError(name)
    return model


class SimpleModel(nn.Module):

    deterministic = True

    def __init__(self, args: dict|Namespace):
        super().__init__()
        if isinstance(args, Namespace):
            args = vars(args)
        self.args = args
        self.params = OrderedDict()
        self.loss_fn = tc.nn.MSELoss()
        self.optimized = False

    def init_from_model_path(self, model_path, *args, **kwargs):
        self.args = utils.load_args(model_path)
        self.params = tc.load(os.path.join(model_path, 'model.pt'))

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, model_path):
        tc.save(self.params, os.path.join(model_path, 'model.pt'))

    def get_parameters(self):
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
        
    def plot_generated_against_obs(self, *args, **kwargs):
        raise NotImplementedError
    
    def plot_loss(self, *args, **kwargs):
        raise NotImplementedError

    def validate(self, val_dataset: MultimodalDataset|None) -> tc.Tensor:
        if val_dataset is not None:
            obs, inputs = val_dataset.data()
            generated = self.generate_free_trajectory(obs[0], obs.shape[0],
                                                        inputs=inputs, return_hidden=False)
            validation_target = obs[1:self.args['validation_len']+1]
            generated = generated[:len(validation_target)]                
            val_loss = self.loss_fn(generated[~validation_target.isnan()], validation_target[~validation_target.isnan()]) 
        else:
            val_loss = tc.tensor(0.)
        return val_loss
    

def MovingAverage(p: int):

    class MovingAverageModel(SimpleModel):

        def fit(self, dataset: MultimodalDataset, *args, **kwargs):
            global_mean = dataset.timeseries['emas'][:int(float(self.args['train_on_data_until_timestep']))+1]
            self.params['mean'] = tc.nanmean(global_mean[-p:], dim=0)
            self.optimized = True

        def forward(self, data: tc.Tensor, inputs: Optional[Any]=None, steps: Optional[int]=None, *args, **kwargs):
            if steps is None:
                steps = data.shape[0]
            result = tc.zeros((steps, data.shape[1]))
            for t in range(steps):
                result[t] = self.params['mean']
            return result
        
        def get_p(self):
            return p
    
    return MovingAverageModel

    
class MeanPredictor(SimpleModel):

    def fit(self, dataset: MultimodalDataset, *args, **kwargs):
        self.params['mean'] = tc.nanmean(dataset.timeseries['emas'][:int(float(self.args['train_on_data_until_timestep']))+1], dim=0)
        self.optimized = True

    def forward(self, data: tc.Tensor, inputs: Optional[Any]=None, steps: Optional[int]=None, *args, **kwargs):
        if steps is None:
            steps = data.shape[0]
        result = tc.zeros((steps, data.shape[1]))
        for t in range(steps):
            result[t] = self.params['mean']
        return result
    
    
class InputsRegression(SimpleModel):

    def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor]=None, steps: Optional[int]=None, *args, **kwargs):
        if steps is None:
            steps = data.shape[0]
        mean, C, intercept = itemgetter('mean', 'C', 'intercept')(self.params)
        if inputs is None:
            inputs = tc.zeros((steps, C.shape[1]))        
        result = tc.einsum('ij,tj->ti', C, inputs) + intercept
        result = result[:steps]
        result = result + mean
        return result

    def fit(self, dataset: MultimodalDataset, *args, **kwargs):
        emas = dataset.timeseries['emas'].data
        if dataset.timeseries['inputs'] is not None:
            inputs = dataset.timeseries['inputs'].data
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
            data = SimpleModelUtils.impute(data)
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

        
class VAR1(SimpleModel):    

    def __init__(self, args: dict|Namespace):
        super().__init__(args)
        self.params['A'] = tc.zeros((self.args['dim_x'], self.args['dim_x']))
        self.params['B'] = tc.zeros((self.args['dim_x'], self.args['dim_s']))
        self.params['intercept'] = tc.zeros(self.args['dim_x'])
        self.params['mean'] = tc.zeros((1, self.args['dim_x']))
        self.params['lmbda'] = tc.nan

    def fit_to_data(self, data: tc.Tensor, inputs: Optional[tc.Tensor], *args, **kwargs):
        log.info('Started ridge regression on VAR model')
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
            data = SimpleModelUtils.impute(data)
        
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
            log.info('Finished ridge regression on VAR model')
            self.optimized = True

    def fit(self, dataset: MultimodalDataset, *args, **kwargs):
        emas = dataset.timeseries['emas'].data
        if dataset.timeseries['inputs'] is not None:
            inputs = dataset.timeseries['inputs'].data
        else:
            inputs = None
        self.fit_to_data(emas, inputs)
        
    def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor]=None, steps: Optional[int]=None, *args, **kwargs):
        if steps is None:
            steps = data.shape[0]
        result = tc.zeros((steps, data.shape[1]))
        #### HOTFIX: if A couldn't be stabilized, the params are an empty dict. Then the result must be nan.
        if len(self.params)== 5:
            mean, A, B, intercept = itemgetter('mean', 'A', 'B', 'intercept')(self.params)
        elif len(self.params)== 4:
            A, B, intercept = itemgetter('A', 'B', 'intercept')(self.params)
            mean = tc.zeros(data.shape[1])
        else:
            raise ValueError('Not all model parameters exist. Please fit the model first.')
        # ### HOTFIX: have some models where the 0-intercept is 1d
        # if intercept.ndim==1:
        #     intercept = intercept.unsqueeze(0)

        data = data - mean
        step = data[0]
        for t in range(steps):
            step = tc.einsum('ij,j->i', A, step) + intercept
            if inputs is not None:
                step += tc.einsum('ij,j->i', B, inputs[t])
            result[t] = step
        result = result + mean
        return result
        

class KalmanFilter(SimpleModel):

    deterministic = False

    def fit_to_data(self, data: tc.Tensor, inputs: Optional[tc.Tensor]):        
        if self.args['mean_centering']:
            data_mean = data.nanmean(dim=0, keepdim=True).nan_to_num(nan=0)
        else:
            data_mean = tc.zeros((1, data.shape[1]))
        data = data - data_mean
        if self.args['impute_missing_values']:
            data = SimpleModelUtils.impute(data)
        if inputs is not None:
            inputs = tc.nan_to_num(inputs, nan=0).T
        data = data.T
        log.info('Started EM on Kalman Filter')
        try:
            (A, B, C, Gamma, 
            Sigma, mu0, ELL) = expectation_maximization.EM_algorithm(data, inputs, self.args['dim_z'], 
                                                                    max_A_eigval=self.args['max_A_eigval'],
                                                                    max_iter=1000,
                                                                    pbar_descr=self.args['pbar_descr'])
        except expectation_maximization.EM_Error as e:
            log.error(f'EM algorithm terminated with error {e}.')
            self.optimized = False
        else:
            if tc.abs(tc.linalg.eig(A)[0]).max() < self.args['max_A_eigval']:
                self.params['mean'] = data_mean
                self.params['A'] = A
                self.params['B'] = B
                self.params['C'] = C
                self.params['Gamma'] = Gamma
                self.params['Sigma'] = Sigma
                self.params['mu0'] = mu0
                self.params['loss'] = ELL
                self.optimized = True
            else:
                log.error('EM algorithm did not converge to a stable model.')
                self.optimized = False
    
    def fit(self, dataset: MultimodalDataset, *args, **kwargs):
        emas = dataset.timeseries['emas'].data
        if dataset.timeseries['inputs'] is not None:
            inputs = dataset.timeseries['inputs'].data
        else:
            inputs = None
        self.fit_to_data(emas, inputs)
        
    def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor]=None, steps: Optional[int]=None, 
                recognition_matrix: Optional[tc.Tensor]=None, observation_matrix: Optional[tc.Tensor]=None, 
                *args, **kwargs):
        if steps is None:
            steps = data.shape[0]
        if not data[0].isnan().any():
            mean, A, B, C = itemgetter('mean', 'A', 'B', 'C')(self.params)
            if observation_matrix is not None:
                B = observation_matrix
            data = data - mean
            Z = tc.zeros((steps+1, self.args['dim_z']))
            if recognition_matrix is not None:
                Z[0] = tc.einsum('ij,j->i', recognition_matrix, data[0])
            else:
                Z[0] = tc.linalg.lstsq(B, data[0]).solution
            for t in range(steps):
                Z[t+1] = tc.einsum('ij,j->i', A, Z[t])
                if inputs is not None:
                    Z[t+1] += tc.einsum('ij,j->i', C, inputs[t])
            result = tc.einsum('ij,tj->ti', B, Z[1:])
            result = result + mean
        else:
            result = tc.zeros((steps, self.args['dim_x'])) * tc.nan
        return result
    
    def plot_loss(self):
        plt.figure()
        plt.plot(self.params['loss'])
        return plt.gcf()
    

class SimpleModelUtils:

    @staticmethod
    def impute(data):
        df = pd.DataFrame(data)
        df.fillna(method='ffill', inplace=True, axis=0)
        df.fillna(method='bfill', inplace=True, axis=0)
        df.fillna(value=0, inplace=True)
        array = df.to_numpy()
        return tc.tensor(array)    


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import data_utils
    # data = tc.tensor([[0.5085,0.6443,0.3507,0.6225,0.4709,0.2259,0.3111,0.9049,0.2581,0.6028],
    #                     [0.5108,0.3786,0.9390,0.5870,0.2305,0.1707,0.9234,0.9797,0.4087,0.7112],
    #                     [0.8176,0.8116,0.8759,0.2077,0.8443,0.2277,0.4302,0.4389,0.5949,0.2217],
    #                     [0.7948,0.5328,0.5502,0.3012,0.1948,0.4357,0.1848,0.1111,0.2622,0.1174]]).T
    # inputs = None
    # dataset = MultimodalDataset()
    # dataset.add_timeseries(data, 'emas')
    # dataset.add_timeseries(inputs, 'inputs')
    _, train_dataset, test_dataset = data_utils.easy_reallabor_dataset(1, 12, 'processed_csv_no_con_smoothed', 150)
    data, inputs = test_dataset.data()
    mse = []
    for model_cls in [MeanPredictor, MovingAverage(5), VAR1, KalmanFilter]:
        model = model_cls({'train_on_data_until_timestep': 150, 'impute_missing_values': 0, 'max_A_eigval': 1, 'dim_z':12, 'mean_centering': 1, 'pbar_descr': ''})
        model.fit(train_dataset)
        pred = model.forward(data, inputs, 10)
        mse.append(((pred - data[1:11])**2).mean())
    # print(pred)
    # pred = model.forward(data[5:], inputs, 4)
    # print(pred)
    # plt.figure()
    # plt.plot(pred)
    # plt.savefig('test_prediction.png')
    print(mse)