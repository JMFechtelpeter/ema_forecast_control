import os
from typing import Optional
from collections import OrderedDict
from operator import itemgetter
import torch as tc
import pandas as pd

from ema_forecast_control.dataset.time_series_dataset import TimeSeriesDataset
from ema_forecast_control.kalman_filter.em_algorithm import stable_EM_algorithm, EM_Error
from ema_forecast_control.utils import training_utils

import logging
log = logging.getLogger(__name__)

def impute(data) -> tc.Tensor:
        df = pd.DataFrame(data)
        df.ffill(inplace=True, axis=0)
        df.bfill(inplace=True, axis=0)
        df.fillna(value=0, inplace=True)
        array = df.to_numpy()
        return tc.tensor(array) 

class KalmanFilter(tc.nn.Module):

    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.params = OrderedDict()
        self.loss_fn = tc.nn.MSELoss()
        self.optimized = False

    def init_from_model_path(self, model_path: str, *args, **kwargs):
        self.args = training_utils.load_args(model_path)
        self.params = tc.load(os.path.join(model_path, 'model.pt'))

    def save(self, model_path: str):
        tc.save(self.params, os.path.join(model_path, 'model.pt'))

    def get_parameters(self) -> OrderedDict:
        return self.params

    def generate_free_trajectory(self, x0: tc.Tensor, T: int, inputs: Optional[tc.Tensor]=None, 
                                 return_hidden: bool=False, *args, **kwargs) -> tc.Tensor|tuple[tc.Tensor, None]:
        data = x0.unsqueeze(0)
        result = self.forward(data, inputs, T, *args, **kwargs)
        if return_hidden:
            return result, None
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

    def fit_to_data(self, data: tc.Tensor, inputs: Optional[tc.Tensor]):
        if self.args['mean_centering']:
            data_mean = data.nanmean(dim=0, keepdim=True).nan_to_num(nan=0)
        else:
            data_mean = tc.zeros((1, data.shape[1]))
        data = data - data_mean
        if self.args['impute_missing_values']:
            data = impute(data)
        if inputs is not None:
            inputs = tc.nan_to_num(inputs, nan=0).T
        data = data.T
        log.info('Started EM on Kalman Filter')
        try:
            (A, B, C, Gamma, 
            Sigma, mu0, ELL) = stable_EM_algorithm(data, inputs, self.args['dim_z'], 
                                                                    max_A_eigval=self.args['max_A_eigval'],
                                                                    max_iter=1000,
                                                                    pbar_descr=self.args['pbar_descr'])
        except EM_Error as e:
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
    
    def fit(self, dataset: TimeSeriesDataset, *args, **kwargs):
        emas = dataset.data
        inputs = dataset.inputs
        self.fit_to_data(emas, inputs)
        
    def forward(self, data: tc.Tensor, inputs: Optional[tc.Tensor]=None, steps: Optional[int]=None, 
                recognition_matrix: Optional[tc.Tensor]=None, observation_matrix: Optional[tc.Tensor]=None, 
                *args, **kwargs) -> tc.Tensor:
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