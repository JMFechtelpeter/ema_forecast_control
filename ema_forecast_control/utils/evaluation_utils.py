import os
from typing import Optional
import pandas as pd
import torch as tc

from ema_forecast_control.plrnn.plrnn_model import PLRNN
from ema_forecast_control.transformer.autoregressive_transformer_model import AutoregressiveTransformer
from ema_forecast_control.kalman_filter.kalman_filter_model import KalmanFilter
from ema_forecast_control.simple_models.simple_models import VAR1, MovingAverage, InputsRegression, MeanPredictor

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

def single_forecast(model: PLRNN|SimpleModel|AutoregressiveTransformer, x0: tc.Tensor, steps: int,
                              Gamma: Optional[tc.Tensor]=None, B: Optional[tc.Tensor]=None,                              
                              prewarm_obs: Optional[tc.Tensor]=None, prewarm_inputs: Optional[tc.Tensor]=None,
                              random_z0: bool=False):

    recognition_matrix = model.get_recognition_model(Gamma=Gamma)
    observation_matrix = B          

    if random_z0:
        z0 = tc.randn(model.args['dim_z'])
    else:
        z0 = None
    generated, latent_traj = model.generate_free_trajectory(
        x0, ahead_prediction_steps, inputs=gt_inputs, z0=z0,
        prewarm_data=prewarm_data, prewarm_inputs=prewarm_inputs,
        recognition_matrix=recognition_matrix, observation_matrix=observation_matrix,
        return_hidden=True, 
    )
    generated = tc.cat([tc.full((1, generated.shape[1]), tc.nan), generated], dim=0)
    
    return generated