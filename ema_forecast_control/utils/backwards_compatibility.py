from ema_forecast_control.simple_models import simple_models
from ema_forecast_control.plrnn.plrnn_model import PLRNN
from ema_forecast_control.kalman_filter.kalman_filter_model import KalmanFilter
from ema_forecast_control.transformer.autoregressive_transformer_model import AutoregressiveTransformer

def determine_model_class(args):
    if 'dim_model' in args.keys() and 'n_heads' in args.keys():
        model_class = AutoregressiveTransformer
        args['latent_model'] = 'AutoregressiveTransformer'
    elif 'PLRNN' in args['latent_model']:
        model_class = PLRNN    
    elif 'KalmanFilter' in args['latent_model']:
        model_class = KalmanFilter
    else:
        model_class = simple_models.get_class(args['latent_model'])
    return model_class

def treat_legacy_args(args):
    if 'train_on_data_until_timestep' in args:
        args['train_test_split'] = args['train_on_data_until_timestep']
    return args