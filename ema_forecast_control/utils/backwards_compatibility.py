import re
import os

from ema_forecast_control.utils import path_utils

def treat_legacy_args(args: dict):
    if 'train_on_data_until_timestep' in args:
        args['train_test_split'] = args['train_on_data_until_timestep']
    if 'model' not in args:
        if 'latent_model' in args:
            if 'PLRNN' in args['latent_model']:
                args['model'] = 'PLRNN'
            else:
                args['model'] = args['latent_model']
        elif 'dim_model' in args and 'n_heads' in args:
            args['model'] = 'Transformer'
        else:
            raise ValueError('Cannot determine model type from legacy args.')
    if 'timestamp' not in args:
        args['timestamp'] = {'absolute_datetime_column': 'DateTime'}
    if 'preprocessing' not in args:
        args['preprocessing'] = {}

    return args

def update_data_path(args: dict):
    """
    Updates the data path in the args dictionary to the current server's path.
    """
    if 'data_path' in args.keys() and 'reallaborai4u' in args['data_path']:
        if '_MRT' in args['data_path']:
            mrt = re.search(r'MRT([2-3])', args['data_path']).group(1)
        else:
            mrt = '1'  # default to MRT1 if not specified
        filename = os.path.basename(args['data_path'])
        dirname = f'AI4U_sample{mrt}'
        args['data_path'] = path_utils.join_base_path('data', dirname, filename)
    return args