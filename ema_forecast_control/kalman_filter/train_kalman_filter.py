import argparse
import os
import torch as tc
import pandas as pd

from ema_forecast_control.utils import training_utils, path_utils, logging_utils, check_args
from ema_forecast_control.kalman_filter.kalman_filter_model import KalmanFilter

tc.set_num_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(description="ema_forecast_control Kalman Filter Training")
    parser.add_argument('--project_name', type=str, default='test_project_kalman')
    parser.add_argument('--configuration_name', type=str, default='config1')
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--overwrite', type=int, default=1)

    # general settings
    parser.add_argument('--verbose', type=str, choices=['none','print','log'], default='print')
    parser.add_argument('--pbar_descr', type=str, default='')

    # dataset
    parser.add_argument('--data_path', default=path_utils.get_data_file('processed_csv_no_con/12600_12.csv'))
    parser.add_argument('--participant', default=None)
    parser.add_argument('--obs_features', type=str, nargs='+', default=['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
                                                            'EMA_down','EMA_sad','EMA_confidence','EMA_stress','EMA_lonely',
                                                            'EMA_energetic','EMA_concentration','EMA_resilience','EMA_tired',
                                                            'EMA_satisfied', 'EMA_relaxed'])#,'EMA_emotion_control','EMA_emotion_change'])
    parser.add_argument('--input_features', type=str, nargs='+', default=['interactive1', 'interactive2', 'interactive3', 'interactive4',
                                                                'interactive5', 'interactive6','interactive7', 'interactive8'])#,
                                                                #   'EMA_sleep','EMA_joyful_day','EMA_feelactive_sincebeep','EMA_activity_pleas',
                                                                #   'EMA_social_satisfied','EMA_social_alone_yes','EMA_firstsignal'])   
    parser.add_argument('--timestamp', type=dict, default={'absolute_datetime_column':'DateTime'})
    parser.add_argument('--train_test_split', type=str, default=186)
    parser.add_argument('--train_on_last_n_steps', default=None)
    parser.add_argument('--data_dropout_to_level', default=None)
    parser.add_argument('--preprocessing', type=list, default=[
        {'time_smoothing': {'columns_to_smooth': ['EMA_mood']}},
    ]
    )
    
    # model
    parser.add_argument('--intercept', type=int, default=1)
    parser.add_argument('--dim_z', type=int, default=7)
    parser.add_argument('--max_A_eigval', type=float, default=0.999)
    parser.add_argument('--mean_centering', type=int, default=1)

    # training
    parser.add_argument('--validation_len', '-vl', type=int, default=6)
    parser.add_argument('--impute_missing_values', type=int, default=1)

    return parser

def get_args():
    parser = get_parser()
    args = vars(parser.parse_args())
    return args

def get_default_args():
    parser = get_parser()
    return {action.dest: action.default for action in parser._actions}

def train_kalman_filter(args: dict):

    args = {**get_default_args(), **args}
    args = check_args.check_args(args)

    save_path = training_utils.create_model_dir(args['project_name'], args['configuration_name'], args['run'], args['overwrite'])
    logger = logging_utils.configure_logging(save_path, args['verbose'])
    args, dataset, test_data, test_inputs = training_utils.prepare_dataset_update_args(args)
    
    training_utils.save_args(args, save_path)
    model = KalmanFilter(args)
    model.fit(dataset)
    if model.optimized:
        model.save(save_path)
        model.init_from_model_path(save_path)
        val_loss = model.validate(test_data, test_inputs)
        loss_df = pd.DataFrame(data=[[model.params['loss'][-1].item(), val_loss.item()]], index=[0], columns=['epoch_loss', 'validation_loss'])
        loss_df.to_csv(os.path.join(save_path, 'loss.csv'))
    else:
        logger.error('Kalman Filter training was not successful, model not saved.')
    return save_path



if __name__ == '__main__':
    train_kalman_filter(get_args())