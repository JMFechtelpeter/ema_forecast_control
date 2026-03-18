import argparse
import torch as tc

from ema_forecast_control.utils import training_utils, path_utils, logging_utils, check_args
from ema_forecast_control.plrnn.bptt_algorithm import BPTT
from ema_forecast_control.plrnn.plrnn_model import PLRNN

tc.set_num_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(description="ema_forecast_control PLRNN Training")
    parser.add_argument('--project_name', type=str, default='test_project')
    parser.add_argument('--configuration_name', type=str, default='config1')
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--overwrite', type=int, default=1)

    # gpu
    parser.add_argument('--use_gpu', type=int, default=0)
    parser.add_argument('--device_id', type=int, default=-1)     # cuda:0, cuda:1 etc. If -1, find GPU with lowest utilization

    # general settings
    parser.add_argument('--verbose', type=str, choices=['none','print','log'], default='print')
    parser.add_argument('--pbar_descr', type=str, default='')

    # dataset
    parser.add_argument('--data_path', default=path_utils.get_data_file('processed_csv_no_con/12600_12.csv'))
    parser.add_argument('--participant', default=None)
    parser.add_argument('--obs_features', type=str, nargs='+', default=['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
                                                               'EMA_down','EMA_sad','EMA_confidence','EMA_stress','EMA_lonely',
                                                               'EMA_energetic','EMA_concentration','EMA_resilience','EMA_tired',
                                                               'EMA_satisfied','EMA_relaxed'])#,'EMA_emotion_control','EMA_emotion_change'])
    parser.add_argument('--input_features', type=str, nargs='+', default=['interactive1','interactive2','interactive3','interactive4',
                                                                'interactive5','interactive6','interactive7','interactive8'])
                                                                # 'EMA_sleep','EMA_joyful_day','EMA_feelactive_sincebeep','EMA_activity_pleas',
                                                                # 'EMA_social_satisfied','EMA_social_alone_yes','EMA_firstsignal'])   
    parser.add_argument('--timestamp', type=dict, default={'absolute_datetime_column':'DateTime'})
    parser.add_argument('--train_test_split', type=str, default=186)
    parser.add_argument('--train_on_last_n_steps', default=None)
    parser.add_argument('--data_dropout_to_level', default='0.5')
    parser.add_argument('--preprocessing', type=list, default=[
        {'time_smoothing': {'columns_to_smooth': ['EMA_mood']}},
    ]
    )
    
    # resume from a model checkpoint
    parser.add_argument('--load_model_path', type=str, default=None)
    # epoch is inferred if None
    parser.add_argument('--resume_epoch', type=int, default=None)

    # model
    parser.add_argument('--latent_model', '-ml', type=str,
                        choices=PLRNN.LATENT_MODELS, default='clipped-shallow-PLRNN')
    parser.add_argument('--dim_z', type=int, default=20) 
    parser.add_argument('--dim_x_proj', type=int, default=0)    # if >0, an observation model will be learnt
    parser.add_argument('--clip_range', '-clip', type=float, default=10)    
    parser.add_argument('--mean_centering', type=int, default=1)

    # shallow PLRNN args
    parser.add_argument('--dim_y', type=int, default=15) 

    # BPTT
    parser.add_argument('--tf_alpha', '-ta', type=float, default=1) 
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--batches_per_epoch', '-bpi', type=int, default=0)
    parser.add_argument('--seq_len', '-sl', type=int, default=30)

    # training
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--lr_annealing', '-lra', type=int, default=1)
    parser.add_argument('--n_epochs', '-n', type=int, default=10)
    parser.add_argument('--gradient_clipping', '-gc', type=float, default=10.)
    parser.add_argument('--model_save_step', default='best')
    parser.add_argument('--info_save_step', type=int, default=5)
    parser.add_argument('--early_stopping', type=int, default=1)
    parser.add_argument('--validation_len', type=int, default=7)
    parser.add_argument('--validation_prewarming', type=int, default=0)


    return parser

def get_args():
    parser = get_parser()
    args = vars(parser.parse_args())
    return args

def get_default_args():
    parser = get_parser()
    return {action.dest: action.default for action in parser._actions}

def train_plrnn(args: dict) -> str:

    args = {**get_default_args(), **args}
    args = check_args.check_args(args)

    save_path = training_utils.create_model_dir(args['project_name'], args['configuration_name'], args['run'], args['overwrite'])
    log = logging_utils.configure_logging(save_path, args['verbose'])
    device = training_utils.prepare_device(args)
    args, dataset, test_data, test_inputs = training_utils.prepare_dataset_update_args(args)

    training_utils.save_args(args, save_path)

    training_algorithm = BPTT(args, dataset, test_data, test_inputs, save_path, device)
    training_algorithm.train()

    args['final_epoch'] = training_algorithm.final_epoch
    training_utils.save_args(args, save_path)
    return save_path


if __name__ == '__main__':
    train_plrnn(get_args())