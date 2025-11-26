# main.py 
import argparse
import os
import logging
from typing import Union, Optional
import torch as tc

import utils
import data_utils
from bptt.bptt_algorithm import BPTT
from bptt.plrnn import PLRNN

tc.set_num_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(description="TF RNN Training")
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--experiment', type=str, default='_debug')
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--overwrite', type=int, default=1)

    # gpu
    parser.add_argument('--use_gpu', type=int, default=0)
    parser.add_argument('--device_id', type=int, default=-1)     # cuda:0, cuda:1 etc. If -1, find GPU with lowest utilization

    # general settings
    parser.add_argument('--verbose', type=str, choices=['none','print','log'], default='print')
    parser.add_argument('--use_tb', type=int, default=0)
    parser.add_argument('--plot_trajectories_after_training', type=int, default=1)
    parser.add_argument('--plot_loss_after_training', type=int, default=1)
    parser.add_argument('--pbar_descr', type=str, default='')

    # dataset
    parser.add_argument('--data_path', default=data_utils.get_data_file(1, 12, 'processed_csv_no_con_smoothed'))
    parser.add_argument('--participant', default=None)
    parser.add_argument('--obs_features', type=str, nargs='+', default=['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
                                                               'EMA_down','EMA_sad','EMA_confidence','EMA_stress','EMA_lonely',
                                                               'EMA_energetic','EMA_concentration','EMA_resilience','EMA_tired',
                                                               'EMA_satisfied','EMA_relaxed'])#,'EMA_emotion_control','EMA_emotion_change'])
    parser.add_argument('--input_features', type=str, nargs='+', default=['interactive1','interactive2','interactive3','interactive4',
                                                                'interactive5','interactive6','interactive7','interactive8'])
                                                                # 'EMA_sleep','EMA_joyful_day','EMA_feelactive_sincebeep','EMA_activity_pleas',
                                                                # 'EMA_social_satisfied','EMA_social_alone_yes','EMA_firstsignal'])   
    parser.add_argument('--train_on_data_until_timestep', type=str, default=186)
    parser.add_argument('--train_on_data_until_datetime', type=str, default='None')
    parser.add_argument('--train_on_last_n_steps', default=None)
    parser.add_argument('--data_dropout_to_level', default='0.5')
    
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

    # ALRNN args
    parser.add_argument('--nonlinear_units', type=int, default=2)

    # BPTT
    parser.add_argument('--tf_alpha', '-ta', type=float, default=1) 
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--batches_per_epoch', '-bpi', type=int, default=0)
    parser.add_argument('--seq_len', '-sl', type=int, default=30)

    # training
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--lr_annealing', '-lra', type=int, default=1)
    parser.add_argument('--n_epochs', '-n', type=int, default=1000)
    parser.add_argument('--gradient_clipping', '-gc', type=float, default=10.)
    parser.add_argument('--data_augmentation', type=int, default=0)
    parser.add_argument('--model_save_step', default='best')
    parser.add_argument('--info_save_step', type=int, default=5)
    parser.add_argument('--validation_len', '-vl', type=Optional[int], default=7)       # if this is positive, will create a validation set and validate model on this
    parser.add_argument('--validation_prewarming', type=int, default=12) 
    parser.add_argument('--early_stopping', type=int, default=1)
    parser.add_argument('--use_differences_for_loss', type=int, default=1)
    parser.add_argument('--use_abs_values_for_loss', type=int, default=0)

    # regularization
    # parser.add_argument('--use_reg', '-r', type=int, default=0)
    # parser.add_argument('--reg_ratios', '-rr', nargs='*', type=float, default=[0.5])
    # parser.add_argument('--reg_alphas', '-ra', nargs='*', type=float, default=[.1])
    # parser.add_argument('--reg_norm', '-rn', type=str, choices=['l2', 'l1'], default='l2')

    # default data and input size; this is automatically updated by the create_dataset_reallabor method
    parser.add_argument('--dim_x', type=int, default=15)
    parser.add_argument('--dim_s', type=int, default=2)
    return parser

def get_args():
    parser = get_parser()
    return parser.parse_args()

def get_default_args():
    parser = get_parser()
    return {action.dest: action.default for action in parser._actions}

def configure_logging(path, verbose):    
    logger = logging.getLogger('root')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    if verbose=='log':
        hdlr = logging.FileHandler(os.path.join(path, 'log.txt'))
    elif verbose=='print':
        hdlr = logging.StreamHandler()
    else:
        hdlr = logging.NullHandler()
    hdlr.setLevel(logging.INFO) 
    formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger

def train(args):
    save_path = utils.create_savepath(args)
    #writer = utils.init_writer(args, save_path)
    logger = configure_logging(save_path, args.verbose)
    device = utils.prepare_device(args)
    args, dataset, test_dataset = data_utils.create_dataset_reallabor(args, min_valid_training_timesteps=10)
    # args, dataset, test_dataset = utils.create_dataset(args)

    utils.check_args(args)
    utils.save_args(args, save_path)

    training_algorithm = BPTT(args, dataset, test_dataset,
                                              save_path, device)
    training_algorithm.train()
    args.final_epoch = training_algorithm.final_epoch
    utils.save_args(args, save_path)
    return save_path

def main(args):
    train(args)


if __name__ == '__main__':
    main(get_args())

