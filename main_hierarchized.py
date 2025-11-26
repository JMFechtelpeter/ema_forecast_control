# main.py 
import argparse
import os
import logging
from typing import Union, Optional
import torch as tc
import utils
import data_utils
from hierarchized_bptt import bptt_algorithm
from hierarchized_bptt.hierarchized_models import HierarchizedPLRNN

tc.set_num_threads(1)

MRT = 1

def get_parser():
    parser = argparse.ArgumentParser(description="TF RNN Training")
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--experiment', type=str, default='_debug_hiera')
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--run', type=int, default=1) 
    parser.add_argument('--overwrite', type=int, default=1)     

    # gpu
    parser.add_argument('--use_gpu', type=int, default=0)
    # cuda:0, cuda:1 etc.
    parser.add_argument('--device_id', type=int, default=0)

    # general settings
    parser.add_argument('--verbose', type=str, choices=['none','print','log'], default='none')
    parser.add_argument('--plot_trajectories_after_training', type=int, default=0)
    parser.add_argument('--plot_loss_after_training', type=int, default=1)
    parser.add_argument('--pbar_descr', type=str, default='')

    # dataset
    parser.add_argument('--data_path', default=data_utils.dataset_path(MRT, 'processed_csv_no_con_smoothed'))
    parser.add_argument('--participant_subset_selector', default=data_utils.train_test_split_path(MRT, 'valid_first_alarms_no_con_smoothed.csv'))
    parser.add_argument('--obs_features', type=str, nargs='+', default=['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
                                                               'EMA_down','EMA_sad','EMA_confidence','EMA_stress','EMA_lonely',
                                                               'EMA_energetic','EMA_concentration','EMA_resilience','EMA_tired',
                                                               'EMA_satisfied','EMA_relaxed'])#,'EMA_emotion_control','EMA_emotion_change'])
    parser.add_argument('--input_features', type=str, nargs='+', default=['interactive1','interactive2','interactive3','interactive4',
                                                                'interactive5','interactive6','interactive7','interactive8'])
                                                                # 'EMA_sleep','EMA_joyful_day','EMA_feelactive_sincebeep','EMA_activity_pleas',
                                                                # 'EMA_social_satisfied','EMA_social_alone_yes','EMA_firstsignal'])   
    parser.add_argument('--train_until', default='2022-08-14')#data_utils.train_test_split_path(MRT, 'valid_first_alarms_no_con_smoothed.csv'))
    parser.add_argument('--train_test_split_row', default=20)
    parser.add_argument('--train_on_last_n_steps', default=None)
    
    # resume from a model checkpoint
    parser.add_argument('--load_model_path', type=str, default=None)#data_utils.join_ordinal_bptt_path('results/hiera_MRT1_all_data/001'))
    # how to treat new subjects
    parser.add_argument('--new_subjects', type=str, choices=['ignore', 'add', 'replace'], default='replace')
    # retrain only individual parameters?
    parser.add_argument('--freeze_shared_params', type=int, default=0)
    # epoch is inferred if None
    parser.add_argument('--resume_epoch', type=int, default=None)

    # model
    parser.add_argument('--dim_p', type=int, default=8)
    parser.add_argument('--dim_z', type=int, default=6)
    parser.add_argument('--dim_y', type=int, default=15)
    parser.add_argument('--clip_range', '-clip', type=float, default=10)
    parser.add_argument('--latent_model', '-ml', type=str,
                        choices=HierarchizedPLRNN.LATENT_MODELS, default='clipped-shallow-PLRNN')
    parser.add_argument('--mean_centering', type=int, default=1)
    parser.add_argument('--dim_x_proj', type=int, default=6)    # if >0, an observation model will be learnt

    # BPTT
    parser.add_argument('--tf_alpha', '-ta', type=float, default=0.125)
    parser.add_argument('--adaptive_alpha_rate', '-aa', type=float, default=1)
    parser.add_argument('--subjects_per_batch', type=int, default=16)
    parser.add_argument('--seq_per_subject', type=int, default=8)
    parser.add_argument('--batches_per_epoch', '-bpi', type=int, default=0)
    parser.add_argument('--seq_len', '-sl', type=int, default=10)

    # training
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'RAdam'], default='RAdam')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--learning_rate_individual', type=float, default=1e-3)
    parser.add_argument('--individual_lr_bonus_per_subject', type=float, default=1e-2 / 60) 
    parser.add_argument('--lr_annealing', '-lra', type=str, choices=['ReduceLROnPlateau', 'ExponentialLR', 'LinearLR', 'None'], default='LinearLR')
    parser.add_argument('--n_epochs', '-n', type=int, default=50)
    parser.add_argument('--l2_reg', type=float, default=10e-5)
    parser.add_argument('--gradient_clipping', '-gc', type=float, default=10.)
    parser.add_argument('--data_augmentation', type=int, default=0)
    parser.add_argument('--model_save_step', default='best')
    parser.add_argument('--info_save_step', type=int, default=2)
    parser.add_argument('--validation_len', '-vl', type=int, default=6)
    parser.add_argument('--validation_prewarming', type=int, default=0) # heads up upto 4 datapoints from the training dataset before predictions in validation 
    parser.add_argument('--early_stopping', type=int, default=0)

    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()

def configure_logging(path, verbose):    
    logger = logging.getLogger('root')
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
    args, dataset, test_dataset = data_utils.create_dataset_for_hierarchized_model(args, min_valid_training_timesteps=20)

    utils.check_args(args)
    utils.save_args(args, save_path)

    training_algorithm = bptt_algorithm.HierarchizedBPTT(args, dataset, test_dataset,
                                              save_path, device)
    training_algorithm.train()
    return save_path

def main(args):
    train(args)


if __name__ == '__main__':
    main(get_args())

