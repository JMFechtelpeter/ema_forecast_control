import argparse
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../..')
import logging
from typing import Union, Optional
import torch as tc
import utils
import data_utils
from comparison_models.transformer.transformer_trainer import TransformerTrainer

tc.set_num_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(description="TF RNN Training")
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--experiment', type=str, default='_debug_transformer')
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--run', type=int, default=None)    
    parser.add_argument('--overwrite', type=int, default=1)  

    # gpu
    parser.add_argument('--use_gpu', type=int, default=0)
    # cuda:0, cuda:1 etc.
    parser.add_argument('--device_id', type=int, default=0)

    # general settings
    parser.add_argument('--verbose', type=str, choices=['none','print','log'], default='log')
    parser.add_argument('--use_tb', type=int, default=0)
    parser.add_argument('--plot_trajectories_after_training', type=int, default=0)
    parser.add_argument('--plot_loss_after_training', type=int, default=False)
    parser.add_argument('--pbar_descr', type=str, default='')

    # dataset
    parser.add_argument('--data_path', default=data_utils.get_data_file(2, 24, 'processed_csv_no_con_smoothed_imputed'))
    parser.add_argument('--participant', default=None)
    parser.add_argument('--participant_id_pattern',  default='.*?_([0-9]+).csv')
    parser.add_argument('--obs_features', type=list, default=['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
    'EMA_down','EMA_sad','EMA_confidence','EMA_stress','EMA_lonely',
    'EMA_energetic','EMA_concentration','EMA_resilience','EMA_tired',
    'EMA_satisfied', 'EMA_relaxed'])#,'EMA_emotion_control','EMA_emotion_change'])
    parser.add_argument('--input_features', type=list, default=['interactive1', 'interactive2', 'interactive3', 'interactive4',
                          'interactive5', 'interactive6','interactive7', 'interactive8'])#,
                        #   'EMA_sleep','EMA_joyful_day','EMA_feelactive_sincebeep','EMA_activity_pleas',
                        #   'EMA_social_satisfied','EMA_social_alone_yes','EMA_firstsignal'])   
    parser.add_argument('--train_on_data_until_datetime', type=str, default=None)
    parser.add_argument('--train_on_data_until_timestep', type=str, default=-20)
    parser.add_argument('--train_on_last_n_steps', default=None)
    
    # resume from a model checkpoint
    parser.add_argument('--load_model_path', type=str, default=None)
    # epoch is inferred if None
    parser.add_argument('--resume_epoch', type=int, default=None)

    # model
    parser.add_argument('--dim_model', type=int, default=12)             # must be even, even if num_heads==1
    parser.add_argument('--dim_feedforward', type=int, default=4)
    parser.add_argument('--decoder_seq_len', type=int, default=6)
    parser.add_argument('--n_encoder_layers', type=int, default=1)
    parser.add_argument('--n_decoder_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int, default=500)
    # parser.add_argument('--mean_centering', type=int, default=1)
    # parser.add_argument('--boxcox', type=int, default=0)

    # training
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    parser.add_argument('--batches_per_epoch', '-bpi', type=int, default=0)
    parser.add_argument('--seq_len', '-sl', type=int, default=7)

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_annealing', '-lra', type=int, default=0)
    parser.add_argument('--n_epochs', '-n', type=int, default=1200)
    parser.add_argument('--gradient_clipping', '-gc', type=float, default=10.)
    parser.add_argument('--data_augmentation', type=int, default=0)
    parser.add_argument('--model_save_step', default='best')
    parser.add_argument('--info_save_step', type=int, default=2)
    parser.add_argument('--validation_len', '-vl', type=int, default=0)
    parser.add_argument('--validation_prewarming', type=int, default=4)
    parser.add_argument('--early_stopping', type=int, default=1)

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
    # writer = utils.init_writer(args, save_path)
    logger = configure_logging(save_path, args.verbose)
    device = utils.prepare_device(args)
    args, dataset, test_dataset = data_utils.create_dataset_reallabor(args)
    # args, dataset = utils.create_dataset(args)

    utils.save_args(args, save_path)

    training_algorithm = TransformerTrainer(args, dataset, test_dataset,
                                            None, save_path, device)
    training_algorithm.train()
    args.final_epoch = training_algorithm.final_epoch
    utils.save_args(args, save_path)
    return save_path

def main(args):
    train(args)


if __name__ == '__main__':
    main(get_args())

