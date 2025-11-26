import argparse
import logging
import os
import torch as tc
import pandas as pd
import matplotlib.pyplot as plt
import models
import utils
import data_utils

tc.set_num_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(description="Comparison Training")
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--experiment', type=str, default='_debug_comparison')
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--run', type=int, default=None)    
    parser.add_argument('--overwrite', type=int, default=1)  

    # general settings
    parser.add_argument('--verbose', type=str, choices=['none','print','log'], default='log')
    parser.add_argument('--plot_trajectories_after_training', type=int, default=0)
    parser.add_argument('--plot_loss_after_training', type=int, default=0)
    parser.add_argument('--pbar_descr', type=str, default='')

    # dataset
    parser.add_argument('--data_path', default=data_utils.get_data_file(2, 43, 'processed_csv_no_con_smoothed'))
    parser.add_argument('--participant', default=None)
    parser.add_argument('--participant_id_pattern',  default='.*?_([0-9]+).csv')
    parser.add_argument('--obs_features', type=str, nargs='+', default=['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
                                                            'EMA_down','EMA_sad','EMA_confidence','EMA_stress','EMA_lonely',
                                                            'EMA_energetic','EMA_concentration','EMA_resilience','EMA_tired',
                                                            'EMA_satisfied', 'EMA_relaxed'])#,'EMA_emotion_control','EMA_emotion_change'])
    parser.add_argument('--input_features', type=str, nargs='+', default=['interactive1', 'interactive2', 'interactive3', 'interactive4',
                                                                'interactive5', 'interactive6','interactive7', 'interactive8'])#,
                                                                #   'EMA_sleep','EMA_joyful_day','EMA_feelactive_sincebeep','EMA_activity_pleas',
                                                                #   'EMA_social_satisfied','EMA_social_alone_yes','EMA_firstsignal'])   
    parser.add_argument('--train_on_data_until_datetime', type=str, default=None)
    parser.add_argument('--train_on_data_until_timestep', type=str, default=-100)
    parser.add_argument('--train_on_last_n_steps', default=None)
    parser.add_argument('--impute_missing_values', type=int, default=96)
    parser.add_argument('--data_dropout_to_level', default=None)
    
    # model
    parser.add_argument('--latent_model', '-ml', type=str, default='KalmanFilter')
    parser.add_argument('--intercept', type=int, default=1)
    parser.add_argument('--dim_z', type=int, default=7)
    parser.add_argument('--max_A_eigval', type=float, default=0.999)
    parser.add_argument('--mean_centering', type=int, default=1)
    parser.add_argument('--boxcox', type=int, default=0)

    # training
    parser.add_argument('--validation_len', '-vl', type=int, default=6)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--batches_per_epoch', '-bpi', type=int, default=0)
    parser.add_argument('--seq_len', '-sl', type=int, default=0)

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
    # logger = configure_logging(save_path, args.verbose)
    args, dataset, test_dataset = data_utils.create_dataset_reallabor(args)
    # args, dataset = utils.create_dataset(args)
    
    utils.save_args(args, save_path)
    args = vars(args)
    model_class = models.get_class(args['latent_model'])
    model = model_class(args)
    model.fit(dataset)
    if model.optimized:
        model.save(save_path)
        model.init_from_model_path(save_path)
        if not model.deterministic:
            val_loss = model.validate(test_dataset)
            loss_df = pd.DataFrame(data=[[model.params['loss'][-1].item(), val_loss.item()]], index=[0], columns=['epoch_loss', 'validation_loss'])
            loss_df.to_csv(os.path.join(save_path, 'loss.csv'))
        if args['plot_trajectories_after_training']:
            fig = model.plot_generated_against_obs(test_dataset, args['validation_len'])
            fig.savefig(os.path.join(save_path, 'GT_vs_Generated.png'), dpi=200)
            plt.close()
        if args['plot_loss_after_training']:
            fig = model.plot_loss()
            if fig is not None:
                fig.savefig(os.path.join(save_path, 'loss.png'), dpi=200)
                plt.close()
    else:
        print('Fail.')
    return save_path

def main(args):
    train(args)


if __name__ == '__main__':
    main(get_args())