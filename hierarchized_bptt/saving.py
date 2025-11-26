import os
from typing import Optional
from collections import OrderedDict
from argparse import Namespace
from tqdm import tqdm
import torch as tc
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
#from tensorboardX import utils as tb_utils
#from tensorboardX import SummaryWriter
import seaborn as sns
import pandas as pd
from hierarchized_bptt.hierarchized_models import HierarchizedPLRNN
from hierarchized_bptt import plot_trajectories as pltraj
from dataset.multimodal_dataset import DatasetWrapper
import utils

class Saver:
    def __init__(self, save_path: str, args: Namespace, 
                 train_wrapper: DatasetWrapper, test_wrapper: Optional[DatasetWrapper]):
        self.save_path = save_path
        # self.use_tb = args.use_tb
        self.gradient_clipping = args.gradient_clipping
        self.validation_len = args.validation_len
        self.prewarm_steps = args.validation_prewarming
        self.train_wrapper = train_wrapper
        self.test_warpper = test_wrapper
        self.current_epoch = None
        self.current_model = None
        self.loss_fn = nn.MSELoss()
        self.loss_df = pd.DataFrame()
        # self.loss_df = pd.DataFrame(columns=['epoch_loss', 'validation_loss', 'learning_rate', 
        #                                      'mean_L2_norm_A', 'sem_L2_norm_A', 'mean_L2_norm_C', 'sem_L2_norm_C'])
        self.metrics_df = []

    def update(self, model: HierarchizedPLRNN, epoch: int, 
               epoch_loss: float, validation_loss: float, l2_reg_loss: float,
               learning_rate_collective: float, learning_rate_individual: float,
               gradient_norm: float, param_change: tc.Tensor):
        self.current_epoch = epoch
        self.current_model = model
        self.epoch_loss = epoch_loss
        self.val_loss = validation_loss
        self.reg_loss = l2_reg_loss
        self.lr_coll = learning_rate_collective
        self.lr_ind = learning_rate_individual
        self.grad_norm = gradient_norm
        self.param_change = param_change

    def save_info(self, model: HierarchizedPLRNN, epoch: int, 
                  epoch_loss: float, validation_loss: float, l2_reg_loss: float,
                  learning_rate_collective: float, learning_rate_individual: float,
                  gradient_norm: float, epoch_param_change: tc.Tensor):
        self.update(model, epoch, epoch_loss, validation_loss, l2_reg_loss, learning_rate_collective, learning_rate_individual, gradient_norm, epoch_param_change)
        with tc.no_grad():           
            self.save_loss_terms()
            # if self.use_tb:
            #     self.update_tb_plots()
            # self.tb_prediction()
            # self.tb_ahead_prediction()
            # self.tb_parameter_plots()

    def save_state_dict(self, state_dict: dict, epoch: int):    
        tc.save(state_dict, os.path.join(self.save_path, f'model_{epoch}.pt'))

    def save_loss_terms(self):

        loss_df = pd.DataFrame(index=[self.current_epoch])
        if isinstance(self.epoch_loss, tc.Tensor):
            loss_df['epoch_loss'] = self.epoch_loss.item()
        else:
            loss_df['epoch_loss'] = self.epoch_loss
        if isinstance(self.val_loss, tc.Tensor):
            loss_df['validation_loss'] = self.val_loss.item()
        else:
            loss_df['validation_loss'] = self.val_loss
        if isinstance(self.reg_loss, tc.Tensor):
            loss_df['L2_reg_loss'] = self.reg_loss.item()
        else:
            loss_df['L2_reg_loss'] = self.reg_loss
        loss_df['learning_rate_collective'] = self.lr_coll
        loss_df['learning_rate_individual'] = self.lr_ind

        if self.current_model is not None:

            model_parameters = self.current_model.get_individual_params()
            L2A = tc.linalg.matrix_norm(tc.diag_embed(model_parameters['A']), ord=2, dim=(1,2))
            if model_parameters['C'] is not None:
                L2C = tc.linalg.matrix_norm(model_parameters['C'], ord=2, dim=(1,2))
            else:
                L2C = tc.tensor(0.)

            loss_df['mean_L2_norm_A'] = L2A.mean().item()
            loss_df['sem_L2_norm_A'] = L2A.std().item() / np.sqrt(len(L2A))
            loss_df['mean_L2_norm_C'] = L2C.mean().item()
            loss_df['sem_L2_norm_C'] = L2C.std().item() / np.sqrt(len(L2C))

            # total_grad_norm = utils.compute_total_grad_norm(self.current_model)
            total_grad_norm = self.grad_norm
            loss_df['total_grad_norm'] = total_grad_norm
            loss_df['param_change'] = self.param_change.mean().item()

        if len(self.loss_df)==0:
            self.loss_df = loss_df
        else:
            self.loss_df = pd.concat((self.loss_df, loss_df))
        self.loss_df.to_csv(os.path.join(self.save_path, 'loss.csv'))
        
        # if self.use_tb:
        #     self.writer.add_scalar(tag='epoch_loss', scalar_value=self.epoch_loss, global_step=self.current_epoch)
        #     self.writer.add_scalar(tag='validation_loss', scalar_value=self.val_loss, global_step=self.current_epoch)
        #     self.writer.add_scalar(tag='L2-norm_A', scalar_value=L2A, global_step=self.current_epoch)
        #     self.writer.add_scalar(tag='total_grad_norm', scalar_value=total_norm, global_step=self.current_epoch)
        
            
    # def update_tb_plots(self):
    #     figures, keys = self.parameter_plots()
    #     figures.append(self.prediction_plot())
    #     keys.append('GT vs Prediction')
    #     figures.append(self.ahead_prediction_plot())
    #     keys.append('GT vs Generated')
    #     for f, k in zip(figures, keys):
    #         image = tb_utils.figure_to_image(f)
    #         self.writer.add_image(k, image, global_step=self.current_epoch)
    #     plt.close('all')

    def save_plots(self):
        f = self.individual_parameter_plot()
        f.savefig(os.path.join(self.save_path, 'individual_parameters.pdf'), dpi=200)
        figures, keys = self.shared_parameter_plots()
        with PdfPages(os.path.join(self.save_path, 'shared_parameters.pdf')) as parameter_pages:
            for f, k in tqdm(zip(figures, keys), total=len(figures), desc='saving shared parameter plots'):
                parameter_pages.savefig(f, dpi=200)
        plt.close('all')
        with PdfPages(os.path.join(self.save_path, 'GT_vs_Prediction.pdf')) as prediction_pages:
            for i in tqdm(self.train_wrapper.dataset_indices, desc='saving prediction plots'):
                f = self.prediction_plot(i)
                prediction_pages.savefig(f, dpi=200)
                plt.close(f)
        with PdfPages(os.path.join(self.save_path, 'latent_generated.pdf')) as latent_generated_pages:
            for i in tqdm(self.train_wrapper.dataset_indices, desc='saving latent/generated plots'):
                f = self.latent_generated_plot(i)
                latent_generated_pages.savefig(f, dpi=200)
                plt.close(f)
        with PdfPages(os.path.join(self.save_path, 'GT_vs_Generated.pdf')) as ahead_prediction_pages:
            for i in tqdm(self.train_wrapper.dataset_indices, desc='saving ahead prediction plots'):
                f = self.ahead_prediction_plot(i)
                ahead_prediction_pages.savefig(f, dpi=200)
                plt.close(f)
        plt.close('all')

    def individual_parameter_plot(self):
        figure = None
        if self.current_model is not None:
            par = self.current_model.individual_parameters.detach().cpu()
            if len(par.shape) == 1:
                par = np.expand_dims(par, 1)
            figure = par_to_image(par, par_name='individual parameters', xlabel='dim_p', ylabel='subject_idx')
        return figure

    def shared_parameter_plots(self):
        plots = []
        keys = []
        if self.current_model is not None:
            par_dict = self.current_model.shared_projection_matrices
            keys = list(par_dict.keys())
            for key in par_dict.keys():
                par = par_dict[key].detach().cpu()
                if len(par.shape) == 1:
                    par = tc.unsqueeze(par, 1)
                elif len(par.shape) > 2:
                    par = tc.reshape(par, (par.shape[0], -1))
                # tranpose weight matrix of nn.Linear
                # to get true weight (Wx instead of xW)
                elif '.weight' in key:
                    par = par.T
                figure = par_to_image(par, par_name=f'{key} projection')
                plots.append(figure)
        return plots, keys


    def prediction_plot(self, subject_idx: int):
        obs, inputs = self.train_wrapper.datasets.loc[subject_idx].data()
        pltraj.plot_prediction(self.current_model, subject_idx, obs, inputs, ylim=(0.5, 7.5))
        return plt.gcf()
    
    def latent_generated_plot(self, subject_idx: int):
        obs, inputs = self.train_wrapper.datasets.loc[subject_idx].data()
        pltraj.plot_latent_generated(self.current_model, subject_idx, obs[0], 100, inputs)
        return plt.gcf()
    
    def ahead_prediction_plot(self, subject_idx: int):
        if self.test_warpper is not None:
            obs, inputs = self.test_warpper.datasets.loc[subject_idx].data(slice(0, self.validation_len+1))
            if self.prewarm_steps > 0:
                prewarm_obs, prewarm_inputs = self.test_warpper.datasets.loc[subject_idx].data(slice(-self.prewarm_steps-1, -1))
            else:
                prewarm_obs, prewarm_inputs = None, None
        else:
            obs, inputs = self.train_wrapper.datasets.loc[subject_idx].data(slice(-7, None))
            if self.prewarm_steps > 0:
                prewarm_obs, prewarm_inputs = self.train_wrapper.datasets.loc[subject_idx].data(slice(-self.prewarm_steps-7, -7))
            else:
                prewarm_obs, prewarm_inputs = None, None
        pltraj.plot_generated_against_obs(self.current_model, subject_idx, obs, inputs,
                                          prewarm_data=prewarm_obs, prewarm_inputs=prewarm_inputs, 
                                          ylim=(0.5,7.5))
        return plt.gcf()


                 
    # def tb_parameter_plots(self):
    #     '''
    #     Save all parameters as heatmap plots
    #     '''
    #     state_dict = self.current_model.state_dict()
    #     par_dict = {**dict(state_dict)}
    #     if self.use_tb:
    #         for key in par_dict.keys():
    #             par = par_dict[key].cpu()
    #             if len(par.shape) == 1:
    #                 par = np.expand_dims(par, 1)
    #             # tranpose weight matrix of nn.Linear
    #             # to get true weight (Wx instead of xW)
    #             elif '.weight' in key:
    #                 par = par.T
    #             par_to_image(par, par_name=key)
    #             image = tb_utils.figure_to_image(plt.gcf())
    #             self.writer.add_image(key, image, global_step=self.current_epoch)
    #             plt.close()

    # def tb_prediction(self):       
    #     obs, inputs = self.dataset.data()
    #     if self.use_tb:
    #         self.current_model.plot_prediction(obs, inputs, xlim=(0,300), ylim=(0.5,7.5))
    #         image = tb_utils.figure_to_image(plt.gcf())
    #         self.writer.add_image('GT_vs_Prediction', image, global_step=self.current_epoch)
    #         plt.close()
    
    # def tb_ahead_prediction(self):
    #     if self.test_dataset is not None:
    #         obs, inputs = self.test_dataset.data()
    #         prewarm_obs, prewarm_inputs = self.dataset.data()
    #     else:
    #         obs, inputs = self.dataset.data(slice(-1, None))
    #         prewarm_obs, prewarm_inputs = self.dataset.data(slice(-1))
    #     if self.use_tb:
    #         self.current_model.plot_generated_against_obs(obs, inputs,
    #                                                       prewarm_data=prewarm_obs, prewarm_inputs=prewarm_inputs,
    #                                                       prewarm_kwargs={'alpha':0.3}, xlim=(0,300), ylim=(0.5,7.5))
    #         image = tb_utils.figure_to_image(plt.gcf())
    #         self.writer.add_image('GT_vs_Generated', image, global_step=self.current_epoch)
    #         plt.close()
        
    def plot_loss(self):
        fig, axes = plt.subplots(6, 1, figsize=(10,12))
        self.loss_df[['epoch_loss', 'validation_loss', 'L2_reg_loss']].plot(ax=axes[0], title='loss')
        self.loss_df[['learning_rate_collective', 'learning_rate_individual']].plot(ax=axes[1], title='learning_rates', legend=True, logy=True)
        self.loss_df['total_grad_norm'].plot(ax=axes[2], title='total_grad_norm')
        self.loss_df['param_change'].plot(ax=axes[3], title='mean_param_change')
        # self.loss_df[['mean_L2_norm_A', 'mean_L2_norm_C']].plot(ax=axes[5], yerr=self.loss_df[['sem_L2_norm_A', 'sem_L2_norm_C']], legend=True, title='param norms')
        for i, (col, err) in enumerate(zip(['mean_L2_norm_A', 'mean_L2_norm_C'], ['sem_L2_norm_A', 'sem_L2_norm_C'])):
            self.loss_df[col].plot(ax=axes[i+4], yerr=self.loss_df[err], title=col)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'loss.pdf'), dpi=200)
        plt.close()


def data_plot(x):
    x = x.cpu().detach().numpy()
    plt.ylim(top=4, bottom=-4)
    plt.xlim(right=4, left=-4)
    plt.scatter(x[:, 0], x[:, -1], s=3)
    plt.title('{} time steps'.format(len(x)))
    return plt.gcf()

# def save_data_to_tb(data, writer, text, global_step=None):
#     if type(data) is list:
#         for i in range(len(data)):
#             plt.figure()
#             plt.title('trial {}'.format(i))
#             plt.plot(data[i])
#             image = tb_utils.figure_to_image(plt.gcf())
#             writer.add_image(text[i], image, global_step=global_step)
#     else:
#         plt.figure()
#         plt.plot(data)
#         image = tb_utils.figure_to_image(plt.gcf())
#         writer.add_image(text, image, global_step=global_step)


def par_to_image(par: tc.Tensor, par_name: str, 
                 xlabel: str|None=None, ylabel: str|None=None):
    fig = plt.figure()
    # plt.title(par_name)
    sns.set_context('paper', font_scale=1.)
    sns.set_style('white')
    max_dim = max(par.shape)
    use_annot = not (max_dim > 20)
    sns.heatmap(data=par.detach().numpy(), annot=use_annot, linewidths=float(use_annot), cmap='Blues_r', square=True, fmt='.2f')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None: 
        plt.ylabel(ylabel)
    plt.title(par_name)
    return fig