import os
from argparse import Namespace
import torch as tc
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
#from tensorboardX import utils as tb_utils
#from tensorboardX import SummaryWriter
import seaborn as sns
import pandas as pd
from bptt.plrnn import PLRNN
from plotting import plot_trajectories as pltraj
import os


class Saver:
    def __init__(self, save_path: str, args: Namespace, 
                 train_dataset, test_dataset, regularizer):
        self.save_path = save_path
        self.use_tb = args.use_tb
        self.use_reg = regularizer is not None
        self.gradient_clipping = args.gradient_clipping
        self.validation_len = args.validation_len
        self.prewarm_steps = args.validation_prewarming
        self.dataset = train_dataset
        self.test_dataset = test_dataset
        self.current_epoch = None
        self.current_model = None
        self.regularizer = regularizer
        self.loss_fn = nn.MSELoss()
        self.loss_df = None
        self.metrics_df = None

    def update(self, model: PLRNN, epoch: int, epoch_loss: float, validation_loss: float, learning_rate: float):
        self.current_epoch = epoch
        self.current_model = model
        self.epoch_loss = epoch_loss
        self.val_loss = validation_loss
        self.lr = learning_rate
        self.current_model.eval()

    def save_info(self, model: PLRNN, epoch: int, epoch_loss: float, validation_loss: float, learning_rate: float):
        self.update(model, epoch, epoch_loss, validation_loss, learning_rate)
        with tc.no_grad():           
            self.save_loss_terms()
            # if self.use_tb:
            #     self.update_tb_plots()
            # self.tb_prediction()
            # self.tb_ahead_prediction()
            # self.tb_parameter_plots()

    def save_state_dict(self, state_dict: dict, epoch: int):           
        tc.save(state_dict, os.path.join(self.save_path, f'model_{epoch}.pt'))

    # def test_loss(self):
    #     if self.test_dataset is not None:
    #         obs, inputs = self.test_dataset.data()
    #         prewarm_obs, prewarm_inputs = self.dataset.data(slice(-4,None))
    #         generated, _ = self.current_model.generate_free_trajectory(obs, inputs, obs.shape[0],
    #                                                                 prewarm_data=prewarm_obs,
    #                                                                 prewarm_inputs=prewarm_inputs)
    #         test_loss = self.loss_fn(generated, obs)
    #     else:
    #         test_loss = 0.
    #     return test_loss

    def save_loss_terms(self):

        if self.current_model is not None:

            model_parameters = self.current_model.get_parameters()

            # manifold attractor regularization (MAR)
            if self.use_reg:
                loss_reg = self.regularizer.loss(model_parameters)
            else:
                loss_reg = tc.tensor(0)

            # Keep in mind: We clip the gradients from the last backward pass of the training loop at
            # current epoch here, which are already clipped during training
            # so this line has the sole purpose of getting the total_norm from the last gradients
            total_grad_norm = nn.utils.clip_grad_norm_(self.current_model.parameters(),
                                                    self.gradient_clipping)
        
            # if self.use_tb:
            #     self.writer.add_scalar(tag='epoch_loss', scalar_value=self.epoch_loss, global_step=self.current_epoch)
            #     self.writer.add_scalar(tag='validation_loss', scalar_value=self.val_loss, global_step=self.current_epoch)
            #     self.writer.add_scalar(tag='MAR_Loss', scalar_value=loss_reg, global_step=self.current_epoch)
            #     self.writer.add_scalar(tag='L2-norm_A', scalar_value=L2A, global_step=self.current_epoch)
            #     self.writer.add_scalar(tag='total_grad_norm', scalar_value=total_norm, global_step=self.current_epoch)
            
            loss_df = pd.DataFrame(index=[self.current_epoch])
            if isinstance(self.epoch_loss, tc.Tensor):
                loss_df['epoch_loss'] = self.epoch_loss.item()
            else:
                loss_df['epoch_loss'] = self.epoch_loss
            if isinstance(self.val_loss, tc.Tensor):
                loss_df['validation_loss'] = self.val_loss.item()
            else:
                loss_df['validation_loss'] = self.val_loss
            loss_df['learning_rate'] = self.lr
            # loss_df['MAR_loss'] = loss_reg.item()
            loss_df['L2_norm_A'] = tc.linalg.norm(tc.diag(model_parameters['A']), 2).item()
            if model_parameters['C'] is not None:
                loss_df['L2_norm_C'] = tc.linalg.norm(model_parameters['C'], 2).item()
            loss_df['total_grad_norm'] = total_grad_norm.item()
            if self.loss_df is None:
                self.loss_df = loss_df
            else:
                self.loss_df = pd.concat((self.loss_df, loss_df))          
            self.loss_df.to_csv(os.path.join(self.save_path, 'loss.csv'))
            
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
        figures, keys = self.parameter_plots()
        for f, k in zip(figures, keys):
            f.savefig(os.path.join(self.save_path, k+'.png'), dpi=100)
        prediction_plot = self.prediction_plot()
        prediction_plot.savefig(os.path.join(self.save_path, 'GT_vs_Prediction.png'), dpi=200)
        latent_generated_plot = self.latent_generated_plot()
        latent_generated_plot.savefig(os.path.join(self.save_path, 'generated_TS.png'), dpi=200)
        ahead_prediction_plot = self.ahead_prediction_plot()
        ahead_prediction_plot.savefig(os.path.join(self.save_path, 'GT_vs_Generated.png'), dpi=200)
        plt.close('all')
        

    def parameter_plots(self):
        '''
        Save all parameters as heatmap plots
        '''
        if self.current_model is not None:
            plots = []
            state_dict = self.current_model.state_dict()
            par_dict = {**dict(state_dict)}
            keys = list(par_dict.keys())
            for key in par_dict.keys():
                par = par_dict[key].cpu()
                if len(par.shape) == 1:
                    par = np.expand_dims(par, 1)
                # tranpose weight matrix of nn.Linear
                # to get true weight (Wx instead of xW)
                elif '.weight' in key:
                    par = par.T
                figure = par_to_image(par, par_name=key)
                plots.append(figure)
            return plots, keys
        else:
            raise ValueError('No model to plot parameters from')


    def prediction_plot(self):       
        obs, inputs = self.dataset.data()
        pltraj.plot_prediction(self.current_model, obs, inputs, ylim=(0.5, 7.5))
        return plt.gcf()
    
    def latent_generated_plot(self):
        obs, inputs = self.dataset.data()
        pltraj.plot_latent_generated(self.current_model, obs[0], 100, inputs)
        return plt.gcf()
    
    def ahead_prediction_plot(self):
        if self.test_dataset is not None:
            obs, inputs = self.test_dataset.data(slice(0, self.validation_len+1))
            prewarm_obs, prewarm_inputs = self.dataset.data(slice(-self.prewarm_steps-1, -1))
        else:
            obs, inputs = self.dataset.data(slice(-7, None))
            prewarm_obs, prewarm_inputs = self.dataset.data(slice(-self.prewarm_steps-7, -7))
        pltraj.plot_generated_against_obs(self.current_model, obs, inputs,
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
        if self.loss_df is not None:
            self.loss_df.plot(figsize=(10,12), subplots=True)
            plt.savefig(os.path.join(self.save_path, 'loss.png'), dpi=200)
            plt.close()
        else:
            raise ValueError('No loss data to plot')


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


def par_to_image(par, par_name):
    fig = plt.figure()
    # plt.title(par_name)
    sns.set_context('paper', font_scale=1.)
    sns.set_style('white')
    max_dim = max(par.shape)
    use_annot = not (max_dim > 20)
    sns.heatmap(data=par, annot=use_annot, linewidths=float(use_annot), cmap='Blues_r', square=True, fmt='.2f',
                yticklabels=False, xticklabels=False)
    plt.title(par_name)
    return fig