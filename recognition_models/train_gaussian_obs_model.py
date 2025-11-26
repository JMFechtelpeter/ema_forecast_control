import os, sys
import torch as tc
import torch.nn as nn
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
import utils
import data_utils
import eval_reallabor_utils
from rec_model_utils import get_PLRNN_network, matrix_distance, matrix_cosine_distance, estimate_observation_covariance_from_residuals


def train_gaussian_obs_model(model_dir: str, lr: float=0.001, n_epochs: int=1000, batch_size: int=32,
                                  l2_reg_gamma: float=0.0, penalty_on_diff_to_B: float=0.0, penalty_on_diff_to_B_inv: float=0.0,
                                  matrix_comparison_function=matrix_distance, 
                                  create_plots: bool=True, save_model: bool=True, filename: str='gaussian_obs_model.pt',
                                  use_tqdm: bool=True, pbar_desc: str='Training probabilistic observation model'):

    with_args = {'data_path':data_utils.swap_base_path(utils.load_args(model_dir)['data_path'])}
    model, train_dataset, test_dataset = eval_reallabor_utils.load_model_and_data(model_dir, with_args=with_args)
    data = train_dataset.timeseries['emas'].data
    centered_data = data - model.data_mean
    inputs = train_dataset.timeseries['inputs'].data
    mask = ~data[:-1].isnan().any(1) & ~data[1:].isnan().any(1)

    dataset = tc.utils.data.TensorDataset(centered_data[:-1][mask], inputs[:-1][mask], data[1:][mask])
    dataloader = tc.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    B = model.get_parameters()['B'].detach().clone()
    B_inv = tc.linalg.pinv(B)
    empirical_Gamma = estimate_observation_covariance_from_residuals(model_dir, diagonal_cov=True)
    original_network = eval_reallabor_utils.get_network_matrix(model, data).nanmean(0)

    B_new = nn.Parameter(model.get_parameters()['B'].clone().detach(), requires_grad=True)
    log_Gamma = nn.Parameter(tc.zeros(model.args['dim_x']) * 0.01, requires_grad=True) 

    optimizer = tc.optim.Adam([B_new, log_Gamma], lr=lr)
    mse_loss_fn = tc.nn.MSELoss()
    nll_losses = []
    l2_reg_losses = []
    diff_to_B_losses = []
    diff_to_B_inv_losses = []
    mse_losses = []
    B_norm = []
    Gamma_norm = []
    diff_to_original_B = []
    diff_to_pseudoinverse = []
    diff_to_empirical_Gamma = []
    diff_to_original_network = []
    z0_norm = []
    z1_norm = []

    if use_tqdm:
        iterator = tqdm(range(n_epochs), desc=pbar_desc)
    else:
        iterator = range(n_epochs)
    for epoch in iterator:
            
        for batch_data, batch_inputs, batch_target in dataloader:

            optimizer.zero_grad()
            precision = tc.exp(-log_Gamma)
            Gamma = tc.exp(log_Gamma)
            prior_on_z0 = (B_new.T * precision @ B_new).inverse() @ B_new.T * precision
            z0 = batch_data @ prior_on_z0.T
            _, z1 = model(batch_data.unsqueeze(1), batch_inputs.unsqueeze(1), z0=z0, tf_alpha=0.0, initial_alpha=0.0, return_hidden=True)
            one_step_ahead_prediction = (z1.squeeze(1) @ B_new.T) + model.data_mean
            nll_loss = 0.5 * data.shape[0] * tc.sum(log_Gamma) + 0.5 * tc.sum(((batch_target - one_step_ahead_prediction)**2) * precision)
            l2_regularization_loss = l2_reg_gamma * tc.norm(Gamma)
            diff_to_B_loss = penalty_on_diff_to_B * tc.norm(B_new - B)
            diff_to_B_inv_loss = penalty_on_diff_to_B_inv * tc.norm(prior_on_z0 - B_inv)
            loss = l2_regularization_loss + nll_loss + diff_to_B_loss + diff_to_B_inv_loss
            mse_loss = mse_loss_fn(one_step_ahead_prediction, batch_target)

            iterator.set_postfix(nll_loss=nll_loss.item())

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            with tc.no_grad():
                nll_losses.append(nll_loss.item())
                l2_reg_losses.append(l2_regularization_loss.item())
                diff_to_B_losses.append(diff_to_B_loss.item())
                diff_to_B_inv_losses.append(diff_to_B_inv_loss.item())
                B_norm.append(tc.norm(B_new).item())
                Gamma_norm.append(tc.norm(Gamma).item())
                diff_to_original_B.append(matrix_comparison_function(B_new, B).item())
                diff_to_empirical_Gamma.append(matrix_comparison_function(tc.diag(Gamma), empirical_Gamma).item())
                diff_to_pseudoinverse.append(matrix_comparison_function(prior_on_z0, B_inv).item())
                new_network = get_PLRNN_network(model, prior_on_z0, B_new, data).nanmean(0)
                diff_to_original_network.append(matrix_comparison_function(new_network, original_network).item())
                z0_norm.append(tc.norm(z0).item())
                z1_norm.append(tc.norm(z1).item())
                mse_losses.append(mse_loss.item())


    one_step_ahead_prediction = model(data[:-1][mask].unsqueeze(1), inputs[:-1][mask].unsqueeze(1)).squeeze(1)
    comparison_loss = mse_loss_fn(one_step_ahead_prediction, data[1:][mask])

    if create_plots:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].plot(nll_losses, color='blue', label='NLL loss')
        axes[0].plot(l2_reg_losses, color='red', label='l2 reg loss')
        axes[0].plot(diff_to_B_losses, color='orange', label='diff to B loss')
        axes[0].plot(diff_to_B_inv_losses, color='purple', label='diff to $B^+$ loss')
        axes[0].legend()
        axes[1].plot([comparison_loss.item()]*len(nll_losses), color='green', label='Original MSE Loss')
        axes[1].plot(mse_losses, color='orange', label='MSE loss')
        axes[1].legend()
        axes[2].plot(B_norm, color='orange', label='B norm')
        axes[2].plot(Gamma_norm, color='purple', label='Gamma norm')
        axes[2].legend()
        axes[3].plot(diff_to_original_B, color='red', label='diff to original B')
        axes[3].plot(diff_to_empirical_Gamma, color='green', label='diff to empirical Gamma')
        axes[3].plot(diff_to_pseudoinverse, color='blue', label='diff to $B^+$')
        axes[3].plot(diff_to_original_network, color='brown', label='diff to original network')
        axes[3].legend()
        plt.savefig(data_utils.join_ordinal_bptt_path('recognition_models/temp_plots/probabilistic_obs_model_training.png'))

    if save_model:
        tc.save({'B': B_new.detach(), 'Gamma': tc.diag(tc.exp(log_Gamma)).detach()}, f'{model_dir}/{filename}')


if __name__ == '__main__':

    mrt = 2

    model_folder = f'results/v3_MRT{mrt}_every_day'
    model_dir = data_utils.join_base_path('ordinal-bptt/results/v3_MRT2_every_day/data_12600_12.csv_participant_12_date_234.0/004')
    train_gaussian_obs_model(model_dir, lr=0.01, n_epochs=1000, batch_size=256,
                                  l2_reg_gamma=0, 
                                  penalty_on_diff_to_B=200, penalty_on_diff_to_B_inv=0, 
                                  matrix_comparison_function=matrix_distance, save_model=True)