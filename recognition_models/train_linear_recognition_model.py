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
from rec_model_utils import get_PLRNN_network, matrix_distance, matrix_cosine_distance


def train_linear_deterministic_recognition_model(model_dir: str, lr: float=0.001, n_epochs: int=1000, batch_size: int=32,
                                                 l2_reg: float=0.01, penalty_on_diff_to_B_inv: float=0.0,
                                                 matrix_comparison_function=matrix_distance, 
                                                 create_plots: bool=True, save_model: bool=True, filename: str='linear_recognition_model.pt'):

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
    original_network = eval_reallabor_utils.get_network_matrix(model, data).nanmean(0)

    recognition_model = nn.Linear(model.args['dim_x'], model.args['dim_z'], bias=False)

    optimizer = tc.optim.Adam(recognition_model.parameters(), lr=lr)
    rec_loss_fn = tc.nn.MSELoss()
    rec_losses = []
    l2_reg_losses = []
    diff_to_p_inv_losses = []
    norm = []
    diff_to_pseudoinverse = []
    diff_to_original_network = []
    z0_norm = []
    z1_norm = []

    iterator = tqdm(range(n_epochs), desc='Training recognition model')
    for epoch in iterator:
            
        for batch_data, batch_inputs, batch_target in dataloader:

            optimizer.zero_grad()
            z0 = recognition_model(batch_data)
            one_step_ahead_prediction, z1 = model(batch_data.unsqueeze(1), batch_inputs.unsqueeze(1), z0, tf_alpha=0.0, initial_alpha=0.0, return_hidden=True)
            one_step_ahead_prediction = one_step_ahead_prediction.squeeze(1)
            reconstruction_loss = rec_loss_fn(one_step_ahead_prediction, batch_target)
            l2_regularization_loss = l2_reg * tc.norm(recognition_model.weight)
            diff_to_p_inv_loss = penalty_on_diff_to_B_inv * tc.norm(recognition_model.weight - B_inv)
            loss = reconstruction_loss + l2_regularization_loss + diff_to_p_inv_loss
            iterator.set_postfix(reconstruction_loss=reconstruction_loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                rec_losses.append(reconstruction_loss.item())
                l2_reg_losses.append(l2_regularization_loss.item())
                diff_to_p_inv_losses.append(diff_to_p_inv_loss.item())
                norm.append(tc.norm(recognition_model.weight).item())
                diff_to_pseudoinverse.append(matrix_comparison_function(recognition_model.weight, B_inv).item())
                new_network = get_PLRNN_network(model, recognition_model.weight, B, data).nanmean(0)
                diff_to_original_network.append(matrix_comparison_function(new_network, original_network).item())
                z0_norm.append(tc.norm(z0).item())
                z1_norm.append(tc.norm(z1).item())

    one_step_ahead_prediction = model(data[:-1][mask].unsqueeze(1), inputs[:-1][mask].unsqueeze(1)).squeeze(1)
    comparison_loss = rec_loss_fn(one_step_ahead_prediction, data[1:][mask])

    if create_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(rec_losses, color='blue', label='reconstruction loss')
        axes[0].plot(l2_reg_losses, color='red', label='l2 reg loss')
        axes[0].plot(diff_to_p_inv_losses, color='orange', label='diff to $B^+$ loss')
        axes[0].plot([comparison_loss.item()]*len(rec_losses), color='green', label='Original MSE Loss')
        axes[0].legend()
        axes[1].plot(norm, color='orange', label='rec model norm')
        axes[1].plot(diff_to_pseudoinverse, color='red', label='diff to $B^+$')
        axes[1].plot(diff_to_original_network, color='blue', label='diff to original network')
        axes[1].legend()
        axes[2].plot(z0_norm, color='green', label='$z_0$ norm')
        axes[2].plot(z1_norm, color='purple', label='$z_1$ norm')
        axes[2].legend()
        plt.savefig(data_utils.join_ordinal_bptt_path('recognition_models/temp_plots/linear_recognition_model_training.png'))

    if save_model:
        tc.save(recognition_model.state_dict(), f'{model_dir}/{filename}')


if __name__ == '__main__':

    mrt = 2

    model_folder = f'results/v3_MRT{mrt}_every_day'
    # model_dirs = [os.path.split(p)[0] for p in glob.glob(f'{model_folder}/*/*/*.pt')]
    model_dir = data_utils.join_base_path('ordinal-bptt/results/v3_MRT2_every_day/data_12600_12.csv_participant_12_date_234.0/004')
    train_linear_deterministic_recognition_model(model_dir, lr=0.01, n_epochs=1000, batch_size=256, l2_reg=0.0, penalty_on_diff_to_B_inv=0.01,
                                                 matrix_comparison_function=matrix_cosine_distance, save_model=False)