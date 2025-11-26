from typing import Optional
import torch as tc

from bptt.plrnn import PLRNN
import data_utils
import eval_reallabor_utils
import utils

def get_PLRNN_network(model: PLRNN, rec_matrix: tc.Tensor, obs_matrix: tc.Tensor, x: Optional[tc.Tensor]):

    if x is not None:
        squeeze=False
        if x.ndim==1:
            squeeze=True
            x = x.unsqueeze(0)
        if model.args['mean_centering']:
            x = x - model.data_mean
        z = tc.einsum('lo,bo->bl', rec_matrix, x)
        if 0 < model.args['dim_x_proj'] < model.args['dim_z']:
            z = tc.cat((z, tc.zeros((z.shape[0], model.args['dim_z']-z.shape[1]))), dim=1)
        J = model.jacobian(z)
        if 0 < model.args['dim_x_proj'] < model.args['dim_z']:
            J = J.transpose(-2, 0).transpose(-1, 1)[:3, :3].transpose(1, -1).transpose(0, -2)
        network = tc.einsum('bok,kp->bop', tc.einsum('ol,blk->bok', obs_matrix, J), rec_matrix).detach()
        if squeeze:
            network = network.squeeze(0)
    else:
        raise ValueError('PLRNN jacobians require z')
    return network

def matrix_cosine_distance(A: tc.Tensor, B: tc.Tensor):

    assert A.shape == B.shape
    assert A.ndim == 2

    inner = tc.trace(A.T @ B)
    cosine = inner / (tc.norm(A, 'fro') * tc.norm(B, 'fro'))
    distance = tc.sqrt(2 * (1 - cosine))
    return distance

def matrix_distance(A: tc.Tensor, B: tc.Tensor, ord: Optional[int] = None):

    assert A.shape == B.shape
    assert A.ndim == 2

    return tc.norm(A - B, ord)

def estimate_observation_covariance_from_residuals(model_dir: str, diagonal_cov: bool = True):

    with_args = {'data_path':data_utils.swap_base_path(utils.load_args(model_dir)['data_path'])}
    model, train_dataset, test_dataset = eval_reallabor_utils.load_model_and_data(model_dir, with_args=with_args)
    data = train_dataset.timeseries['emas'].data
    centered_data = data - model.data_mean
    inputs = train_dataset.timeseries['inputs'].data
    mask = ~data[:-1].isnan().any(1) & ~data[1:].isnan().any(1)

    B_inv = tc.linalg.pinv(model.get_parameters()['B'])
    z0 = tc.matmul(centered_data[:-1][mask], B_inv.T)
    one_step_ahead_prediction = model(data[:-1][mask].unsqueeze(1), inputs[:-1][mask].unsqueeze(1), z0, tf_alpha=0.0).squeeze(1)
    residuals = data[1:][mask] - one_step_ahead_prediction

    if diagonal_cov:
        estimated_covariance = tc.diag(tc.var(residuals, dim=0))
    else:
        estimated_covariance = tc.cov(residuals.T)

    return estimated_covariance