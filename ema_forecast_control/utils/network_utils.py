from typing import Optional

import torch as tc
import numpy as np
import pandas as pd
from scipy import stats, linalg
import matplotlib.pyplot as plt

from ema_forecast_control.plrnn.plrnn_model import PLRNN
from ema_forecast_control.kalman_filter.kalman_filter_model import KalmanFilter
from ema_forecast_control.simple_models.simple_models import VAR1
from ema_forecast_control.transformer.autoregressive_transformer_model import AutoregressiveTransformer
from ema_forecast_control.plotting.plot_circular_graph import plot_circular_graph

def plot_network_graph(networks: tc.Tensor, directed: bool=True, inverted_items: list=[], hide_self_connections: bool=True,
                       alpha_level: float=0.01, fisher_transform: bool=False, edge_threshold: Optional[float]=None, max_edge_number: Optional[int]=None,
                       node_labels: Optional[list]=None, title: Optional[str]=None, ax: Optional[plt.Axes]=None,
                       reveal_counterintuitive_connections: bool=True):    

    if fisher_transform:
        networks = tc.arctanh(networks)
    sig_adj = tc.tensor(stats.ttest_1samp(networks, 0, axis=0).pvalue)   # Show only connections with a weight significantly different from 0
    if fisher_transform:
        networks = tc.tanh(networks)
    networks = tc.nanmean(networks, dim=0)
    if fisher_transform:
        networks = tc.tanh(networks)
    if hide_self_connections:
        for i in range(networks.shape[0]):
            networks[i,i] = 0
    max_abs = networks.abs().max()
    if len(inverted_items) > 0:
        networks[inverted_items, :] *= -1
        networks[:, inverted_items] *= -1
        if node_labels is not None:
            node_labels = [node_labels[i] + '*' if i in inverted_items else node_labels[i] for i in range(len(node_labels)) ]
    if edge_threshold is not None:
        networks = networks * (networks.abs() > edge_threshold)
    networks = networks * (sig_adj < alpha_level)
    if max_edge_number is not None:
        flat_networks = networks.abs().flatten()
        threshold_value = tc.topk(flat_networks, k=max_edge_number).values[-1]
        networks = networks * (networks.abs() >= threshold_value)

    networks_pos = networks * (networks > 0)
    networks_neg = networks * (networks < 0)
    max_pos = networks_pos.abs().max()
    max_neg = networks_neg.abs().max()

    if reveal_counterintuitive_connections:
        if len(inverted_items) > 0:
            expected_sign = tc.ones_like(networks)
            non_inverted_items = [i for i in range(networks.shape[0]) if i not in inverted_items]
            expected_sign[tc.tensor(inverted_items)[:, None], tc.tensor(non_inverted_items)[None, :]] *= -1
            expected_sign[tc.tensor(non_inverted_items)[:, None], tc.tensor(inverted_items)[None, :]] *= -1
        else:
            expected_sign = tc.ones_like(networks)
        counterintuitive_connections = ((networks * expected_sign) < 0).nonzero()
        if len(counterintuitive_connections) > 0:
            print('Counterintuitive connections found between the following items:')
            for i in range(counterintuitive_connections.shape[0]):
                print(f'  {node_labels[counterintuitive_connections[i, 0]]} --> {node_labels[counterintuitive_connections[i, 1]]}')
        else:
            print('No counterintuitive connections found.')
        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.27, 6.27))
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    node_color = 'grey'#colors.item_color_codes(LABELS) if networks.shape[0]==len(LABELS) else None
    plot_circular_graph(networks_pos, directed=directed, labels=node_labels, ax=ax, max_edge_width=max_pos/max_abs * 3, labelpad=5,
                            edge_kwargs={'edge_color':'k'}, node_kwargs={'node_color':node_color})
    plot_circular_graph(-networks_neg, directed=directed, labels=node_labels, ax=ax, max_edge_width=max_neg/max_abs * 3, labelpad=5,
                            edge_kwargs={'edge_color':'red'}, node_kwargs={'node_color':node_color})
    ax.set(xlim=(-2,2), ylim=(-2,2))
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    return ax, sig_adj


def plot_individual_network_graph(network: tc.Tensor, directed: bool=True, inverted_items: list=[], hide_self_connections: bool=True,
                       edge_threshold: Optional[float]=None, max_edge_number: Optional[int]=None,
                       node_labels: Optional[list]=None, title: Optional[str]=None, ax: Optional[plt.Axes]=None,
                       reveal_counterintuitive_connections: bool=True):    

    if hide_self_connections:
        for i in range(network.shape[0]):
            network[i,i] = 0
    max_abs = network.abs().max()
    if len(inverted_items)>0:
        network[inverted_items, :] *= -1
        network[:, inverted_items] *= -1
        if node_labels is not None:
            node_labels = [node_labels[i] + '*' if i in inverted_items else node_labels[i] for i in range(len(node_labels)) ]
    if edge_threshold is not None:
        network = network * (network.abs() > edge_threshold)
    if max_edge_number is not None:
        flat_network = network.abs().flatten()
        threshold_value = tc.topk(flat_network, k=max_edge_number).values[-1]
        network = network * (network.abs() >= threshold_value)

    network_pos = network * (network > 0)
    network_neg = network * (network < 0)
    max_pos = network_pos.abs().max()
    max_neg = network_neg.abs().max()

    if reveal_counterintuitive_connections:
        if len(inverted_items) > 0:
            expected_sign = tc.ones_like(network)
            non_inverted_items = [i for i in range(network.shape[0]) if i not in inverted_items]
            expected_sign[tc.tensor(inverted_items)[:, None], tc.tensor(non_inverted_items)[None, :]] *= -1
            expected_sign[tc.tensor(non_inverted_items)[:, None], tc.tensor(inverted_items)[None, :]] *= -1
        else:
            expected_sign = tc.ones_like(network)
        counterintuitive_connections = ((network * expected_sign) < 0).nonzero()
        if len(counterintuitive_connections) > 0:
            print('Counterintuitive connections found between the following items:')
            for i in range(counterintuitive_connections.shape[0]):
                print(f'  {node_labels[counterintuitive_connections[i, 0]]} --> {node_labels[counterintuitive_connections[i, 1]]}')
        else:
            print('No counterintuitive connections found.')
        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.27, 6.27))
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    node_color = 'grey'#
    plot_circular_graph(network_pos, directed=directed, labels=node_labels, ax=ax, max_edge_width=max_pos/max_abs * 3, labelpad=5,
                            edge_kwargs={'edge_color':'k'}, node_kwargs={'node_color':node_color})
    plot_circular_graph(-network_neg, directed=directed, labels=node_labels, ax=ax, max_edge_width=max_neg/max_abs * 3, labelpad=5,
                            edge_kwargs={'edge_color':'red'}, node_kwargs={'node_color':node_color})
    ax.set(xlim=(-2,2), ylim=(-2,2))
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    return ax


def get_network_matrix(model: PLRNN|KalmanFilter|VAR1, x: Optional[tc.Tensor]=None, Gamma: Optional[tc.Tensor]=None, B: Optional[tc.Tensor]=None):

    B_inv = model.get_recognition_model(Gamma=Gamma, B=B)

    if isinstance(model, PLRNN):
        if x is not None:
            squeeze=False
            if x.ndim==1:
                squeeze=True
                x = x.unsqueeze(0)
            if model.args['mean_centering']:
                x = x - model.data_mean
            if B is None:
                B = model.get_observation_model()
            z = tc.einsum('lo,bo->bl', B_inv, x)
            if 0 < model.args['dim_x_proj'] < model.args['dim_z']:
                z = tc.cat((z, tc.zeros((z.shape[0], model.args['dim_z']-z.shape[1]))), dim=1)
            J = model.jacobian(z)
            if 0 < model.args['dim_x_proj'] < model.args['dim_z']:
                J = J.transpose(-2, 0).transpose(-1, 1)[:3, :3].transpose(1, -1).transpose(0, -2)
            network = tc.einsum('bok,kp->bop', tc.einsum('ol,blk->bok', B, J), B_inv).detach()
            if squeeze:
                network = network.squeeze(0)
        else:
            raise ValueError('PLRNN jacobians require z')
    elif isinstance(model, KalmanFilter):
        A = model.params['A'].to(tc.float64)
        B = model.params['B'].to(tc.float64)
        network = B @ A @ B_inv
        # network = B @ A @ tc.pinverse(B)
    elif isinstance(model, VAR1):
        network = model.params['A']
    return network


def weighted_degree_centrality(network: tc.Tensor, mode: str='out', absolute: bool=True):
    if mode=='in':
        network = network.transpose(-2, -1)
    if absolute:
        network = tc.abs(network)
    hubness = tc.sum(network, dim=-1)
    return hubness

def target_input_on_node(B: tc.Tensor, C: tc.Tensor, target_node: int, eps: float = 1e-8):
    """
    Returns u (||u||=1) maximizing alignment of Au with e_j via
    argmax_u (u^T A^T e_j e_j^T A u) / (u^T A^T A u).
    Also returns c* = e_j^T A u and Au.
    """
    A = B @ C
    G = A.T @ A
    a_j = A[target_node]
    u = tc.linalg.solve(G, a_j)
    u = u / (tc.norm(u) + eps)
    return u

def impulse_selectivity_score(B: tc.Tensor, C: tc.Tensor, u: tc.Tensor, target_node: int):
    delta_x = (B @ C @ u).abs()
    return delta_x[target_node] / tc.sum(delta_x)


def impulse_response(model: PLRNN|KalmanFilter|VAR1|AutoregressiveTransformer, u: tc.Tensor, T: int,
                     Gamma: Optional[tc.Tensor]=None, B: Optional[tc.Tensor]=None,
                     x0: Optional[tc.Tensor]=None, cumulative: bool=False, relative: bool=False) -> tc.Tensor:
    ''' IR is of shape (batch * T * dim_x), or (batch * dim_x) if cumulative. If x0 has no batch dimension, it is omitted.'''

    if x0 is None:
        x0 = tc.zeros(model.args['dim_x'])
    inputs = tc.zeros((T, model.args['dim_s']))
    inputs[0] = u
    recognition_matrix = model.get_recognition_model(Gamma=Gamma)
    ir = model.generate_free_trajectory(x0, T, inputs, 
                                        recognition_matrix=recognition_matrix,
                                        observation_matrix=B) - x0.unsqueeze(-2)
    if relative:
        ir0 = model.generate_free_trajectory(x0, T, tc.zeros((T, model.args['dim_s'])),
                                             recognition_matrix=recognition_matrix,
                                             observation_matrix=B) - x0.unsqueeze(-2)
        ir = ir - ir0
    if cumulative:
        ir = tc.sum(ir, dim=-2)
    return ir


def get_proximal_effects(data: pd.DataFrame, obs_features: list, input_features: list,
                         from_timestep: Optional[int]=None, until_timestep: Optional[int]=None,
                         sum_over_emas: bool=True, binned_policy: str='ignore'):
    
    if binned_policy == 'ignore':
        data.loc[data['Form'] == 'binned', obs_features] = np.nan
    elif binned_policy == 'drop':
        data = data.loc[data['Form'] != 'binned']
    diff = data.set_index(['Participant'], append=True)
    diff = data[obs_features].diff().shift(-1)
    diff[input_features] = data[input_features]
    diff = diff.dropna()
    diff = diff.loc[diff[input_features].sum(axis=1)>0]
    if from_timestep is not None:
        diff = diff.loc[from_timestep:]
    if until_timestep is not None:
        diff = diff.loc[:until_timestep]
    input_occurrence = diff[input_features].sum().astype(int)
    input_occurrence[input_occurrence==0] = 1
    # Get the sum of effects of each intervention on each EMA
    sum_effects = diff[obs_features].T.dot(diff[input_features])
    # Normalize them by the number of intervention presentations
    sum_effects = sum_effects.div(input_occurrence)
    if sum_over_emas:
        sum_effects = sum_effects.mean()
    return sum_effects, input_occurrence