import matplotlib.pyplot as plt
import torch as tc
from hierarchized_bptt.hierarchized_models import HierarchizedPLRNN

@tc.no_grad()
def plot_latent_generated(model: HierarchizedPLRNN, subject_idx: int, 
                          x0: tc.Tensor, T: int, inputs: tc.Tensor, 
                          prewarm_data=None, prewarm_inputs=None, 
                          prewarm_alpha=0.125):
    ''' WITHOUT forcing '''
    X, Z = model.generate_free_trajectory(subject_idx, x0, T, inputs=inputs,
                                          prewarm_data=prewarm_data, prewarm_inputs=prewarm_inputs, prewarm_alpha=prewarm_alpha, 
                                          return_hidden=True)
    init_index = X.shape[1] - T
    # sns.set_theme()
    fig = plt.figure(figsize=(10, 10))
    # plt.axis('off')
    plot_list = [X, Z]
    names = ['observations', 'latent states']
    for i, x in enumerate(plot_list):
        fig.add_subplot(len(plot_list), 1, i + 1)
        plt.plot(x.cpu())
        if init_index > 0: 
            plt.plot((init_index, init_index), plt.ylim(), linestyle='--', color='gray')
        plt.title(names[i])
    plt.suptitle('simulated (w/o forcing)')
    plt.subplots_adjust(top=0.85)
    plt.xlabel('time steps')
    return X, Z


@tc.no_grad()
def plot_generated_against_obs(model: HierarchizedPLRNN, subject_idx: int,
                               data: tc.Tensor, inputs: tc.Tensor,
                                prewarm_data=None, prewarm_inputs=None, prewarm_alpha=0.125,
                                features=None,
                                plot_mean=False, xlim=None, ylim=None,
                                adapt_figsize=True):
    '''
    as in generate_free_trajectory: make sure data and prewarm_data are disjoint.
    '''
    X, Z = model.generate_free_trajectory(subject_idx, data[0], data.shape[0], inputs=inputs,
                                            prewarm_data=prewarm_data, prewarm_inputs=prewarm_inputs, prewarm_alpha=prewarm_alpha,
                                            return_hidden=True)
    if features is None:
        features = model.args['obs_features']
    feat_idx = [i for i in range(X.shape[1]) if model.args['obs_features'][i] in features]
    mean = data.mean(dim=0)
    n_units = len(features)
    if prewarm_data is not None:
        gt = tc.cat((prewarm_data, data), dim=0)
    else:
        gt = data
    time_steps = gt.shape[0]
    generation_start = time_steps - X.shape[0]
    if adapt_figsize:
        fig, axes = plt.subplots(n_units, 1, sharex=True, sharey=True, figsize=(8, 1+1.5*n_units), layout='constrained')
    else:   
        fig, axes = plt.subplots(n_units, 1, sharex=True, sharey=True, layout='constrained')
    # plt.axis('off')
    for i in range(n_units):
        ax = axes[i]
        gt_line, = ax.plot(tc.arange(time_steps), 
                           gt[:, feat_idx[i]].cpu(), 
                           'o', fillstyle='none', label='data')  
        gen_line, = ax.plot(tc.arange(generation_start, time_steps), 
                            X[:, feat_idx[i]].cpu(), 
                            label='generated')
        if plot_mean:
            mean_line, = ax.plot(ax.get_xlim(), [mean[feat_idx[i]]]*2, label='train set mean')
        if generation_start > 0:
            init_line, = ax.plot((generation_start, generation_start), plt.ylim(), linestyle='--', color='gray', label='prewarm limit')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(features[i])        
    plt.suptitle('simulated (w/o forcing)')
    plt.legend()
    plt.xlabel('time steps')
    # plt.tight_layout()
    return X, Z


@tc.no_grad()
def plot_prediction(model: HierarchizedPLRNN, subject_idx: int,
                    data: tc.Tensor, inputs: tc.Tensor, 
                    z0=None, alpha=None,
                    adaptive_alpha_rate=None,
                    xlim=None, ylim=None, adapt_figsize=True):
    '''
    Plot prediction of the model for a given
    input sequence with teacher forcing (interleaved
    observations)
    '''
    T_full, dx = data.size()
    n_units = dx

    # input and model prediction
    data_ = data.unsqueeze(0)
    if inputs is not None:
        inputs_ = inputs.unsqueeze(0)
    else:
        inputs_ = None
    if alpha is None:
        alpha = model.args['tf_alpha']
    # if adaptive_alpha_rate is None:
    #     adaptive_alpha_rate = model.args['adaptive_alpha_rate']
    pred = model.forward(subject_idx, data_, inputs=inputs_, tf_alpha=alpha, return_hidden=False)
    data_ = data_.squeeze(0)
    pred_ = pred.squeeze(0)
    if adapt_figsize:
        fig, axes = plt.subplots(n_units, 1, sharex=True, sharey=True, figsize=(8, 1+1.5*n_units), layout='constrained')
    else:   
        fig, axes = plt.subplots(n_units, 1, sharex=True, sharey=True, layout='constrained')
    plt.axis('off')
    for i in range(n_units):
        ax = axes[i]
        ax.plot(data_[:,i].cpu(), label='data', marker='o',
                linestyle='', fillstyle='none')
        ax.plot(pred_[:,i].cpu(), label='prediction')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(prop={"size": 5})
        ax.set_title(model.args['obs_features'][i])
    plt.suptitle('Prediction (w/ forcings)')
    plt.xlabel('t')
    # plt.tight_layout()
    return pred
