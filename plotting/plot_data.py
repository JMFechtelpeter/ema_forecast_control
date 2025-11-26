

from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotting_utils

def data_as_heatmap(data: pd.DataFrame, observation_feat: list, input_feat: list, figsize=(10,5), binary_inputs=False, **kwargs):
    
    subplots_share = [len(observation_feat), len(input_feat)]
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'height_ratios': subplots_share}, figsize=figsize)
    
    # Plot EMA time series as heatmap
    emas = data[observation_feat].to_numpy().T
    plotting_utils.heatmap_with_colorbar(axes[0], emas, plotting_utils.discretized_colormap('jet', 7),
                                            #mpl.colors.ListedColormap(plt.cm.jet(np.linspace(1,0,7))),
                          np.linspace(1,7,7), range(1,8), vmin=0.5, vmax=7.5, **kwargs)   
    axes[0].set(yticks=range(len(observation_feat)), yticklabels=observation_feat)

    inputs = data[input_feat].to_numpy().T
    if binary_inputs:        
        plotting_utils.heatmap_with_colorbar(axes[1], inputs, plotting_utils.discretized_colormap('binary', 2),
                            np.linspace(0,1,2), np.linspace(0,1,2, dtype=int),
                            vmin=-0.5, vmax=1.5, **kwargs)
    else:
        min_in = inputs.min()
        max_in = inputs.max()
        input_steps = int(max_in - min_in)
        plotting_utils.heatmap_with_colorbar(axes[1], inputs, plotting_utils.discretized_colormap('coolwarm', input_steps),
                            np.linspace(min_in, max_in, input_steps), np.linspace(min_in, max_in, input_steps),
                            vmin=min_in, vmax=max_in, **kwargs)
    axes[1].set(yticks=range(len(input_feat)), yticklabels=input_feat)
    
    
    answer_rate = round(data[observation_feat].notna().all(axis=1).mean() * 100)
    title = f"Subject {data['Participant'].unique()}, answer rate {answer_rate}%"
    plt.suptitle(title)

    fig.tight_layout()
    return fig, axes