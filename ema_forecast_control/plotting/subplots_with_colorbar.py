from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def subplots_with_colorbar(*data: ArrayLike, nrows: int=1, figsize: tuple=(6.4,4.8), axes_pad: int|tuple=0.3, share_all: bool=False,
                           cbar_location: str='right', cbar_size: str='5%', cbar_pad: float=0.3,
                           cbar_ticks: Optional[ArrayLike]=None, cbar_ticklabels: Optional[ArrayLike]=None, 
                           image_grid_aspect: bool=True,
                           **kwargs):
    ncols = int(np.ceil(len(data) / nrows))
    fig = plt.figure(figsize=figsize)
    axes = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=axes_pad, share_all=share_all, label_mode='all',
                     cbar_mode='single', cbar_location=cbar_location, cbar_pad=cbar_pad,
                     cbar_size=cbar_size, aspect=image_grid_aspect)
    for i, ax in enumerate(axes):
        im = ax.imshow(data[i], **kwargs)
    cbar = ax.cax.colorbar(im)
    if cbar_ticks is not None:
        if cbar_ticklabels is None:
            cbar_ticklabels = cbar_ticks
        cbar.set_ticks(cbar_ticks, labels=cbar_ticklabels)
    return fig, axes