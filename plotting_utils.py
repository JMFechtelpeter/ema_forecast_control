from typing import Optional
import copy
import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes
import matplotlib as mpl
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, Size, Divider
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

def discretized_colormap(base_cmap: str|mpl.colors.Colormap, segments: int, reversed: bool=False):
    if isinstance(base_cmap, str):
        base_cmap = mpl.colormaps[base_cmap]
    sampling = np.linspace(0, 1, segments)
    if reversed:
        sampling = sampling[::-1]
    return mpl.colors.ListedColormap(base_cmap(sampling))

def heatmap_with_colorbar(ax: mpl_axes.Axes, data: ArrayLike, 
                          cmap: Optional[mpl.colors.Colormap|str]=None, 
                          cbar_ticks: Optional[ArrayLike]=None, cbar_ticklabels: Optional[ArrayLike]=None, 
                          symmetrical: bool=False,
                          **kwargs):
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='5%', pad=0.5)
    if symmetrical:
        vmin = np.min(data)
        vmax = np.max(data)
        vmax = max(vmax, -vmin)
        kwargs['vmin'] = -vmax
        kwargs['vmax'] = vmax
    graph = ax.imshow(data, cmap=cmap, aspect='auto', **kwargs)
    cbar = plt.colorbar(graph, cax=cax, orientation='vertical')
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks, labels=cbar_ticklabels)
    # cbar.set_ticklabels(cmap_ticklabels)
    return ax

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

def adjust_figsize(fig: Figure, width: Optional[float]=None, height: Optional[float]=None, unit: str='in'):
    if width is not None or height is not None:
        if width is None:
            width = fig.get_figwidth()
        if height is None:
            height = fig.get_figheight()
        fig.set_size_inches(width, height, forward=True)
    return fig

def get_axis_size(ax: mpl.axes.Axes):
    sp = ax.figure.subplotpars
    fig_width, fig_height = ax.figure.get_size_inches() 
    ax_width = fig_width * (sp.right - sp.left)
    ax_height = fig_height * (sp.top - sp.bottom)
    return ax_width, ax_height

def set_axis_size(ax: mpl.axes.Axes, width: Optional[float]=None, height: Optional[float]=None):
    current_width, current_height = get_axis_size(ax)
    new_width = current_width if width is None else width
    new_height = current_height if height is None else height
    horizontal = [Size.Fixed(0), Size.Fixed(new_width)]
    vertical = [Size.Fixed(0), Size.Fixed(new_height)]
    divider = Divider(ax.get_figure(), (0, 0, 1, 1), horizontal, vertical, aspect=False)
    ax.set_axes_locator(divider.new_locator(1,1))

def adjust_lim(ax: mpl.axes.Axes, upper: Optional[float]=None, lower: Optional[float]=None, axis: str|int='x'):
    if axis=='x' or axis==1:
        lim = np.array(ax.get_xlim())
    elif axis=='y' or axis==0:
        lim = np.array(ax.get_ylim())
    else:
        raise ValueError('axis must be one of ("x"/"y") or (1/0).')
    lim_range = lim[1] - lim[0]
    if upper is not None:
        lim[1] += upper * lim_range
    if lower is not None:
        lim[0] -= lower * lim_range
    if axis in ('x', 1):
        ax.set_xlim(lim)
    else:
        ax.set_ylim(lim)

def adjust_ylim(ax: mpl.axes.Axes, bottom: Optional[float]=None, top: Optional[float]=None):
    ylim = np.array(ax.get_ylim())
    ylim_range = ylim[1] - ylim[0]
    if bottom is None and top is None:
        bottom = 0.03
        top = 0.03
    if bottom is not None:
        ylim[0] -= bottom * ylim_range
    if top is not None:
        ylim[1] += top * ylim_range
    ax.set_ylim(ylim)
    return ylim

def bars(data: np.ndarray, *args, horizontal: bool=False, ax=None, **kwargs):
    ''' Plot bar chart. Axis 1 of data defines the x axis, while axis 0 defines individual
    neighbouring bars. Args and kwargs are passed to pyplot.bar.'''

    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(data, tuple) or isinstance(data, list):
        data = np.array(data)
    N = data.shape[0]
    if data.ndim==1:
        data = data[:, np.newaxis]
    if 'width' in kwargs.keys():
        total_width = kwargs['width']
        kwargs.pop('width')
    else:
        total_width = 0.8
    special_kwarg_keys = ['yerr', 'xerr', 'color']
    special_kwarg_values = [None]*len(special_kwarg_keys)
    for i, kk in enumerate(special_kwarg_keys):
        if kk in kwargs.keys():
            special_kwarg_values[i] = kwargs[kk]
    
    
    individual_width = total_width/N
    barsize = individual_width
    offsets = -0.5*total_width + 0.5*individual_width + np.linspace(0, (N-1)*individual_width, N)
    ticks = []

    for j in range(N):
        for i, kk in enumerate(special_kwarg_keys):
            if special_kwarg_values[i] is not None:
                kwargs[kk] = special_kwarg_values[i][j]
        x = np.arange(len(data[j])) + offsets[j]
        if horizontal:
            ax.barh(x, data[j], barsize, *args, **kwargs)
        else:
            ax.bar(x, data[j], barsize, *args, **kwargs)
        ticks.append(x)

    ticks = np.hstack(ticks)
    ax.set_xticks(ticks, labels='')
    return ax


def plot_regression(x, y, ax=None, scatter_kwargs={}, line_kwargs={}, 
                    nan_policy='propagate', test_alternative='two-sided'):
    ''' Scatter plots x against y and fits affine-linear curve. 
        Returns axis and stats.PearsonResult '''
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, **scatter_kwargs)
    if nan_policy=='omit':
        mask = (~np.isnan(x)) * (~np.isnan(y))
        x, y = x[mask], y[mask]
    elif nan_policy=='propagate':
        pass
    elif nan_policy=='raise':
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError('x or y contains NaN values.')
    else:
        raise ValueError('nan_policy must be one of ("propagate", "omit", "raise").')
    
    regr = stats.linregress(x, y)
    a, b = regr.slope, regr.intercept   
    span = np.array(ax.get_xlim())
    ax.plot(span, span*a + b, **line_kwargs)
    ax.set_xlim(span)
    corr = stats.pearsonr(x, y, alternative=test_alternative)
    return ax, regr, corr