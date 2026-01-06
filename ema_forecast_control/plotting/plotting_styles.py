from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
from matplotlib import RcParams
import matplotlib as mpl
import numpy as np
import yaml
import data_utils

class PlottingContext(RcParams):
    ''' Base context manager for plotting styles '''
    
    def __enter__(self):
        self._original_rcparams = mpl.rcParams.copy()
        mpl.rcParams.update(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        mpl.rcParams.update(self._original_rcparams)

class DefaultStyle(PlottingContext):

    def __init__(self, settings: Optional[dict]=None):
        self.update(mpl.rcParams)
        if settings is not None:
            self.update(settings)

class PaperStyle(PlottingContext):

    def __init__(self, settings: Optional[dict]=None, base_style: Optional[str]='ggplot'):
        super().__init__()
        if base_style is not None:
            self.update(style.library[base_style])

        # Spines and Edges
        self['axes.spines.bottom'] = True
        self['axes.spines.left'] = True
        self['axes.spines.right'] = True
        self['axes.spines.top'] = True
        self['axes.linewidth'] = 1
        self['axes.ymargin'] = 0
        self['axes.edgecolor'] = 'black'
        # Grid
        self['axes.grid'] = True
        self['axes.grid.axis'] = 'y'
        # self['grid.color'] = 'grey'
        self['grid.linewidth'] = 0.5
        # Lines
        self['lines.linewidth'] = 1.5
        self['errorbar.capsize'] = 2.5
        self['axes.prop_cycle'] = mpl.cycler('color', ["3DB7E9", "e69f00", "359B73", "f0e442", "2271B2", "d55e00", "F748A5", "000000"])
        # General Text
        self['font.size'] = 10
        self['font.family'] = 'sans-serif'  
        self['font.sans-serif'] = 'Inter'
        self['text.color'] = 'black'
        # self['text.parse_math'] = True
        self['mathtext.fontset'] = 'cm'
        # Title
        self['figure.titlesize'] = 'medium'
        self['axes.titlesize'] = 'medium'
        # Axis Labels
        self['axes.labelcolor'] = 'black'
        self['axes.labelsize'] = 'medium'
        self['axes.formatter.use_mathtext'] = True
        self['axes.axisbelow'] = True
        # Ticks
        self['xtick.color'] = 'black'
        self['ytick.color'] = 'black'
        self['xtick.labelsize'] = 'medium'
        self['ytick.labelsize'] = 'medium'
        self['xtick.major.width'] = 1
        self['ytick.major.width'] = 1
        self['xtick.major.size'] = 4
        self['ytick.major.size'] = 4
        # Legend
        self['legend.markerscale'] = 1
        self['legend.fontsize'] = 'small'
        # Saving
        self['figure.figsize'] = (6.27, 4.5)
        self['savefig.bbox'] = 'tight'
        self['savefig.dpi'] = 300
        self['savefig.transparent'] = False
        # Imshow
        self['image.aspect'] = 'auto'
        self['image.origin'] = 'lower'

        if settings is not None:
            self.update(settings)


class colors:    
    standard = 'black'
    lightgrey = '#929591'
    misc = '#929591'

    construct_colors = {'positive affect': '#3DB7E9',
                        'negative affect': '#E69F00',
                        'self-esteem': '#359B73',
                        'activity level': '#f0e442',
                        'sleep': misc,
                        'quality of life': '#d55e00',
                        'worrying': '#F748A5',
                        'stress': '#2271B2',
                        'emotion regulation': misc,
                        'social isolation': '#000000',
                        'resilience': misc}
    manus_colors = {'positive': 'blue',
                    'negative': 'red',
                    'none': misc}
    rescaled_colors = {True: 'red',
                      False: 'blue'}
    
    gt = '#1f77b4'
    pred = '#ff7f0e'

    plrnn = '#0000FF'
    transformer = '#722F37'
    var = '#008000'
    kalman = '#800080'
    model_colors = {'clipped-shallow-PLRNN': plrnn,
                    'Transformer': transformer,
                    'VAR1': var,
                    'KalmanFilter': kalman}

    color_cycle = ["#3DB7E9", "#e69f00", "#359B73", "#f0e442", "#2271B2", "#d55e00", "#F748A5", "#000000"]

    @classmethod
    def discrete_cmap(cls, cmap: mpl.colors.Colormap, grid: np.ndarray):
        return mpl.colors.ListedColormap(cmap(grid))
    
    @classmethod
    def item_color_codes(cls, items=None, version='construct'):
        with open(data_utils.join_ordinal_bptt_path('eval_reallabor/features.yml'), 'r') as file:
            features = yaml.safe_load(file)
        color_codes = []
        if items is None:
            items = features['ema_items'].keys()
        if version=='construct':
            key = 'target_construct'
            colors = cls.construct_colors
        elif version=='manu':
            key = 'category_manu'
            colors = cls.manus_colors
        elif version=='rescaled':
            key = 'rescaled'
            colors = cls.rescaled_colors
        else:
            raise ValueError('version must be one of "construct", "manu", or "rescaled"')
        if isinstance(items, str):
            return colors[features['ema_items'][items][key]]
        else:
            for item in items:
                color_codes.append(colors[features['ema_items'][item][key]])
        return color_codes
    
    @classmethod
    def item_cosntructs(cls, items=None, version='construct'):
        with open(data_utils.join_ordinal_bptt_path('eval_reallabor/features.yml'), 'r') as file:
            features = yaml.safe_load(file)
        if items is None:
            items = features['ema_items'].keys()
        if version=='construct':
            key = 'target_construct'
            colors = cls.construct_colors
        elif version=='manu':
            key = 'category_manu'
            colors = cls.manus_colors
        elif version=='rescaled':
            key = 'rescaled'
            colors = cls.rescaled_colors
        else:
            raise ValueError('version must be one of "construct", "manu", or "rescaled"')
        if isinstance(items, str):
            return features['ema_items'][items][key]
        else:
            return [features['ema_items'][item][key] for item in items]
    
    @classmethod
    def construct_legend_handles_labels(cls, *constructs):
        # color_dict = dict(colors)
        hdls = []
        lbls = []
        for c in constructs:
            hdls.append(mpatches.Patch(color=cls.construct_colors[c]))
            lbls.append(c)
        return hdls, lbls
    

def cm2in(*cm):
    return tuple([i/2.54 for i in cm])

