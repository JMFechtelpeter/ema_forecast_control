from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

def plot_circular_graph(weights: np.ndarray, directed: bool=False, labels: Optional[ArrayLike]=None, labelpad: int=3,
                        max_edge_width: float=3, max_edge_rad: float=0.9,
                        node_kwargs: Optional[dict]=None, edge_kwargs: Optional[dict]=None, label_kwargs: Optional[dict]=None, ax=None):
    ''' plots a graph with nodes arranged around circle, with edges that represent weights '''
    ax = ax if ax is not None else plt.gca()
    labels = labels if labels is not None else np.arange(weights.shape[0])
    node_kwargs = node_kwargs if node_kwargs is not None else {}
    edge_kwargs = edge_kwargs if edge_kwargs is not None else {}
    label_kwargs = label_kwargs if label_kwargs is not None else {}

    scaled_weights = weights / weights.max()
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
        if not (weights.T == weights).all():
            raise ValueError('weights matrix must be symmetric for undirected graph')
    size = weights.shape[0]
    node_container = list(range(size))
    graph.add_nodes_from(node_container)
    layout = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos=layout, ax=ax, **node_kwargs)

    edge_container = []
    edge_widths = []
    edge_styles = []
    for x in range(size):
        if directed:
            loopthrough = range(size)
        else:
            loopthrough = range(x)
        for y in loopthrough:
            if weights[x,y] > 0:
                edge_container.append((x,y))
                edge_widths.append(scaled_weights[x,y]*max_edge_width)
                edge_rad = -max_edge_rad*np.abs(y-x) / (size/2 + 1) + max_edge_rad
                helper_graph = graph.copy()
                width = scaled_weights[x,y]*max_edge_width
                style = f'arc3,rad={edge_rad}'
                helper_graph.add_edge(x, y)
                nx.draw_networkx_edges(helper_graph, pos=layout, ax=ax, width=width, arrows=True,
                           connectionstyle=style, **edge_kwargs)

    graph.add_edges_from(edge_container)
    # bboxes = []
    for node, pos in layout.items():
        if pos[0]==0:
            pos[0] = 1e-6
        h_align = 'left' if pos[0]>=0 else 'right'
        label = f'{" "*labelpad}{labels[node]}' if pos[0]>=0 else f'{labels[node]}{" "*labelpad}'
        angle = np.arctan(pos[1]/pos[0]) / (0.5*np.pi) * 90
        node_labels = nx.draw_networkx_labels(graph, pos={node: pos}, labels={node: label}, 
                                              horizontalalignment=h_align,
                                              ax=ax, **label_kwargs)
        node_labels[node].set_rotation_mode('anchor')
        node_labels[node].set_rotation(angle)
        # renderer = plt.gcf().canvas.get_renderer()
        # bbox = node_labels[node].get_window_extent(renderer)
        # matplotlib.patches.draw_bbox(bbox, renderer)
        # bboxes.append(bbox)
        # bbox = bbox.transformed(ax.transData.inverted())                
        # ax.update_datalim(bbox.corners())
        # ax.autoscale_view()    

    return ax