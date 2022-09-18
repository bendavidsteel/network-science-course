import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

import utils

def get_degrees(graph):
    # get adj matrix from graph
    adj = nx.to_numpy_array(graph)
    # get node degrees
    return adj.sum(axis=0, keepdims=False)

def get_coefs(graph):
    # get adj matrix from graph
    adj = nx.to_numpy_array(graph)
    # multiple adj thrice
    a3 = np.linalg.matrix_power(adj, 3)
    # get node degrees
    d = get_degrees(graph)
    # compute clustering coef
    diag_a3 = np.diag(a3)
    denom = (d * (d - 1))
    # do divide where x/0 is 0
    return np.divide(diag_a3, denom, out=np.zeros_like(diag_a3), where=denom!=0)

def main():
    graphs = utils.load_graphs()

    fig, axes = plt.subplots(nrows=1, ncols=len(graphs), figsize=(15,5))
    for ax, (graph_name, graph) in zip(axes, graphs.items()):
        
        

        ax.plot()
        ax.set_title(graph_name.title())
        ax.set_xlabel('Source Degree')
        ax.set_ylabel('Destination Degree')

    fig.tight_layout()
    utils.save_fig(fig, 'clustering_coef_dist')

if __name__ == '__main__':
    main()