import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

import utils

def get_degree(graph):
    # get adj matrix from graph
    adj = nx.to_numpy_matrix(graph)
    # multiple adj thrice
    a3 = np.linalg.matrix_power(adj, 3)
    # get node degrees
    return np.sum(adj, axis=0)

def get_coefs(graph):
    # get adj matrix from graph
    adj = nx.to_numpy_matrix(graph)
    # multiple adj thrice
    a3 = np.linalg.matrix_power(adj, 3)
    # get node degrees
    d = get_degree(graph)
    # compute clustering coef
    return np.diag(a3) / (d * (d - 1))

def main():
    graphs = utils.load_graphs()

    fig, axes = plt.subplots(nrows=1, ncols=len(graphs), figsize=(15,5))
    for ax, (graph_name, graph) in zip(axes, graphs.items()):
        
        

        ax.plot()
        ax.set_title(graph_name.title())
        ax.set_xlabel('Source Degree')
        ax.set_ylabel('Destination Degree')

    fig.tight_layout()
    utils.save_fig(fig, 'degree_correlations')

if __name__ == '__main__':
    main()