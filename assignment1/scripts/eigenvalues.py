import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

import utils

def main():
    graphs = utils.load_graphs()

    fig, axes = plt.subplots(nrows=1, ncols=len(graphs))
    for ax, (graph_name, graph) in zip(axes, graphs.items()):
        adj = nx.to_numpy_matrix(graph)
        eigenvalues = scipy.sparse.linalg.eigs(adj, k=100, return_eigenvectors=False)
        ax.stem(eigenvalues.real)
        ax.set_title(graph_name.title())
        ax.set_xlabel('Rank')
        ax.set_ylabel('Eigenvalue')

    plt.show()

if __name__ == '__main__':
    main()