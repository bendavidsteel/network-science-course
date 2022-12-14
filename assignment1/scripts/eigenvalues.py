import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

import utils, generate

def main():
    ab_graph = True

    if ab_graph:
        graphs = generate.generate_like_graphs()
        fig_name = 'eigenvalues_ab'
    else:
        graphs = utils.load_graphs()
        fig_name = 'eigenvalues'

    fig, axes = plt.subplots(nrows=1, ncols=len(graphs), figsize=(15,5))
    for ax, (graph_name, graph) in zip(axes, graphs.items()):
        adj = nx.to_numpy_array(graph)
        eigenvalues = scipy.sparse.linalg.eigs(adj, k=100, return_eigenvectors=False)
        spectral_gap = np.abs(eigenvalues)[-1] - np.abs(eigenvalues)[-2]
        print(f"Spectral gap of {graph_name} is: {spectral_gap}")
        ax.stem(eigenvalues.real)
        ax.set_title(graph_name[0].upper() + graph_name[1:])
        ax.set_xlabel('Rank')
        ax.set_ylabel('Eigenvalue')

    fig.tight_layout()
    utils.save_fig(fig, fig_name)

if __name__ == '__main__':
    main()