import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

import clustering_coef_dist, utils

def main():
    graphs = utils.load_graphs()

    fig, axes = plt.subplots(nrows=1, ncols=len(graphs), figsize=(15,5))
    for ax, (graph_name, graph) in zip(axes, graphs.items()):
        
        degrees = clustering_coef_dist.get_degrees(graph)
        c_coeffs = clustering_coef_dist.get_coefs(graph)

        ax.scatter(degrees, c_coeffs, s=2, alpha=0.2)
        ax.set_title(graph_name.title())
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Clustering Coefficient')

    fig.tight_layout()
    utils.save_fig(fig, 'degree_clustering_coeff_relation')

if __name__ == '__main__':
    main()