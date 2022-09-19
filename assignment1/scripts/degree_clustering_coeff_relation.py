import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

import clustering_coef_dist, utils, generate

def main():

    ab_graph = True
    if ab_graph:
        graphs = generate.generate_like_graphs()
        fig_name = 'degree_clustering_coeff_relation_ab'
    else:
        graphs = utils.load_graphs()
        fig_name = 'degree_clustering_coeff_relation'

    fig, axes = plt.subplots(nrows=1, ncols=len(graphs), figsize=(15,5))
    for ax, (graph_name, graph) in zip(axes, graphs.items()):
        
        degrees = clustering_coef_dist.get_degrees(graph)
        c_coeffs = clustering_coef_dist.get_coefs(graph)

        ax.scatter(degrees, c_coeffs, s=3, alpha=0.3)
        ax.set_title(graph_name[0].upper() + graph_name[1:])
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Clustering Coefficient')
        ax.set_ylim(bottom=-0.05, top=1.05)

    fig.tight_layout()
    utils.save_fig(fig, fig_name)

if __name__ == '__main__':
    main()