import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

import utils

def main():
    graphs = utils.load_graphs()

    fig, axes = plt.subplots(nrows=1, ncols=len(graphs), figsize=(15,5))
    for ax, (graph_name, graph) in zip(axes, graphs.items()):
        
        source_degrees = []
        destination_degrees = []
        for edge in graph.edges:
            source_degrees += [graph.degree[edge[0]], graph.degree[edge[1]]]
            destination_degrees += [graph.degree[edge[1]], graph.degree[edge[0]]]

        corr, p = scipy.stats.pearsonr(source_degrees, destination_degrees)
        print(f"Pearson correlation of {graph_name} is: {corr}")

        ax.scatter(source_degrees, destination_degrees, s=2, alpha=0.2)
        ax.set_title(graph_name.title())
        ax.set_xlabel('Source Degree')
        ax.set_ylabel('Destination Degree')

    fig.tight_layout()
    utils.save_fig(fig, 'degree_correlations')

if __name__ == '__main__':
    main()