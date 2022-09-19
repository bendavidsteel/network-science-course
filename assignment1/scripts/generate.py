
from random import random

import networkx as nx
import numpy as np

import utils

def generate_ab_graph(n, m):
    # start with 2 node graph with edge between them
    graph = nx.Graph()
    graph.add_edge(0, 1)

    for i in range(2, n):
        # add new node
        graph.add_node(i)

        # get existing node degrees
        nodes = np.fromiter((node for node, _ in graph.degree), int)
        degrees = np.fromiter((degree for _, degree in graph.degree), int)
        
        # normalize degrees to get probabilities
        probs = degrees / degrees.sum()
        num_new_edges = min(len(nodes)-1, m)

        # randomly pick the nodes to make edges to
        nodes_to_edge = np.random.choice(nodes, size=num_new_edges, replace=False, p=probs)
        new_edges = [(i, node_to_edge) for node_to_edge in nodes_to_edge]
        graph.add_edges_from(new_edges)

    return graph


def generate_like_graphs():
    graphs = utils.load_graphs()

    random_graphs = {}
    for graph_name, graph in graphs.items():
        n = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        m0 = 2
        m = round((num_edges - m0)/(n - m0))

        random_graph = generate_ab_graph(n, m)
        random_graphs[f"{graph_name}_AB"] = random_graph

    return random_graphs
