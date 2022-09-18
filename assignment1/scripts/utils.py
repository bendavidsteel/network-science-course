import os

import networkx as nx

def load_graphs():
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    graphs = {}
    for file_name in os.listdir(data_dir_path):
        file_path = os.path.join(data_dir_path, file_name)
        # load graph
        graph = nx.read_edgelist(file_path)

        # remove self loops
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # remove multi edges and directed edges
        graph = nx.Graph(graph)

        name = file_name.split('.')[0]
        graphs[name] = graph

    return graphs