import os 
import networkx as nx
import pickle as pkl
import sys
import numpy as np

from networkx.generators.community import LFR_benchmark_graph

def load_classic():
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data/real-classic')

    graphs = {}
    for file_name in os.listdir(data_dir_path):
        file_path = os.path.join(data_dir_path, file_name)
        # load graph
        print(file_name)
        graph = nx.read_gml(file_path)

        #graph.remove_edges_from(nx.selfloop_edges(graph))

        # remove multi edges and directed edges
        graph = nx.Graph(graph)

        name = file_name.split('.')[0]
        graphs[name] = graph

    return graphs

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_node_label():
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data/real-node-label/')

    graphs = {}

    for dataset in ["citeseer","cora","pubmed"]:

        with open(data_dir_path + "ind.{}.{}".format(dataset, 'graph'), 'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
            else:
                graph = pkl.load(f)
        
        graph = nx.from_dict_of_lists(graph)

        print(graph.number_of_nodes())
    
        graphs[dataset] = graph

        # labels

        with open(data_dir_path + "ind.{}.{}".format(dataset, 'ally'), 'rb') as f:
            if sys.version_info > (3, 0):
                ally = pkl.load(f, encoding='latin1')
            else:
                ally = pkl.load(f)

        with open(data_dir_path + "ind.{}.{}".format(dataset, 'ty'), 'rb') as f:
            if sys.version_info > (3, 0):
                ty = pkl.load(f, encoding='latin1')
            else:
                ty = pkl.load(f)

        with open(data_dir_path + "ind.{}.{}".format(dataset, 'y'), 'rb') as f:
            if sys.version_info > (3, 0):
                y = pkl.load(f, encoding='latin1')
            else:
                y = pkl.load(f)
        

        # get indices of nodes

        test_idx_reorder = parse_index_file(data_dir_path + "ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        # fix citeseer dataset,
        # as number of nodes in graph does not match with number of labels

        if dataset == 'citeseer':

            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
        
        # one-hot encoding matrix for labels

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = list(np.argmax(labels, axis=1))
        attr = {k: v for k, v in enumerate(labels)}

        # add label as node attribute

        nx.set_node_attributes(graph, attr, "labels")

    return graphs


def load_synthetic():
    n = 1000
    tau1 = 3
    tau2 = 1.5
    average_degree=5
    min_community=20

    mus = [0.1, 0.5, 0.9]

    return {f"mu_{mu}": LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=average_degree, min_community=min_community) for mu in mus}


def save_fig(fig, fig_name):
    figs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')
    fig.savefig(os.path.join(figs_path, f"{fig_name}.png"))


def save_graph_gexf(graph, name):
    data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'clustered')
    nx.write_gexf(graph, os.path.join(data_dir_path, f"{name}.gexf"))
