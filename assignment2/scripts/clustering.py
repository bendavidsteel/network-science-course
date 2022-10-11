import collections
import json

from cdlib import algorithms, NodeClustering, viz
import networkx as nx
import networkx.algorithms.community as nx_comm
import utils
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as sk_metrics


def rename_nodes(graph):
    new_graph = nx.Graph()
    node_mapping = {}
    for idx, (node, node_data) in enumerate(graph.nodes(data=True)):
        new_graph.add_node(idx, name=node, **node_data)
        node_mapping[node] = idx

    for edge_u, edge_v in graph.edges():
        new_graph.add_edge(node_mapping[edge_u], node_mapping[edge_v])

    return new_graph

def get_topology_metrics(graph, clustering):
    return {
        "Modularity": nx_comm.modularity(graph, clustering.communities),
        "Average Conductance": clustering.conductance().score
    }

def get_label_metrics(graph, clustering, true_clustering):
    return {
        "NMI Score": clustering.normalized_mutual_information(true_clustering).score,
        "ARI Score": clustering.adjusted_rand_index(true_clustering).score
    }

def apply_communities_and_save(clustering, name):
    graph = clustering.graph
    for community_id, community in enumerate(clustering.communities):
        for node_id in community:
            graph.nodes[node_id]['comm_id'] = community_id
            bad_attrs = [attr_name for attr_name in graph.nodes[node_id] if isinstance(graph.nodes[node_id][attr_name], set)]
            for attr_name in bad_attrs:
                graph.nodes[node_id].pop(attr_name)

    utils.save_graph_gexf(graph, name)


def cluster_graph(raw_graph, graph_name, n_clusters, limit=None, labels=False):

    if not limit:
        limit = graph.number_of_nodes()

    graph = rename_nodes(raw_graph)

    if labels:
        node_attr = 'labels'
    else:
        node_attr = 'value'

    # ground truth labels for clustering
    true_communities = collections.defaultdict(list)
    for node, label in nx.get_node_attributes(graph, node_attr).items():
        true_communities[label] += [node]
    true_clusters = list(true_communities.values())
    true_clustering = NodeClustering(true_clusters, graph=graph, method_name='ground_truth')

    clustering_funcs = {
        'louvain': algorithms.louvain,
        'eigenvector': algorithms.eigenvector
    }

    # this takes a long time to run, so only gonna compare on the smaller real-classic datasets
    add_modern = True
    if add_modern:
        clustering_funcs['gemsec'] = algorithms.gemsec

    metrics = {}
    for name, clustering_func in clustering_funcs.items():
        metrics[name] = {}

        clustering = clustering_func(graph)

        metrics[name]['topology'] = get_topology_metrics(graph, clustering)
        if labels:
            metrics[name]['label'] = get_label_metrics(graph, clustering, true_clustering)

        apply_communities_and_save(clustering, f"{graph_name}_{name}")

    return metrics


def print_metrics(metrics, average_metrics):
    if average_metrics:
        avg_metrics = {
            algo: {
                type: {
                    metric: 0 for metric in metrics[list(metrics)[0]][algo][type]
                } for type in metrics[list(metrics)[0]][algo]
            } for algo in metrics[list(metrics)[0]]
        }

        for graph in metrics:
            for algo in metrics[graph]:
                for type in metrics[graph][algo]:
                    for metric in metrics[graph][algo][type]:
                        avg_metrics[algo][type][metric] += metrics[graph][algo][type][metric] / len(metrics)

        print(json.dumps(avg_metrics, indent=4))
    else:
        print(json.dumps(metrics, indent=4))
    

def main():
    average_metrics = True

    classic = True

    if classic:
        graphs = utils.load_classic()

        metrics = {}
        for graph_name, graph in graphs.items():
            print(graph_name)
            #print(graph.nodes)

            if graph_name == 'polblogs' or graph_name == 'karate':
                n_clusters = 2
            elif graph_name == 'strike' or graph_name == 'polbooks':
                n_clusters = 3
            else:
                n_clusters = 11

            metrics[graph_name] = cluster_graph(graph, graph_name, n_clusters, limit=100)

        print_metrics(metrics, average_metrics)

    # node-label
    node_label = False
    if node_label:
        graphs = utils.load_node_label()

        metrics = {}
        for graph_name, graph in graphs.items():
            print(graph_name)

            if graph_name == 'citeseer':
                n_clusters = 6
            elif graph_name == 'cora':
                n_clusters = 7
            else:
                n_clusters = 3
            
            metrics[graph_name] = cluster_graph(graph, graph_name, n_clusters, labels=True, limit=100)

        print_metrics(metrics, average_metrics)

    synthetic = False
    if synthetic:
        graphs = utils.load_synthetic()
        metrics = {}
        for graph_name, graph in graphs.items():
            print(graph_name)
            metrics[graph_name] = cluster_graph(graph, graph_name, None, limit=100)

        print_metrics(metrics, average_metrics)


if __name__ == "__main__":
    main()