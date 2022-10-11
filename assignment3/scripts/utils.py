import os 

import torch_geometric
import networkx as nx
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset

def get_datasets():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    datasets = []

    dataset_names = "Cora", "CiteSeer", "PubMed"
    datasets = [torch_geometric.datasets.Planetoid(data_path, dataset_name) for dataset_name in dataset_names]

    d_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name = d_name)
    datasets.append(dataset)

    return datasets


def save_fig(fig, fig_name):
    figs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')
    fig.savefig(os.path.join(figs_path, f"{fig_name}.png"))
