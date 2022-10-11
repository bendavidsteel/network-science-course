import os 

import torch
import torch_geometric
import networkx as nx
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset

def get_datasets():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    datasets = []

    dataset_names = "Cora", "CiteSeer", "PubMed"
    for dataset_name in dataset_names:
        dataset = torch_geometric.datasets.Planetoid(data_path, dataset_name, split='random', num_train_per_class=20, num_test=1000)
        datasets.append({
            'data': dataset,
            'train_mask': dataset[0].train_mask,
            'test_mask': dataset[0].test_mask,
        })

    d_name = 'ogbn-arxiv'
    arxiv_dataset = PygNodePropPredDataset(root=data_path, name=d_name)
    arxiv_dataset.data.y = arxiv_dataset.data.y.squeeze()
    arxiv_split = arxiv_dataset.get_idx_split()

    arxiv_train_mask = torch.zeros(arxiv_dataset[0].num_nodes, dtype=torch.bool)
    arxiv_train_mask.scatter_(0, arxiv_split['train'], True)
    arxiv_test_mask = torch.zeros(arxiv_dataset[0].num_nodes, dtype=torch.bool)
    arxiv_test_mask.scatter_(0, arxiv_split['test'], True)

    datasets.append({
        'data': arxiv_dataset,
        'train_mask': arxiv_train_mask,
        'test_mask': arxiv_test_mask
    })

    return datasets



def save_fig(fig, fig_name):
    figs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')
    fig.savefig(os.path.join(figs_path, f"{fig_name}.png"))
