import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, GAT


class GraphModel(torch.nn.Module):
    def __init__(self, model_name, dataset):
        super().__init__()
        in_feats = dataset.num_node_features
        out_feats = dataset.num_classes
        hidden_channels = 16
        num_layers = 2
        if model_name == 'gcn':
            self.model = GCN(in_feats, hidden_channels, num_layers, out_channels=out_feats)
        elif model_name == 'gat':
            self.model = GAT(in_feats, hidden_channels, num_layers, out_channels=out_feats, v2=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        return F.log_softmax(x, dim=1)


def train_and_test(dataset, device):
    model_names = ['gcn', 'gat']
    num_runs = 10
    print(f"Dataset name: {dataset['data'].name}")
    for model_name in model_names:
        accs = []
        for run_idx in range(num_runs):
            model = GraphModel(model_name, dataset['data']).to(device)
            data = dataset['data'][0].to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out[dataset['train_mask']], data.y[dataset['train_mask']])
                loss.backward()
                optimizer.step()

            model.eval()
            pred = model(data).argmax(dim=1)
            correct = (pred[dataset['test_mask']] == data.y[dataset['test_mask']]).sum()
            acc = int(correct) / int(dataset['test_mask'].sum())
            accs.append(acc)

        print(f'Model: {model_name}')
        print(f'Average Accuracy: {np.mean(accs):.4f}')
        print(f'Accuracy Variance: {np.var(accs):.4f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = utils.get_datasets()
    for dataset in datasets:
        train_and_test(dataset, device)

if __name__ == '__main__':
    main()