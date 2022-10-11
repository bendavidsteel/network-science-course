from pkgutil import get_data
import utils

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from ogb.nodeproppred import PygNodePropPredDataset

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def get_datasets():
    graphs = utils.load_node_label()

    d_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name = d_name) 

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0] # pyg graph object

def train_and_test(dataset, device):
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = get_datasets()
    for dataset in datasets:
        train_and_test(dataset, device)

if __name__ == '__main__':
    main()