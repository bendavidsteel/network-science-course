from pkgutil import get_data
import utils

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class GCN(torch.nn.Module):
    def __init__(self, dataset):
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


def train_and_test(dataset, device):
    model = GCN(dataset['data']).to(device)
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
    print(f'Accuracy: {acc:.4f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = utils.get_datasets()
    for dataset in datasets:
        train_and_test(dataset, device)

if __name__ == '__main__':
    main()