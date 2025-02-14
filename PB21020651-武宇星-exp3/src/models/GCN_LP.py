import torch
import torch.nn as nn
from layers import GCNLayer

def drop_edge(adj_matrix, drop_prob):
    mask = torch.rand(adj_matrix.size()) > drop_prob
    mask = mask.to(adj_matrix.device)
    adj_matrix_dropped = adj_matrix * mask
    return adj_matrix_dropped

class GCN_LP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, layer_num, normalize, drop_prob):
        super(GCN_LP, self).__init__()
        self.drop_edge = drop_prob
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_channels, hidden_channels, normalize))
        
        for _ in range(layer_num - 2):
            self.layers.append(GCNLayer(hidden_channels, hidden_channels, normalize))
        
        self.layers.append(GCNLayer(hidden_channels, out_channels, normalize))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        num_nodes = x.size(0)
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        adj_matrix = drop_edge(adj_matrix, self.drop_edge)
        deg_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
        deg_inv_sqrt = torch.pow(deg_matrix, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_matrix_normalized = torch.mm(torch.mm(deg_inv_sqrt, adj_matrix), deg_inv_sqrt)

        for i, layer in enumerate(self.layers):
            x = layer(x, adj_matrix_normalized)
            if i < len(self.layers) - 1:  
                x = self.activation(x)
                x = self.dropout(x)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)    