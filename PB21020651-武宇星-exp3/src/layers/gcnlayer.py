import torch
import torch.nn as nn
import torch.nn.functional as F

def pair_norm(x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + 1e-6)

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, normalize):
        super(GCNLayer, self).__init__()
        self.projection = nn.Linear(in_channels, out_channels)
        self.normalize = normalize

    def forward(self, node_feats, adj_matrix):
        adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        deg_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
        deg_inv_sqrt = torch.pow(deg_matrix, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_matrix_normalized = torch.mm(torch.mm(deg_inv_sqrt, adj_matrix), deg_inv_sqrt)
        node_feats = self.projection(node_feats)
        if self.normalize:
            node_feats = pair_norm(node_feats)
        node_feats = torch.mm(adj_matrix_normalized, node_feats)        
        return node_feats     