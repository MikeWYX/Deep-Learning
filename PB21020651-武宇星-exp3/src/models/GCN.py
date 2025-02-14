import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCNLayer

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, layer_num, drop_prob=0.0, normalize=True):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_channels, hidden_channels, drop_prob))        
        for _ in range(layer_num - 2):
            self.layers.append(GCNLayer(hidden_channels, hidden_channels, drop_prob))        
        self.layers.append(GCNLayer(hidden_channels, hidden_channels, drop_prob))        
        self.activation = nn.ReLU()
        self.readout = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.normalize = normalize

    def forward(self, x, adj_matrix):
        for layer in self.layers:
            x = layer(x, adj_matrix, self.normalize)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.readout(x)
        return x