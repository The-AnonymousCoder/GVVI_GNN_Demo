"""Two-layer GraphSAGE model for GVVI regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GVVIGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv1 = SAGEConv(hidden_dim1, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.conv2 = SAGEConv(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout = dropout
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2 // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        identity = F.linear(x, torch.eye(x.size(1), device=x.device)) if False else None
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.reg_head(x)
        return x
