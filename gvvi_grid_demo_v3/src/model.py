"""Multi-scale residual GraphSAGE with 10-class classification head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class SAGEBranch(nn.Module):
    def __init__(self, in_dim, sage_dim1=256, sage_dim2=128, dropout=0.15):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, sage_dim1); self.norm1 = nn.LayerNorm(sage_dim1)
        self.conv2 = SAGEConv(sage_dim1, sage_dim2); self.norm2 = nn.LayerNorm(sage_dim2)
        self.dropout = dropout
    def forward(self, x, ei):
        x = F.gelu(self.norm1(self.conv1(x, ei))); x = F.dropout(x, p=self.dropout, training=self.training)
        return F.gelu(self.norm2(self.conv2(x, ei)))


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes=10, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x): return self.net(x)


class MultiScaleClassSAGE(nn.Module):
    def __init__(self, in_dim, num_classes=10, hidden_dim=256, sage_dim1=256, sage_dim2=128,
                 fusion_dim=256, dropout=0.15):
        super().__init__()
        self.encoder = TabularEncoder(in_dim, hidden_dim, dropout)
        self.local_branch = SAGEBranch(hidden_dim, sage_dim1, sage_dim2, dropout)
        self.context_branch = SAGEBranch(hidden_dim, sage_dim1, sage_dim2, dropout)
        fused_in = hidden_dim + sage_dim2 + sage_dim2
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(),
        )
        self.classifier = ClassifierHead(fusion_dim, num_classes)

    def forward(self, x, edge_index_local, edge_index_context=None):
        encoded = self.encoder(x)
        local_out = self.local_branch(encoded, edge_index_local)
        if edge_index_context is not None:
            ctx_out = self.context_branch(encoded, edge_index_context)
        else:
            ctx_out = torch.zeros_like(local_out)
        fused = self.fusion(torch.cat([encoded, local_out, ctx_out], dim=-1))
        return self.classifier(fused)
