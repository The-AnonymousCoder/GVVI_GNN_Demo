"""Multi-scale residual GraphSAGE with zero-classification and positive-regression heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SAGEBranch(nn.Module):
    def __init__(self, in_dim, sage_dim1=256, sage_dim2=128, dropout=0.15):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, sage_dim1)
        self.norm1 = nn.LayerNorm(sage_dim1)
        self.conv2 = SAGEConv(sage_dim1, sage_dim2)
        self.norm2 = nn.LayerNorm(sage_dim2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.gelu(self.norm1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(self.norm2(self.conv2(x, edge_index)))
        return x


class DualHead(nn.Module):
    """Zero-classification + positive-regression for one GVVI component."""
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.reg_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        p_nonzero = self.cls_head(x).squeeze(-1)
        positive_value = self.reg_head(x).squeeze(-1)
        pred = p_nonzero * positive_value
        return pred, p_nonzero, positive_value


class MultiScaleResSAGE(nn.Module):
    """Multi-scale residual GraphSAGE with dual-head output."""
    def __init__(self, in_dim, hidden_dim=256, sage_dim1=256, sage_dim2=128,
                 fusion_dim=256, dropout=0.15):
        super().__init__()
        self.encoder = TabularEncoder(in_dim, hidden_dim, dropout)
        self.local_branch = SAGEBranch(hidden_dim, sage_dim1, sage_dim2, dropout)
        self.context_branch = SAGEBranch(hidden_dim, sage_dim1, sage_dim2, dropout)

        fused_in = hidden_dim + sage_dim2 + sage_dim2  # encoded + local + context
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.head_s = DualHead(fusion_dim)
        self.head_w = DualHead(fusion_dim)
        self.head_d = DualHead(fusion_dim)

    def forward(self, x, edge_index_local, edge_index_context=None):
        encoded = self.encoder(x)
        local_out = self.local_branch(encoded, edge_index_local)
        if edge_index_context is not None:
            ctx_out = self.context_branch(encoded, edge_index_context)
        else:
            ctx_out = torch.zeros_like(local_out)
        fused = self.fusion(torch.cat([encoded, local_out, ctx_out], dim=-1))

        pred_s, pnz_s, pv_s = self.head_s(fused)
        pred_w, pnz_w, pv_w = self.head_w(fused)
        pred_d, pnz_d, pv_d = self.head_d(fused)

        preds = torch.stack([pred_s, pred_w, pred_d], dim=1)
        p_nonzeros = torch.stack([pnz_s, pnz_w, pnz_d], dim=1)
        p_values = torch.stack([pv_s, pv_w, pv_d], dim=1)
        return preds, p_nonzeros, p_values
