"""
Phase 3b: GraphSAGE Heterogeneous Graph Neural Network.

GVVI-GraphProxy: A graph proxy model for predicting multi-perspective
greenery visibility contributions.

Architecture:
- Heterogeneous GraphSAGE with 2 message-passing layers
- Observation and Greenery node encoders
- Edge decoder with dual-task output (visibility + pixel count)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ObservationEncoder(nn.Module):
    """Encode observation node features."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class GreeneryEncoder(nn.Module):
    """Encode greenery node features."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class EdgeDecoder(nn.Module):
    """
    Decode edge features + updated node embeddings into predictions.
    Dual-task: visibility probability and pixel log-count.
    """
    def __init__(self, obs_dim, grn_dim, edge_dim, hidden_dim):
        super().__init__()
        input_dim = obs_dim + grn_dim + edge_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.visibility_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.pixel_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
        )  # output log1p pixels

    def forward(self, obs_emb, grn_emb, edge_attr):
        combined = torch.cat([obs_emb, grn_emb, edge_attr], dim=-1)
        shared = self.shared(combined)
        vis_prob = self.visibility_head(shared).squeeze(-1)
        pixel_log = self.pixel_head(shared).squeeze(-1)
        return vis_prob, pixel_log


class GVVI_GraphProxy(nn.Module):
    """
    Heterogeneous GraphSAGE model for GVVI prediction.

    Node types: 'observation', 'greenery'
    Edge types: ('observation', 'candidate', 'greenery')
                ('observation', 'nearby', 'observation')
                ('greenery', 'adjacent', 'greenery')
    """
    def __init__(self, obs_feat_dim, grn_feat_dim, edge_feat_dim,
                 hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node encoders
        self.obs_encoder = ObservationEncoder(obs_feat_dim, hidden_dim)
        self.grn_encoder = GreeneryEncoder(grn_feat_dim, hidden_dim)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('observation', 'candidate', 'greenery'): SAGEConv(hidden_dim, hidden_dim),
                ('greenery', 'rev_candidate', 'observation'): SAGEConv(hidden_dim, hidden_dim),
                ('observation', 'nearby', 'observation'): SAGEConv(hidden_dim, hidden_dim),
                ('greenery', 'adjacent', 'greenery'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='mean')
            self.convs.append(conv)

        # Edge decoder
        self.edge_decoder = EdgeDecoder(hidden_dim, hidden_dim, edge_feat_dim, hidden_dim)

        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict, edge_attr):
        """
        Forward pass.

        Args:
            x_dict: {'observation': obs_feats, 'greenery': grn_feats}
            edge_index_dict: edge indices for each edge type
            edge_attr: features for candidate edges
        """
        # Encode nodes
        x_dict = {
            'observation': self.obs_encoder(x_dict['observation']),
            'greenery': self.grn_encoder(x_dict['greenery']),
        }

        # Message passing layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            if i < self.num_layers - 1:
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                          for key, x in x_dict.items()}

        # Decode edges
        obs_idx = edge_index_dict[('observation', 'candidate', 'greenery')][0]
        grn_idx = edge_index_dict[('observation', 'candidate', 'greenery')][1]

        obs_emb = x_dict['observation'][obs_idx]
        grn_emb = x_dict['greenery'][grn_idx]

        vis_prob, pixel_log = self.edge_decoder(obs_emb, grn_emb, edge_attr)

        return vis_prob, pixel_log


def build_hetero_data(obs_features, grn_features, edges_df, obs_feat_cols, grn_feat_cols,
                       edge_feat_cols, obs_scaler=None, grn_scaler=None, edge_scaler=None,
                       fit_scalers=True):
    """
    Build PyTorch Geometric HeteroData object.

    Args:
        obs_features: DataFrame of observation features
        grn_features: DataFrame of greenery features
        edges_df: DataFrame of candidate edges with features
        obs_feat_cols: Columns to use for observation features
        grn_feat_cols: Columns to use for greenery features
        edge_feat_cols: Columns to use for edge features
        obs_scaler, grn_scaler, edge_scaler: StandardScalers
        fit_scalers: Whether to fit the scalers
    """
    # Create node ID mappings
    obs_ids = obs_features.index.tolist()
    grn_ids = grn_features.index.tolist()

    obs_to_idx = {oid: i for i, oid in enumerate(obs_ids)}
    grn_to_idx = {gid: i for i, gid in enumerate(grn_ids)}

    # Prepare node features
    obs_x = obs_features[obs_feat_cols].values.astype(np.float32)
    grn_x = grn_features[grn_feat_cols].values.astype(np.float32)

    if fit_scalers:
        obs_scaler = StandardScaler()
        grn_scaler = StandardScaler()
        edge_scaler = StandardScaler()
        obs_x = obs_scaler.fit_transform(obs_x)
        grn_x = grn_scaler.fit_transform(grn_x)
    else:
        obs_x = obs_scaler.transform(obs_x)
        grn_x = grn_scaler.transform(grn_x)

    obs_x = torch.tensor(obs_x, dtype=torch.float32)
    grn_x = torch.tensor(grn_x, dtype=torch.float32)

    # Build candidate edges (observation -> greenery)
    edges_valid = edges_df[
        edges_df['view_node_id'].isin(obs_to_idx) &
        edges_df['greenery_id'].isin(grn_to_idx)
    ].copy()

    edge_obs_idx = edges_valid['view_node_id'].map(obs_to_idx).values
    edge_grn_idx = edges_valid['greenery_id'].map(grn_to_idx).values

    candidate_edge_index = torch.tensor(
        [edge_obs_idx, edge_grn_idx], dtype=torch.long)

    # Edge features
    edge_feats_available = [c for c in edge_feat_cols if c in edges_valid.columns]
    edge_x = edges_valid[edge_feats_available].values.astype(np.float32)
    if fit_scalers:
        edge_x = edge_scaler.fit_transform(edge_x)
    else:
        edge_x = edge_scaler.transform(edge_x)

    edge_attr = torch.tensor(edge_x, dtype=torch.float32)

    # Build observation-observation edges (spatial kNN)
    obs_coords = obs_features[['x', 'y']].values
    from scipy.spatial import KDTree
    tree = KDTree(obs_coords)
    k = min(5, len(obs_coords) - 1)
    _, obs_nn_idx = tree.query(obs_coords, k=k+1)
    obs_src, obs_dst = [], []
    for i in range(len(obs_coords)):
        for j in obs_nn_idx[i, 1:]:  # skip self
            obs_src.append(i)
            obs_dst.append(j)
    obs_nearby_index = torch.tensor([obs_src, obs_dst], dtype=torch.long)

    # Build greenery-greenery edges (spatial adjacency)
    grn_coords = grn_features[['center_x', 'center_y']].values
    tree = KDTree(grn_coords)
    k = min(5, len(grn_coords) - 1)
    _, grn_nn_idx = tree.query(grn_coords, k=k+1)
    grn_src, grn_dst = [], []
    for i in range(len(grn_coords)):
        for j in grn_nn_idx[i, 1:]:
            grn_src.append(i)
            grn_dst.append(j)
    grn_adj_index = torch.tensor([grn_src, grn_dst], dtype=torch.long)

    # Assemble HeteroData
    data = HeteroData()

    data['observation'].x = obs_x
    data['greenery'].x = grn_x

    data['observation', 'candidate', 'greenery'].edge_index = candidate_edge_index
    data['greenery', 'rev_candidate', 'observation'].edge_index = \
        torch.stack([candidate_edge_index[1], candidate_edge_index[0]])

    data['observation', 'nearby', 'observation'].edge_index = obs_nearby_index
    data['greenery', 'adjacent', 'greenery'].edge_index = grn_adj_index

    # Store edge features and labels
    data['observation', 'candidate', 'greenery'].edge_attr = edge_attr
    data['observation', 'candidate', 'greenery'].y_visible = \
        torch.tensor(edges_valid['is_visible'].values, dtype=torch.float32)
    data['observation', 'candidate', 'greenery'].y_pixel = \
        torch.tensor(edges_valid['pixel_count'].values, dtype=torch.float32)

    # Store masks
    data['observation', 'candidate', 'greenery'].train_mask = \
        torch.tensor((edges_valid['split'] == 'train').values, dtype=torch.bool)
    data['observation', 'candidate', 'greenery'].val_mask = \
        torch.tensor((edges_valid['split'] == 'val').values, dtype=torch.bool)
    data['observation', 'candidate', 'greenery'].test_mask = \
        torch.tensor((edges_valid['split'] == 'test').values, dtype=torch.bool)

    # Store ID mappings
    data.obs_to_idx = obs_to_idx
    data.grn_to_idx = grn_to_idx
    data.obs_ids = obs_ids
    data.grn_ids = grn_ids

    # Store scalers
    data.obs_scaler = obs_scaler
    data.grn_scaler = grn_scaler
    data.edge_scaler = edge_scaler

    # Store edge DataFrame for later evaluation
    data.valid_edges = edges_valid

    return data


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_graphsage(model, data, epochs=200, lr=0.001, weight_decay=1e-4,
                    patience=30, device='cpu'):
    """
    Train the GraphSAGE model.
    """
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    visibility_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    pixel_loss_fn = nn.HuberLoss(delta=1.0)

    x_dict = {'observation': data['observation'].x, 'greenery': data['greenery'].x}
    edge_index_dict = {
        ('observation', 'candidate', 'greenery'): data['observation', 'candidate', 'greenery'].edge_index,
        ('greenery', 'rev_candidate', 'observation'): data['greenery', 'rev_candidate', 'observation'].edge_index,
        ('observation', 'nearby', 'observation'): data['observation', 'nearby', 'observation'].edge_index,
        ('greenery', 'adjacent', 'greenery'): data['greenery', 'adjacent', 'greenery'].edge_index,
    }
    edge_attr = data['observation', 'candidate', 'greenery'].edge_attr

    y_visible_all = data['observation', 'candidate', 'greenery'].y_visible
    y_pixel_all = data['observation', 'candidate', 'greenery'].y_pixel
    train_mask = data['observation', 'candidate', 'greenery'].train_mask
    val_mask = data['observation', 'candidate', 'greenery'].val_mask

    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        vis_prob, pixel_log = model(x_dict, edge_index_dict, edge_attr)

        # Compute losses on training edges
        vis_loss = visibility_loss_fn(vis_prob[train_mask], y_visible_all[train_mask])

        # Pixel loss only on visible edges
        vis_train_mask = train_mask & (y_visible_all > 0)
        if vis_train_mask.sum() > 0:
            pix_loss = pixel_loss_fn(
                pixel_log[vis_train_mask],
                torch.log1p(y_pixel_all[vis_train_mask])
            )
        else:
            pix_loss = torch.tensor(0.0, device=device)

        loss = vis_loss + 0.5 * pix_loss

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            vis_prob_val, pixel_log_val = model(x_dict, edge_index_dict, edge_attr)

            val_vis_loss = visibility_loss_fn(vis_prob_val[val_mask], y_visible_all[val_mask])

            vis_val_mask = val_mask & (y_visible_all > 0)
            if vis_val_mask.sum() > 0:
                val_pix_loss = pixel_loss_fn(
                    pixel_log_val[vis_val_mask],
                    torch.log1p(y_pixel_all[vis_val_mask])
                )
            else:
                val_pix_loss = torch.tensor(0.0, device=device)

            val_loss = val_vis_loss + 0.5 * val_pix_loss

        scheduler.step(val_loss)

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: train_loss={loss.item():.4f}, "
                  f"val_loss={val_loss.item():.4f}")

    # Restore best model
    model.load_state_dict(best_state_dict)
    model.eval()

    return model, history


def predict_graphsage(model, data, device='cpu'):
    """Generate predictions for all candidate edges."""
    model.eval()

    x_dict = {'observation': data['observation'].x.to(device),
              'greenery': data['greenery'].x.to(device)}
    edge_index_dict = {
        ('observation', 'candidate', 'greenery'): data['observation', 'candidate', 'greenery'].edge_index.to(device),
        ('greenery', 'rev_candidate', 'observation'): data['greenery', 'rev_candidate', 'observation'].edge_index.to(device),
        ('observation', 'nearby', 'observation'): data['observation', 'nearby', 'observation'].edge_index.to(device),
        ('greenery', 'adjacent', 'greenery'): data['greenery', 'adjacent', 'greenery'].edge_index.to(device),
    }
    edge_attr = data['observation', 'candidate', 'greenery'].edge_attr.to(device)

    with torch.no_grad():
        vis_prob, pixel_log = model(x_dict, edge_index_dict, edge_attr)

    return vis_prob.cpu().numpy(), np.expm1(pixel_log.cpu().numpy()).clip(0, None)


def run_graphsage_experiment(obs_features, grn_features, edges_df,
                              obs_feat_cols, grn_feat_cols, edge_feat_cols,
                              hidden_dim=64, num_layers=2, dropout=0.3,
                              epochs=200, lr=0.001, device='cpu'):
    """
    Run full GraphSAGE experiment.
    """
    print(f"\n{'='*60}")
    print("Building Heterogeneous Graph...")

    data = build_hetero_data(
        obs_features, grn_features, edges_df,
        obs_feat_cols, grn_feat_cols, edge_feat_cols,
        fit_scalers=True
    )

    obs_feat_dim = data['observation'].x.shape[1]
    grn_feat_dim = data['greenery'].x.shape[1]
    edge_feat_dim = data['observation', 'candidate', 'greenery'].edge_attr.shape[1]

    print(f"  Observation features: {obs_feat_dim}")
    print(f"  Greenery features: {grn_feat_dim}")
    print(f"  Edge features: {edge_feat_dim}")
    print(f"  Candidate edges: {data['observation', 'candidate', 'greenery'].edge_index.shape[1]}")
    print(f"  Observation nearby edges: {data['observation', 'nearby', 'observation'].edge_index.shape[1]}")
    print(f"  Greenery adjacent edges: {data['greenery', 'adjacent', 'greenery'].edge_index.shape[1]}")

    # Create model
    model = GVVI_GraphProxy(
        obs_feat_dim=obs_feat_dim,
        grn_feat_dim=grn_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Train
    t0 = time.time()
    model, history = train_graphsage(model, data, epochs=epochs, lr=lr,
                                      patience=30, device=device)
    train_time = time.time() - t0

    # Predict
    t1 = time.time()
    vis_proba, pixel_pred = predict_graphsage(model, data, device=device)
    infer_time = time.time() - t1

    # Evaluate on test set
    test_mask = data['observation', 'candidate', 'greenery'].test_mask.numpy()
    edges_valid = data.valid_edges

    from .baselines import evaluate_visibility, evaluate_pixels

    y_vis_test = edges_valid.loc[test_mask, 'is_visible'].values
    vis_metrics = evaluate_visibility(
        edges_valid[test_mask], y_vis_test, vis_proba[test_mask], prefix='vis_test_')

    y_pix_test = edges_valid.loc[test_mask, 'pixel_count'].values
    all_pix_metrics = evaluate_pixels(
        edges_valid[test_mask], y_pix_test, pixel_pred[test_mask], prefix='pix_all_')

    vis_true_mask = y_vis_test == 1
    vis_pix_metrics = evaluate_pixels(
        edges_valid[test_mask], y_pix_test, pixel_pred[test_mask],
        mask=vis_true_mask, prefix='pix_visible_')

    results = {
        'model': 'GraphSAGE',
        'n_params': n_params,
        'train_time_s': train_time,
        'infer_time_s': infer_time,
        'final_epoch': len(history['train_loss']),
        'best_train_loss': min(history['train_loss']),
        'best_val_loss': min(history['val_loss']),
        **vis_metrics,
        **all_pix_metrics,
        **vis_pix_metrics,
    }

    print(f"\n  GraphSAGE Results:")
    print(f"    AUPRC: {results['vis_test_auprc']:.4f}")
    print(f"    AUROC: {results['vis_test_auroc']:.4f}")
    print(f"    F1: {results['vis_test_f1']:.4f}")
    print(f"    Pixel MAE (all): {results['pix_all_mae']:.2f}")
    print(f"    Train time: {results['train_time_s']:.2f}s")

    return model, data, results, vis_proba, pixel_pred


if __name__ == '__main__':
    print("GraphSAGE model module loaded. Use run_pipeline.py to execute experiments.")
