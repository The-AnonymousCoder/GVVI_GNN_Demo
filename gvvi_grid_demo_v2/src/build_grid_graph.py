"""Build multi-scale kNN graphs."""
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components


def _build_one_graph(centers, k, name):
    """Build one undirected kNN graph."""
    A = kneighbors_graph(centers, n_neighbors=k, mode="connectivity", include_self=False)
    A = A + A.T
    A.data = np.ones_like(A.data)
    coo = A.tocoo()
    src, dst = coo.row, coo.col
    dx = centers[dst, 0] - centers[src, 0]
    dy = centers[dst, 1] - centers[src, 1]
    dist = np.sqrt(dx**2 + dy**2)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(np.stack([dx, dy, dist], axis=1), dtype=torch.float32)
    n_comp, labels = connected_components(A, directed=False)
    comp_sizes = np.bincount(labels)
    print(f"  {name}: edges={edge_index.shape[1]}, avg_deg={edge_index.shape[1]/len(centers):.1f}, "
          f"components={n_comp}, max_comp={comp_sizes.max()}")
    return edge_index, edge_attr


def build_grid_graph(grid_df, cfg, processed_dir):
    """Build local (k=8) and context (k=24) graphs."""
    print("=" * 60)
    print("BUILD: Multi-Scale kNN Graphs")
    print("=" * 60)

    k_local = cfg["graph"]["k_local"]
    k_context = cfg["graph"]["k_context"]
    n_nodes = len(grid_df)

    centers_x = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    centers_y = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    centers = np.stack([centers_x, centers_y], axis=1)

    ei_local, ea_local = _build_one_graph(centers, k_local, f"Local (k={k_local})")
    ei_context, ea_context = _build_one_graph(centers, k_context, f"Context (k={k_context})")

    save_path = f"{processed_dir}/graphs.pt"
    torch.save({
        "edge_index_local": ei_local,
        "edge_index_context": ei_context,
        "num_nodes": n_nodes,
    }, save_path)
    print(f"Graphs saved to {save_path}\n")
    return ei_local, ei_context, n_nodes
