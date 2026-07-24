"""Build kNN graph from grid centers."""
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix


def build_grid_graph(grid_df, cfg, processed_dir):
    """Build undirected kNN graph.

    Returns:
        edge_index: (2, E) LongTensor
        edge_attr: (E, 3) FloatTensor [dx, dy, dist]
        n_nodes: int
    """
    print("=" * 60)
    print("BUILD: kNN Grid Graph")
    print("=" * 60)

    k = cfg["graph"]["k"]
    n_nodes = len(grid_df)

    centers_x = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    centers_y = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    centers = np.stack([centers_x, centers_y], axis=1)

    # kNN graph (directed)
    A = kneighbors_graph(centers, n_neighbors=k, mode="connectivity", include_self=False)
    A = A + A.T  # Symmetrize
    A.data = np.ones_like(A.data)  # Binarize after addition

    coo = A.tocoo()
    src = coo.row
    dst = coo.col

    # Edge features
    dx = centers_x[dst] - centers_x[src]
    dy = centers_y[dst] - centers_y[src]
    dist = np.sqrt(dx**2 + dy**2)
    edge_attr = np.stack([dx, dy, dist], axis=1).astype(np.float32)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # Compute connected components
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(A, directed=False)
    comp_sizes = np.bincount(labels)

    print(f"Nodes: {n_nodes}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Average degree: {edge_index.shape[1] / n_nodes:.2f}")
    print(f"Connected components: {n_components}")
    print(f"Largest component: {comp_sizes.max()} nodes")
    print(f"Smallest component: {comp_sizes.min()} nodes")

    # Save
    save_path = f"{processed_dir}/grid_graph.pt"
    torch.save({"edge_index": edge_index, "edge_attr": edge_attr, "num_nodes": n_nodes}, save_path)
    print(f"Graph saved to {save_path}\n")

    return edge_index, edge_attr, n_nodes
