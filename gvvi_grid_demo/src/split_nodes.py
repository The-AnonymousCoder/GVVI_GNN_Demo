"""Stratified train/val/test split."""
import numpy as np


def split_nodes(targets, cfg, processed_dir):
    """Split nodes into train/val/test by GVVI quantile stratification."""
    print("=" * 60)
    print("SPLIT: Train / Val / Test")
    print("=" * 60)

    gvvi = targets["gvvi"]
    n = len(gvvi)
    seed = cfg["training"]["seed"]
    train_frac = cfg["training"]["train_frac"]
    val_frac = cfg["training"]["val_frac"]

    rng = np.random.default_rng(seed)

    # Stratify by decile of GVVI
    n_bins = 10
    # Use non-zero GVVI for binning, put zeros in bin 0
    bins = np.zeros(n, dtype=np.int32)
    nonzero_mask = gvvi > 0
    if nonzero_mask.sum() >= n_bins:
        bins[nonzero_mask] = pd_cut_fixed(gvvi[nonzero_mask], n_bins)
    strat_label = bins

    # Split preserving stratification
    indices = np.arange(n)
    train_idx, val_idx, test_idx = _stratified_split(
        indices, strat_label, train_frac, val_frac, rng
    )

    # Create masks
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"Total nodes: {n}")
    print(f"Train: {train_mask.sum()} ({train_mask.sum()/n:.1%})")
    print(f"Val:   {val_mask.sum()} ({val_mask.sum()/n:.1%})")
    print(f"Test:  {test_mask.sum()} ({test_mask.sum()/n:.1%})")

    # Report GVVI distribution per split
    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        g = gvvi[mask]
        nonzero = (g > 0).sum()
        print(f"  {name}: GVVI mean={g.mean():.4f}, std={g.std():.4f}, nonzero={nonzero}")

    # Save
    np.savez(
        f"{processed_dir}/split_seed42.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    print(f"Split saved to {processed_dir}/split_seed42.npz\n")

    return train_mask, val_mask, test_mask


def pd_cut_fixed(values, n_bins):
    """Like pd.cut but returns integer bin indices 0..n_bins-1."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    bin_size = n // n_bins
    bin_edges = sorted_vals[::bin_size][:n_bins]
    bin_edges = np.append(bin_edges, sorted_vals[-1] + 1e-10)
    return np.digitize(values, bin_edges[:-1]) - 1


def _stratified_split(indices, labels, train_frac, val_frac, rng):
    """Stratified split preserving label distribution."""
    unique_labels = np.unique(labels)
    train_list, val_list, test_list = [], [], []

    for lab in unique_labels:
        lab_idx = indices[labels == lab]
        rng.shuffle(lab_idx)
        n_lab = len(lab_idx)
        n_train = max(1, int(n_lab * train_frac))
        n_val = max(1, int(n_lab * val_frac))
        n_test = n_lab - n_train - n_val

        train_list.append(lab_idx[:n_train])
        val_list.append(lab_idx[n_train:n_train + n_val])
        test_list.append(lab_idx[n_train + n_val:])

    train_idx = np.concatenate(train_list)
    val_idx = np.concatenate(val_list)
    test_idx = np.concatenate(test_list)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx
