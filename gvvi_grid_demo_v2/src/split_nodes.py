"""80/5/15 split preserving original test nodes."""
import numpy as np


def split_nodes(targets, cfg, processed_dir, v1_split_path):
    """Create 80% train / 5% val / 15% test split.

    Original test nodes are preserved from V1 split.
    Train+Val from V1 are reallocated to 80% train, 5% val.
    """
    print("=" * 60)
    print("SPLIT: Train 80% / Val 5% / Test 15% (preserved)")
    print("=" * 60)

    # Load V1 split
    v1 = np.load(v1_split_path)
    v1_test_idx = v1["test_idx"]
    v1_train_idx = v1["train_idx"]
    v1_val_idx = v1["val_idx"]

    print(f"V1 test nodes: {len(v1_test_idx)} (permanently frozen)")
    print(f"V1 train+val nodes: {len(v1_train_idx) + len(v1_val_idx)}")

    gvvi = targets["gvvi"]
    n = len(gvvi)
    seed = cfg["training"]["seed"]
    rng = np.random.default_rng(seed)

    # Combine old train+val as the pool for new train/val
    pool = np.concatenate([v1_train_idx, v1_val_idx])
    pool_gvvi = gvvi[pool]

    # Stratify by GVVI decile
    n_bins = 10
    bins = np.zeros(len(pool), dtype=np.int32)
    nonzero_mask = pool_gvvi > 0
    if nonzero_mask.sum() >= n_bins:
        bins[nonzero_mask] = _pd_cut_fixed(pool_gvvi[nonzero_mask], n_bins)

    # Split pool into 80/(80+5) ≈ 94.1% train, 5/(80+5) ≈ 5.9% val
    # Actually: total is 100%. We want 80% train from 100%.
    # Pool is 85% of nodes. 80/85 of pool → train, 5/85 of pool → val.
    pool_train_frac = 80.0 / 85.0
    pool_val_frac = 5.0 / 85.0

    train_list, val_list = [], []
    for lab in np.unique(bins):
        lab_idx = pool[bins == lab]
        rng.shuffle(lab_idx)
        n_lab = len(lab_idx)
        n_train = max(1, int(n_lab * pool_train_frac))
        n_val = max(1, n_lab - n_train)
        train_list.append(lab_idx[:n_train])
        val_list.append(lab_idx[n_train:n_train + n_val])

    train_idx = np.concatenate(train_list)
    val_idx = np.concatenate(val_list)
    test_idx = v1_test_idx.copy()
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    val_mask = np.zeros(n, dtype=bool); val_mask[val_idx] = True
    test_mask = np.zeros(n, dtype=bool); test_mask[test_idx] = True

    print(f"Total nodes: {n}")
    print(f"Train: {train_mask.sum()} ({train_mask.sum()/n:.1%})")
    print(f"Val:   {val_mask.sum()} ({val_mask.sum()/n:.1%})")
    print(f"Test:  {test_mask.sum()} ({test_mask.sum()/n:.1%})")
    assert set(test_idx) == set(v1_test_idx), "Test set changed!"
    assert len(set(train_idx) & set(test_idx)) == 0

    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        g = gvvi[mask]
        print(f"  {name}: GVVI mean={g.mean():.4f}, std={g.std():.4f}, nonzero={(g>0).sum()}")

    np.savez(f"{processed_dir}/split_train80_val5_test15_seed42.npz",
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
             train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    print(f"Split saved.\n")
    return train_mask, val_mask, test_mask


def _pd_cut_fixed(values, n_bins):
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    bin_size = n // n_bins
    bin_edges = sorted_vals[::bin_size][:n_bins]
    bin_edges = np.append(bin_edges, sorted_vals[-1] + 1e-10)
    return np.digitize(values, bin_edges[:-1]) - 1
