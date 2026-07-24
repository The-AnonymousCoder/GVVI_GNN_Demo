"""
Phase 1b: Generate candidate edges using cheap geometric rules.

Uses KNN from viewpoints to greenery + vectorized FoV filtering.
Separate parameters per view type based on data analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_data(base_dir):
    """Load all data."""
    viewpoints = []

    svi = pd.read_csv(os.path.join(base_dir, "viewpoints/SVI_position_road.csv"))
    svi['view_type'] = 'SVI'
    svi = svi.rename(columns={'id': 'view_id', 'X': 'x', 'Y': 'y', 'Z': 'z', 'Heading': 'heading'})
    viewpoints.append(svi[['view_type', 'view_id', 'x', 'y', 'z', 'heading']])

    wvi = pd.read_csv(os.path.join(base_dir, "viewpoints/window_view_location.csv"))
    wvi['view_type'] = 'WVI'
    wvi = wvi.rename(columns={'ID': 'view_id', 'X': 'x', 'Y': 'y', 'Z': 'z', 'Heading': 'heading'})
    viewpoints.append(wvi[['view_type', 'view_id', 'x', 'y', 'z', 'heading']])

    dvi = pd.read_csv(os.path.join(base_dir, "viewpoints/DVI_position_60.csv"))
    dvi['view_type'] = 'DVI'
    dvi = dvi.rename(columns={'id': 'view_id'})
    if 'heading' not in dvi.columns:
        dvi = dvi.rename(columns={'heading_id': 'heading'})
    viewpoints.append(dvi[['view_type', 'view_id', 'x', 'y', 'z', 'heading']])

    all_vp = pd.concat(viewpoints, ignore_index=True)
    print(f"Loaded {len(all_vp)} viewpoints")
    for vt in ['SVI', 'WVI', 'DVI']:
        v = all_vp[all_vp['view_type'] == vt]
        print(f"  {vt}: {len(v)} views, {v[['x','y']].drop_duplicates().shape[0]} unique positions")

    greenery = pd.read_csv(os.path.join(base_dir, "greenery/df_grid_ply.csv"))
    greenery['greenery_id'] = greenery.apply(
        lambda r: f"{int(r['grid_idx_x'])}_{int(r['grid_idx_y'])}", axis=1)
    greenery['center_x'] = (greenery['grid_coords_min_x'] + greenery['grid_coords_max_x']) / 2
    greenery['center_y'] = (greenery['grid_coords_min_y'] + greenery['grid_coords_max_y']) / 2
    greenery['center_z'] = 0.0
    print(f"Loaded {len(greenery)} greenery grids")

    return all_vp, greenery


def heading_to_direction(headings_deg):
    """Convert headings to 2D direction vectors."""
    a = np.radians(90 - np.array(headings_deg, dtype=np.float32))
    return np.column_stack([np.cos(a), np.sin(a)])


def generate_for_viewtype(vp_subset, greenery, k_nearest, max_dist,
                           fov_deg, margin_deg, apply_fov=True):
    """
    Generate candidates for one view type. Fully vectorized, single-pass.

    Returns DataFrame.
    """
    n_views = len(vp_subset)
    n_grn = len(greenery)

    if n_views == 0:
        return pd.DataFrame()

    grn_coords = greenery[['center_x', 'center_y']].values.astype(np.float32)
    grn_z = greenery['center_z'].values.astype(np.float32)
    grn_ids = greenery['greenery_id'].values

    view_coords = vp_subset[['x', 'y']].values.astype(np.float32)
    view_z = vp_subset['z'].values.astype(np.float32)
    view_types = vp_subset['view_type'].values
    view_ids = vp_subset['view_id'].values

    k = min(k_nearest, n_grn)
    print(f"    KDTree.query(k={k}) for {n_views} views...")
    tree = KDTree(grn_coords)
    distances, indices = tree.query(view_coords, k=k)

    # Flatten
    n_pairs = n_views * k
    view_idx_flat = np.repeat(np.arange(n_views), k)
    grn_idx_flat = indices.ravel()
    dist_flat = distances.ravel()

    # Filter by max_dist
    dist_mask = dist_flat <= max_dist
    n_within = dist_mask.sum()
    print(f"    Within {max_dist}m: {n_within:,} / {n_pairs:,}")

    if n_within == 0:
        return pd.DataFrame()

    vi = view_idx_flat[dist_mask]
    gi = grn_idx_flat[dist_mask]
    di = dist_flat[dist_mask]

    del view_idx_flat, grn_idx_flat, dist_flat, distances, indices

    if apply_fov:
        # FoV check
        vp_pos = view_coords[vi]
        grn_pos = grn_coords[gi]
        to_green_norm = (grn_pos - vp_pos) / di[:, np.newaxis]

        headings = vp_subset['heading'].values.astype(np.float32)
        cam_dirs = heading_to_direction(headings)
        vp_dirs = cam_dirs[vi]
        cos_angles = np.sum(to_green_norm * vp_dirs, axis=1)

        half_fov_cos = np.cos(np.radians(fov_deg / 2 + margin_deg))
        fov_mask = cos_angles >= half_fov_cos
        n_fov = fov_mask.sum()
        print(f"    In FoV: {n_fov:,} / {n_within:,}")

        if n_fov == 0:
            return pd.DataFrame()

        vi = vi[fov_mask]
        gi = gi[fov_mask]
        di = di[fov_mask]
        ca = cos_angles[fov_mask]
        angle_diff = np.degrees(np.arccos(np.clip(ca, -1, 1)))
    else:
        angle_diff = np.zeros_like(di)

    vz = view_z[vi]
    gz = grn_z[gi]

    result = pd.DataFrame({
        'view_type': view_types[vi],
        'view_id': view_ids[vi].astype(np.int64),
        'greenery_id': grn_ids[gi],
        'horizontal_distance': di.astype(np.float32),
        'euclidean_distance_3d': np.sqrt(di.astype(np.float64)**2 + (vz - gz).astype(np.float64)**2).astype(np.float32),
        'vertical_distance': np.abs(vz - gz).astype(np.float32),
        'azimuth_difference_deg': angle_diff.astype(np.float32),
    })

    return result


def compute_recall(candidates_df, positive_edges_df):
    """Compute recall using set operations."""
    print("  Computing recall...")

    # Build strings in chunks
    def build_key_set(df, col_prefix=''):
        keys = set()
        chunk_size = 2000000
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start+chunk_size]
            k = (chunk['view_type'] + '_' +
                 chunk['view_id'].astype(str) + '_' +
                 chunk['greenery_id'])
            keys.update(k.values)
        return keys

    pos_keys = build_key_set(positive_edges_df)
    cand_keys = build_key_set(candidates_df)

    captured = pos_keys & cand_keys
    recall = len(captured) / len(pos_keys)

    per_type = {}
    for vt in ['SVI', 'WVI', 'DVI']:
        vt_pos = positive_edges_df[positive_edges_df['view_type'] == vt]
        vt_pk = build_key_set(vt_pos)
        vt_cap = vt_pk & cand_keys
        per_type[vt] = {'recall': len(vt_cap) / len(vt_pk) if len(vt_pk) > 0 else 0,
                        'captured': len(vt_cap), 'total': len(vt_pk)}

    return recall, per_type, len(pos_keys - cand_keys), len(captured)


def main():
    base_dir = os.environ.get('GVVI_BASE_DIR',
        "/Users/wangfugui/线上交流_forLI/05_3D_CIM_概念示例/GVVI_GNN_Demo实施包")
    output_dir = os.path.join(base_dir, "gvvi_graph_demo/processed")
    os.makedirs(output_dir, exist_ok=True)

    viewpoints, greenery = load_data(base_dir)
    pos_path = os.path.join(output_dir, "positive_edges.parquet")
    positive_edges = pd.read_parquet(pos_path)

    # Config per iteration
    configs = [
        # (k_svi, d_svi, fov_s, m_s, k_wvi, d_wvi, fov_w, m_w, k_dvi, d_dvi)
        (200,  800, 120, 30,  400,  1300, 120, 30,  500,  2000),
        (300, 1000, 150, 45,  600,  1500, 150, 45, 1000,  2500),
        (400, 1200, 200, 60,  800,  1800, 200, 60, 1500,  3000),
        (500, 1500, 240, 60, 1000,  2000, 240, 60, 2000,  3500),
        (500, 2000, 240, 90, 1000,  2500, 240, 90, 3000,  4000),
    ]

    for iteration, (ks, ds, fs, ms, kw, dw, fw, mw, kd, dd) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}")
        print(f"  SVI: k={ks}, d={ds}m, fov={fs}°, margin={ms}°")
        print(f"  WVI: k={kw}, d={dw}m, fov={fw}°, margin={mw}°")
        print(f"  DVI: k={kd}, d={dd}m (no FoV)")

        svi_vp = viewpoints[viewpoints['view_type'] == 'SVI']
        wvi_vp = viewpoints[viewpoints['view_type'] == 'WVI']
        dvi_vp = viewpoints[viewpoints['view_type'] == 'DVI']

        svi_cand = generate_for_viewtype(svi_vp, greenery, ks, ds, fs, ms, apply_fov=True)
        print(f"  SVI: {len(svi_cand):,} candidates")

        wvi_cand = generate_for_viewtype(wvi_vp, greenery, kw, dw, fw, mw, apply_fov=True)
        print(f"  WVI: {len(wvi_cand):,} candidates")

        dvi_cand = generate_for_viewtype(dvi_vp, greenery, kd, dd, 0, 0, apply_fov=False)
        print(f"  DVI: {len(dvi_cand):,} candidates")

        candidates = pd.concat([svi_cand, wvi_cand, dvi_cand], ignore_index=True)

        recall, per_type, missed, captured = compute_recall(candidates, positive_edges)
        print(f"\nOverall Recall: {recall:.4f} ({recall*100:.2f}%)")
        for vt in ['SVI', 'WVI', 'DVI']:
            print(f"  {vt}: {per_type[vt]['recall']:.4f} "
                  f"({per_type[vt]['captured']:,}/{per_type[vt]['total']:,})")
        print(f"  Missed: {missed:,}  |  Total candidates: {len(candidates):,}")

        if recall >= 0.99 and all(per_type[vt]['recall'] >= 0.99 for vt in ['SVI', 'WVI', 'DVI']):
            print("✓ 99% recall target achieved!")
            break
        print("  Continuing to next iteration...")

    # Save
    cand_path = os.path.join(output_dir, "candidate_edges.parquet")
    candidates.to_parquet(cand_path, index=False)
    print(f"\nSaved {len(candidates):,} candidate edges ({os.path.getsize(cand_path)/1024/1024:.1f} MB)")

    report = {
        'overall_recall': float(recall),
        'per_type': {vt: {'recall': float(per_type[vt]['recall']),
                          'captured': int(per_type[vt]['captured']),
                          'total_positive': int(per_type[vt]['total'])}
                     for vt in ['SVI', 'WVI', 'DVI']},
        'total_missed': int(missed),
        'total_candidates': int(len(candidates)),
    }
    with open(os.path.join(output_dir, "candidate_recall.json"), 'w') as f:
        json.dump(report, f, indent=2)

    return candidates, report


if __name__ == '__main__':
    main()
