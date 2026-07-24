#!/usr/bin/env python3
"""
GVVI Graph Demo - Main Pipeline Runner

Orchestrates all phases:
  Phase 0: Data Audit
  Phase 1a: Explode Relations → positive_edges.parquet
  Phase 1b: Generate Candidate Edges
  Phase 1c: Spatial Split
  Phase 2: Replay Test
  Phase 3: Train & Evaluate Models
  Phase 4: Generate Report
"""

import os
import sys
import json
import time
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Base directory for data
BASE_DIR = os.environ.get('GVVI_BASE_DIR',
    "/Users/wangfugui/线上交流_forLI/05_3D_CIM_概念示例/GVVI_GNN_Demo实施包")


def phase_data_audit():
    """Phase 0: Generate data audit report."""
    print("\n" + "=" * 70)
    print("PHASE 0: Data Audit")
    print("=" * 70)

    import pandas as pd

    audit_lines = []
    audit_lines.append("# Data Audit Report\n")
    audit_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # File inventory
    audit_lines.append("## File Inventory\n")
    files = [
        ("relations/SVI.csv", "SVI relations (street view)"),
        ("relations/WVI.csv", "WVI relations (window view)"),
        ("relations/DVI.csv", "DVI relations (drone view)"),
        ("viewpoints/SVI_position_road.csv", "SVI viewpoint positions"),
        ("viewpoints/window_view_location.csv", "WVI viewpoint positions"),
        ("viewpoints/DVI_position_60.csv", "DVI viewpoint positions"),
        ("greenery/df_grid_ply.csv", "Greenery grid definitions"),
        ("greenery/veg_raster_buffer_100.tif", "Vegetation raster mask"),
        ("reference/GVVI_S.tif", "Official GVVI_S reference"),
        ("reference/GVVI_W.tif", "Official GVVI_W reference"),
        ("reference/GVVI_D.tif", "Official GVVI_D reference"),
        ("reference/GVVI.tif", "Official GVVI reference"),
    ]

    for fname, desc in files:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            audit_lines.append(f"- **{fname}**: {size_mb:.1f} MB — {desc}")
        else:
            audit_lines.append(f"- **{fname}**: MISSING — {desc}")

    # Relations data audit
    audit_lines.append("\n## Relations Data\n")
    for vt in ['SVI', 'WVI', 'DVI']:
        fpath = os.path.join(BASE_DIR, f"relations/{vt}.csv")
        df = pd.read_csv(fpath)
        audit_lines.append(f"\n### {vt}.csv\n")
        audit_lines.append(f"- Rows: {len(df)}")
        audit_lines.append(f"- Columns: {list(df.columns)}")
        audit_lines.append(f"- Non-null img_id: {df['img_id'].notna().sum()}")
        audit_lines.append(f"- Empty img_id: {(df['img_id'].isna() | (df['img_id'] == '')).sum()}")
        non_empty = df[df['img_id'].notna() & (df['img_id'] != '')]
        if len(non_empty) > 0:
            # Count unique view IDs
            all_ids = []
            for img_str in non_empty['img_id'].values[:1000]:
                all_ids.extend([x.strip() for x in img_str.split(',') if x.strip()])
            audit_lines.append(f"- Mean img_ids per row (sample): {len(all_ids)/1000:.1f}")

    # Viewpoint data audit
    audit_lines.append("\n## Viewpoint Data\n")
    vp_files = {
        'SVI': ('viewpoints/SVI_position_road.csv', 'id'),
        'WVI': ('viewpoints/window_view_location.csv', 'ID'),
        'DVI': ('viewpoints/DVI_position_60.csv', 'id'),
    }
    for vt, (fpath, id_col) in vp_files.items():
        fp = os.path.join(BASE_DIR, fpath)
        df = pd.read_csv(fp)
        audit_lines.append(f"\n### {vt}\n")
        audit_lines.append(f"- Rows: {len(df)}")
        audit_lines.append(f"- Columns: {list(df.columns)}")
        audit_lines.append(f"- Unique IDs: {df[id_col].nunique()}")

    # Greenery audit
    audit_lines.append("\n## Greenery Data\n")
    grn = pd.read_csv(os.path.join(BASE_DIR, "greenery/df_grid_ply.csv"))
    audit_lines.append(f"- Rows: {len(grn)}")
    audit_lines.append(f"- Unique (grid_idx_x, grid_idx_y): {len(grn.groupby(['grid_idx_x', 'grid_idx_y']))}")
    audit_lines.append(f"- grid_idx_x range: [{grn['grid_idx_x'].min()}, {grn['grid_idx_x'].max()}]")
    audit_lines.append(f"- grid_idx_y range: [{grn['grid_idx_y'].min()}, {grn['grid_idx_y'].max()}]")
    audit_lines.append(f"- X range: [{grn['grid_coords_min_x'].min():.1f}, {grn['grid_coords_max_x'].max():.1f}]")
    audit_lines.append(f"- Y range: [{grn['grid_coords_min_y'].min():.1f}, {grn['grid_coords_max_y'].max():.1f}]")

    # ID connection check
    audit_lines.append("\n## ID Connection Check\n")
    for vt in ['SVI', 'WVI', 'DVI']:
        rel = pd.read_csv(os.path.join(BASE_DIR, f"relations/{vt}.csv"))
        non_empty = rel[rel['img_id'].notna() & (rel['img_id'] != '')]

        all_ids = set()
        for img_str in non_empty['img_id']:
            all_ids.update([int(x.strip()) for x in img_str.split(',') if x.strip()])

        vp_fpath, vp_id_col = vp_files[vt]
        vp = pd.read_csv(os.path.join(BASE_DIR, vp_fpath))
        vp_ids = set(vp[vp_id_col].unique())

        connected = all_ids & vp_ids
        missing = all_ids - vp_ids
        audit_lines.append(f"\n### {vt}")
        audit_lines.append(f"- Unique view IDs in relations: {len(all_ids)}")
        audit_lines.append(f"- Viewpoint table IDs: {len(vp_ids)}")
        audit_lines.append(f"- Successfully connected: {len(connected)}")
        audit_lines.append(f"- Missing from viewpoint table: {len(missing)}")

    # Write report
    output_dir = os.path.join(BASE_DIR, "gvvi_graph_demo/reports")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "data_audit.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(audit_lines))
    print(f"Data audit report saved to {report_path}")
    print("Phase 0 complete.")


def phase_explode_relations():
    """Phase 1a: Explode relations CSV files."""
    print("\n" + "=" * 70)
    print("PHASE 1a: Explode Relations")
    print("=" * 70)

    from src.data.explode_relations import main as explode_main
    return explode_main()


def phase_candidate_edges():
    """Phase 1b: Generate candidate edges."""
    print("\n" + "=" * 70)
    print("PHASE 1b: Candidate Edge Generation")
    print("=" * 70)

    from src.data.candidate_edges import main as candidate_main
    return candidate_main()


def phase_spatial_split():
    """Phase 1c: Create spatial splits."""
    print("\n" + "=" * 70)
    print("PHASE 1c: Spatial Split")
    print("=" * 70)

    from src.data.spatial_split import main as split_main
    return split_main()


def phase_replay_test():
    """Phase 2: GVVI Replay Test."""
    print("\n" + "=" * 70)
    print("PHASE 2: GVVI Replay Test")
    print("=" * 70)

    from src.evaluation.official_gvvi_adapter import main as replay_main
    return replay_main()


def phase_train_models(label_budget=1.0, seed=42):
    """Phase 3: Train and evaluate all models."""
    print("\n" + "=" * 70)
    print(f"PHASE 3: Model Training (label_budget={label_budget}, seed={seed})")
    print("=" * 70)

    import pandas as pd
    import numpy as np

    from src.features.build_features import prepare_model_data
    from src.models.baselines import (StatisticalBaseline, MLPBaseline,
                                       XGBoostBaseline, run_baseline_experiment)

    # Prepare data
    obs_features, grn_features, edges_df, positives = prepare_model_data(BASE_DIR)

    # Subsample training labels if needed
    if label_budget < 1.0:
        np.random.seed(seed)
        train_mask = edges_df['split'] == 'train'
        train_view_ids = edges_df.loc[train_mask, ['view_type', 'view_id']].drop_duplicates()
        n_train_views = len(train_view_ids)
        n_sample = max(int(n_train_views * label_budget), 10)

        sampled_views = train_view_ids.sample(n=n_sample, random_state=seed)
        sampled_view_set = set()
        for _, row in sampled_views.iterrows():
            sampled_view_set.add((row['view_type'], row['view_id']))

        # Zero out labels for unsampled views in training set
        unsampled_mask = train_mask & ~edges_df.apply(
            lambda r: (r['view_type'], r['view_id']) in sampled_view_set, axis=1)
        # Temporarily move unsampled training edges to a "hidden" state
        # For baseline models, we simply filter the training set
        edges_df_modified = edges_df.copy()
        training_set = edges_df_modified['split'] == 'train'
        in_sample = edges_df_modified.apply(
            lambda r: (r['view_type'], r['view_id']) in sampled_view_set, axis=1)
        edges_df_modified.loc[training_set & ~in_sample, 'split'] = 'hidden_train'

        print(f"  Label budget: {label_budget} ({n_sample}/{n_train_views} views)")
        # Revert for model training: use hidden_train as part of train for edges that have labels
        # Actually, we should use 'hidden_train' edges for training with their labels
        edges_df_for_training = edges_df_modified.copy()
    else:
        edges_df_for_training = edges_df.copy()

    all_results = []

    # Baseline 0: Statistical
    print("\n--- Statistical Baseline ---")
    stat_model = StatisticalBaseline()
    stat_results = run_baseline_experiment(BASE_DIR, "Statistical", stat_model,
                                            edges_df_for_training)
    all_results.append(stat_results)

    # Baseline 1: XGBoost
    print("\n--- XGBoost Baseline ---")
    try:
        xgb_model = XGBoostBaseline(n_estimators=100, max_depth=6)
        xgb_results = run_baseline_experiment(BASE_DIR, "XGBoost", xgb_model,
                                               edges_df_for_training)
        all_results.append(xgb_results)
    except ImportError:
        print("  XGBoost not available. Install with: pip install xgboost")
        all_results.append({'model': 'XGBoost', 'error': 'not_available'})

    # Baseline 2: MLP
    print("\n--- MLP Baseline ---")
    mlp_model = MLPBaseline(hidden_dims=(64, 32), max_iter=200)
    mlp_results = run_baseline_experiment(BASE_DIR, "MLP", mlp_model,
                                           edges_df_for_training)
    all_results.append(mlp_results)

    # GraphSAGE
    print("\n--- GraphSAGE Model ---")
    try:
        import torch
        from src.models.graphsage_model import run_graphsage_experiment

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")

        # Define feature columns
        obs_feat_cols = ['x', 'y', 'z', 'heading_sin', 'heading_cos',
                         'is_svi', 'is_wvi', 'is_dvi', 'rel_height']
        grn_feat_cols = ['center_x', 'center_y', 'center_z',
                         'width', 'height', 'area',
                         'R_norm', 'G_norm', 'B_norm', 'greenness']
        edge_feat_cols = ['horizontal_distance', 'euclidean_distance_3d',
                          'vertical_distance', 'azimuth_diff_sin', 'azimuth_diff_cos',
                          'log_horizontal_distance', 'log_euclidean_distance_3d',
                          'inv_horizontal_distance']

        model, data, gs_results, vis_proba, pixel_pred = run_graphsage_experiment(
            obs_features, grn_features, edges_df_for_training,
            obs_feat_cols, grn_feat_cols, edge_feat_cols,
            hidden_dim=64, num_layers=2, dropout=0.3,
            epochs=200, lr=0.001, device=device
        )
        all_results.append(gs_results)
    except ImportError as e:
        print(f"  GraphSAGE not available: {e}")
        print("  Install with: pip install torch torch-geometric")
        all_results.append({'model': 'GraphSAGE', 'error': str(e)})
    except Exception as e:
        print(f"  GraphSAGE error: {e}")
        import traceback
        traceback.print_exc()
        all_results.append({'model': 'GraphSAGE', 'error': str(e)})

    # Save results
    output_dir = os.path.join(BASE_DIR, "gvvi_graph_demo/outputs/metrics")
    os.makedirs(output_dir, exist_ok=True)

    budget_str = f"budget{int(label_budget*100):02d}"
    seed_str = f"seed{seed}"
    results_path = os.path.join(output_dir, f"results_{budget_str}_{seed_str}.json")

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    return all_results


def phase_generate_report():
    """Phase 4: Generate final report."""
    print("\n" + "=" * 70)
    print("PHASE 4: Generate Report")
    print("=" * 70)

    import json
    import glob

    output_dir = os.path.join(BASE_DIR, "gvvi_graph_demo/outputs")
    metrics_dir = os.path.join(output_dir, "metrics")

    # Collect all results
    all_result_files = glob.glob(os.path.join(metrics_dir, "results_*.json"))

    report_lines = []
    report_lines.append("# GVVI Graph Demo — Final Report\n")
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Replay test results
    replay_path = os.path.join(output_dir, "replay_test.json")
    if os.path.exists(replay_path):
        with open(replay_path) as f:
            replay = json.load(f)
        report_lines.append("## Replay Test\n")
        report_lines.append(f"All pass: {replay.get('all_pass', 'N/A')}\n")
        for name, c in replay.get('results', {}).items():
            report_lines.append(f"- **{name}**: max_error={c['max_error']:.10f}, "
                              f"consistent={c['consistent_ratio']*100:.1f}%\n")

    # Candidate recall
    recall_path = os.path.join(BASE_DIR, "gvvi_graph_demo/processed/candidate_recall.json")
    if os.path.exists(recall_path):
        with open(recall_path) as f:
            recall = json.load(f)
        report_lines.append("\n## Candidate Recall\n")
        report_lines.append(f"Overall: {recall['overall_recall']*100:.2f}%\n")
        for vt, info in recall.get('per_type', {}).items():
            report_lines.append(f"- {vt}: {info['recall']*100:.2f}% "
                              f"({info['captured']}/{info['total_positive']})\n")

    # Model results
    report_lines.append("\n## Model Comparison\n")

    if all_result_files:
        all_results = []
        for fpath in sorted(all_result_files):
            with open(fpath) as f:
                results = json.load(f)
                all_results.extend(results)

        # Build comparison table
        models = set()
        for r in all_results:
            if 'model' in r:
                models.add(r['model'])

        report_lines.append("\n### Key Metrics (Test Set)\n")
        report_lines.append("| Model | AUPRC | AUROC | F1 | Pixel MAE | Train Time |")
        report_lines.append("|-------|-------|-------|----|-----------|------------|")

        for r in all_results:
            if 'error' in r:
                report_lines.append(f"| {r['model']} | — | — | — | — | {r['error']} |")
            else:
                report_lines.append(
                    f"| {r.get('model', '?')} "
                    f"| {r.get('vis_test_auprc', 0):.4f} "
                    f"| {r.get('vis_test_auroc', 0):.4f} "
                    f"| {r.get('vis_test_f1', 0):.4f} "
                    f"| {r.get('pix_all_mae', 0):.1f} "
                    f"| {r.get('train_time_s', 0):.1f}s |"
                )

        # Per view type analysis
        report_lines.append("\n### Per-View-Type Visibility AUPRC\n")
        report_lines.append("| Model | SVI | WVI | DVI |")
        report_lines.append("|-------|-----|-----|-----|")

        for r in all_results:
            if 'error' not in r:
                report_lines.append(
                    f"| {r.get('model', '?')} "
                    f"| {r.get('SVI_vis_auprc', 0):.4f} "
                    f"| {r.get('WVI_vis_auprc', 0):.4f} "
                    f"| {r.get('DVI_vis_auprc', 0):.4f} |"
                )

        # Compare GNN vs baselines
        report_lines.append("\n### GNN vs Baselines Analysis\n")
        gnn_results = [r for r in all_results if 'GraphSAGE' in str(r.get('model', ''))]
        xgb_results = [r for r in all_results if 'XGBoost' in str(r.get('model', ''))]
        mlp_results = [r for r in all_results if 'MLP' in str(r.get('model', ''))]

        if gnn_results and xgb_results:
            gnn_auprc = gnn_results[0].get('vis_test_auprc', 0)
            xgb_auprc = xgb_results[0].get('vis_test_auprc', 0)
            if gnn_auprc > xgb_auprc:
                report_lines.append(
                    f"✓ GNN (AUPRC={gnn_auprc:.4f}) outperforms XGBoost (AUPRC={xgb_auprc:.4f})\n")
            else:
                report_lines.append(
                    f"✗ GNN (AUPRC={gnn_auprc:.4f}) does NOT outperform XGBoost (AUPRC={xgb_auprc:.4f})\n")

        if gnn_results and mlp_results:
            gnn_auprc = gnn_results[0].get('vis_test_auprc', 0)
            mlp_auprc = mlp_results[0].get('vis_test_auprc', 0)
            if gnn_auprc > mlp_auprc:
                report_lines.append(
                    f"✓ GNN (AUPRC={gnn_auprc:.4f}) outperforms MLP (AUPRC={mlp_auprc:.4f}) "
                    f"— graph structure provides gain\n")
            else:
                report_lines.append(
                    f"✗ GNN (AUPRC={gnn_auprc:.4f}) does NOT outperform MLP (AUPRC={mlp_auprc:.4f}) "
                    f"— graph structure does not help\n")

    # Conclusion
    report_lines.append("\n## Questions to Answer\n")
    report_lines.append("1. Candidate Recall ≥ 99%? → See Candidate Recall section\n")
    report_lines.append("2. GNN significantly better than XGBoost/MLP? → See comparison above\n")
    report_lines.append("3. Which view type is easiest/hardest? → See per-type metrics\n")

    report_path = os.path.join(BASE_DIR, "gvvi_graph_demo/reports/demo_result.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='GVVI Graph Demo Pipeline')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'audit', 'explode', 'candidates',
                                 'split', 'replay', 'train', 'report'],
                        help='Which phase to run')
    parser.add_argument('--label-budget', type=float, default=1.0,
                        help='Fraction of training labels to use (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip-gnn', action='store_true',
                        help='Skip GNN training (if PyTorch not available)')
    args = parser.parse_args()

    total_start = time.time()

    if args.phase in ['all', 'audit']:
        phase_data_audit()

    if args.phase in ['all', 'explode']:
        phase_explode_relations()

    if args.phase in ['all', 'candidates']:
        phase_candidate_edges()

    if args.phase in ['all', 'split']:
        phase_spatial_split()

    if args.phase in ['all', 'replay']:
        phase_replay_test()

    if args.phase in ['all', 'train']:
        phase_train_models(label_budget=args.label_budget, seed=args.seed)

    if args.phase in ['all', 'report']:
        phase_generate_report()

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete. Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
