"""
Phase 3a: Baseline Models.

Implements:
- Baseline 0: Simple statistical baselines
- Baseline 1: XGBoost
- Baseline 2: MLP (Multi-Layer Perceptron)
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_score, recall_score, f1_score,
                              mean_absolute_error, mean_squared_error)
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# Baseline 0: Statistical Baselines
# ============================================================

class StatisticalBaseline:
    """
    Simple statistical baselines for visibility and pixel prediction.
    """
    def __init__(self):
        self.visibility_rate = {}  # view_type -> mean visibility
        self.pixel_mean_by_dist = {}  # view_type -> distance_bin -> mean pixel
        self.dist_bins = None

    def fit(self, edges_df):
        """Fit statistical baselines on training data."""
        train = edges_df[edges_df['split'] == 'train'].copy()

        for vt in ['SVI', 'WVI', 'DVI']:
            vt_data = train[train['view_type'] == vt]
            self.visibility_rate[vt] = vt_data['is_visible'].mean()
            print(f"  {vt} visibility rate: {self.visibility_rate[vt]:.4f}")

        # Distance-binned pixel predictions
        self.dist_bins = [0, 50, 100, 200, 300, 500]
        for vt in ['SVI', 'WVI', 'DVI']:
            vt_data = train[train['view_type'] == vt]
            vt_visible = vt_data[vt_data['is_visible'] == 1]
            self.pixel_mean_by_dist[vt] = {}
            for i in range(len(self.dist_bins) - 1):
                lo, hi = self.dist_bins[i], self.dist_bins[i+1]
                mask = (vt_visible['horizontal_distance'] >= lo) & (vt_visible['horizontal_distance'] < hi)
                if mask.sum() > 0:
                    self.pixel_mean_by_dist[vt][(lo, hi)] = vt_visible.loc[mask, 'pixel_count'].mean()
                else:
                    self.pixel_mean_by_dist[vt][(lo, hi)] = 0

    def predict_visibility(self, edges_df):
        """Predict visibility probability for each edge."""
        probs = np.zeros(len(edges_df))
        for vt in ['SVI', 'WVI', 'DVI']:
            mask = edges_df['view_type'] == vt
            probs[mask] = self.visibility_rate.get(vt, 0.5)
        return probs

    def predict_pixels(self, edges_df):
        """Predict pixel count for each edge."""
        preds = np.zeros(len(edges_df))
        for i in range(len(self.dist_bins) - 1):
            lo, hi = self.dist_bins[i], self.dist_bins[i+1]
            mask = (edges_df['horizontal_distance'] >= lo) & (edges_df['horizontal_distance'] < hi)
            for vt in ['SVI', 'WVI', 'DVI']:
                vt_mask = mask & (edges_df['view_type'] == vt)
                preds[vt_mask] = self.pixel_mean_by_dist[vt].get((lo, hi), 0)
        return preds

    def predict(self, edges_df):
        """Combined prediction: p(visible) * predicted_pixels."""
        return self.predict_visibility(edges_df) * self.predict_pixels(edges_df)


# ============================================================
# Baseline 2: MLP (via sklearn)
# ============================================================

class MLPBaseline:
    """
    MLP baseline using sklearn's MLPClassifier and MLPRegressor.
    """
    def __init__(self, hidden_dims=(64, 32), max_iter=200):
        self.hidden_dims = hidden_dims
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.vis_classifier = None
        self.pixel_regressor = None

    def _prepare_features(self, edges_df):
        """Prepare feature matrix from edges."""
        feature_cols = [
            'horizontal_distance', 'euclidean_distance_3d', 'vertical_distance',
            'azimuth_diff_sin', 'azimuth_diff_cos',
            'log_horizontal_distance', 'log_euclidean_distance_3d',
            'inv_horizontal_distance'
        ]
        available = [c for c in feature_cols if c in edges_df.columns]
        return edges_df[available].values

    def fit(self, edges_df):
        """Train MLP on training data."""
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        train = edges_df[edges_df['split'] == 'train'].copy()
        X = self._prepare_features(train)
        X = self.scaler.fit_transform(X)
        y_vis = train['is_visible'].values

        print(f"  Training visibility classifier on {len(train)} samples...")
        self.vis_classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_dims,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.vis_classifier.fit(X, y_vis)

        # Train pixel regressor on visible edges only
        visible_mask = y_vis == 1
        if visible_mask.sum() > 0:
            print(f"  Training pixel regressor on {visible_mask.sum()} visible edges...")
            self.pixel_regressor = MLPRegressor(
                hidden_layer_sizes=self.hidden_dims,
                max_iter=self.max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            y_pixel = np.log1p(train.loc[visible_mask, 'pixel_count'].values)
            self.pixel_regressor.fit(X[visible_mask], y_pixel)

    def predict_visibility_proba(self, edges_df):
        """Predict visibility probabilities."""
        X = self._prepare_features(edges_df)
        X = self.scaler.transform(X)
        return self.vis_classifier.predict_proba(X)[:, 1]

    def predict_pixels(self, edges_df):
        """Predict pixel counts (log1p scale)."""
        X = self._prepare_features(edges_df)
        X = self.scaler.transform(X)
        pred_log = self.pixel_regressor.predict(X)
        return np.expm1(pred_log).clip(0, None)

    def predict(self, edges_df):
        """Combined prediction."""
        return self.predict_visibility_proba(edges_df) * self.predict_pixels(edges_df)


# ============================================================
# Baseline 1: XGBoost
# ============================================================

class XGBoostBaseline:
    """
    XGBoost baseline for visibility classification and pixel regression.
    """
    def __init__(self, n_estimators=100, max_depth=6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.scaler = StandardScaler()
        self.vis_model = None
        self.pixel_model = None

    def _prepare_features(self, edges_df):
        feature_cols = [
            'horizontal_distance', 'euclidean_distance_3d', 'vertical_distance',
            'azimuth_diff_sin', 'azimuth_diff_cos',
            'log_horizontal_distance', 'log_euclidean_distance_3d',
            'inv_horizontal_distance'
        ]
        available = [c for c in feature_cols if c in edges_df.columns]
        return edges_df[available].values

    def fit(self, edges_df):
        """Train XGBoost models."""
        import xgboost as xgb

        train = edges_df[edges_df['split'] == 'train'].copy()
        X = self._prepare_features(train)
        X = self.scaler.fit_transform(X)
        y_vis = train['is_visible'].values

        # Compute scale_pos_weight for class imbalance
        neg_count = (y_vis == 0).sum()
        pos_count = (y_vis == 1).sum()
        scale_pos_weight = neg_count / max(pos_count, 1)

        print(f"  Class balance: {neg_count} neg / {pos_count} pos (weight={scale_pos_weight:.1f})")

        self.vis_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='aucpr'
        )
        self.vis_model.fit(X, y_vis)

        # Pixel regressor
        visible_mask = y_vis == 1
        if visible_mask.sum() > 0:
            self.pixel_model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
            y_pixel = np.log1p(train.loc[visible_mask, 'pixel_count'].values)
            self.pixel_model.fit(X[visible_mask], y_pixel)

    def predict_visibility_proba(self, edges_df):
        X = self._prepare_features(edges_df)
        X = self.scaler.transform(X)
        return self.vis_model.predict_proba(X)[:, 1]

    def predict_pixels(self, edges_df):
        X = self._prepare_features(edges_df)
        X = self.scaler.transform(X)
        pred_log = self.pixel_model.predict(X)
        return np.expm1(pred_log).clip(0, None)

    def predict(self, edges_df):
        return self.predict_visibility_proba(edges_df) * self.predict_pixels(edges_df)


# ============================================================
# Evaluation Utilities
# ============================================================

def evaluate_visibility(edges_df, y_true, y_pred_proba, threshold=0.5, prefix=''):
    """Evaluate visibility prediction."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        f'{prefix}auprc': average_precision_score(y_true, y_pred_proba),
        f'{prefix}auroc': roc_auc_score(y_true, y_pred_proba),
        f'{prefix}precision': precision_score(y_true, y_pred, zero_division=0),
        f'{prefix}recall': recall_score(y_true, y_pred, zero_division=0),
        f'{prefix}f1': f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def evaluate_pixels(edges_df, y_true, y_pred, mask=None, prefix=''):
    """Evaluate pixel count prediction."""
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)

    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {f'{prefix}mae': 0, f'{prefix}rmse': 0,
                f'{prefix}log_mae': 0, f'{prefix}spearman': 0}

    metrics = {
        f'{prefix}mae': mean_absolute_error(y_t, y_p),
        f'{prefix}rmse': np.sqrt(mean_squared_error(y_t, y_p)),
        f'{prefix}log_mae': mean_absolute_error(np.log1p(y_t), np.log1p(y_p.clip(0, None))),
        f'{prefix}spearman': spearmanr(y_t, y_p)[0],
    }
    return metrics


def evaluate_gvvi(gvvi_true, gvvi_pred, mask, prefix=''):
    """Evaluate GVVI prediction."""
    t = gvvi_true[mask]
    p = gvvi_pred[mask]

    if len(t) == 0:
        return {}

    # Top 10% overlap
    top10_thresh_true = np.percentile(t, 90)
    top10_thresh_pred = np.percentile(p, 90)
    top10_true = set(np.where(t >= top10_thresh_true)[0])
    top10_pred = set(np.where(p >= top10_thresh_pred)[0])
    top10_overlap = len(top10_true & top10_pred) / len(top10_true) if len(top10_true) > 0 else 0

    # Bottom 10%
    bot10_thresh_true = np.percentile(t, 10)
    bot10_thresh_pred = np.percentile(p, 10)
    bot10_true = set(np.where(t <= bot10_thresh_true)[0])
    bot10_pred = set(np.where(p <= bot10_thresh_pred)[0])
    bot10_overlap = len(bot10_true & bot10_pred) / len(bot10_true) if len(bot10_true) > 0 else 0

    metrics = {
        f'{prefix}mae': mean_absolute_error(t, p),
        f'{prefix}nmae': mean_absolute_error(t, p) / (t.max() - t.min()) if t.max() > t.min() else 0,
        f'{prefix}rmse': np.sqrt(mean_squared_error(t, p)),
        f'{prefix}r2': 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum() if t.std() > 0 else 0,
        f'{prefix}pearson': pearsonr(t, p)[0],
        f'{prefix}spearman': spearmanr(t, p)[0],
        f'{prefix}top10_overlap': top10_overlap,
        f'{prefix}bot10_overlap': bot10_overlap,
    }
    return metrics


def run_baseline_experiment(base_dir, model_name, model, edges_df):
    """
    Run a baseline experiment and return all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Running {model_name}...")

    t0 = time.time()

    train_mask = edges_df['split'] == 'train'
    val_mask = edges_df['split'] == 'val'
    test_mask = edges_df['split'] == 'test'

    # Fit
    model.fit(edges_df)

    # Predict
    t1 = time.time()
    train_time = t1 - t0

    # Visibility predictions
    vis_proba_test = model.predict_visibility_proba(edges_df[test_mask])
    pixel_pred_test = model.predict_pixels(edges_df[test_mask])

    y_vis_test = edges_df.loc[test_mask, 'is_visible'].values
    y_pix_test = edges_df.loc[test_mask, 'pixel_count'].values

    t2 = time.time()
    infer_time = t2 - t1

    # Evaluate
    vis_metrics = evaluate_visibility(
        edges_df[test_mask], y_vis_test, vis_proba_test, prefix='vis_test_')

    # Pixel metrics
    vis_true_mask = y_vis_test == 1
    all_pix_metrics = evaluate_pixels(
        edges_df[test_mask], y_pix_test, pixel_pred_test, prefix='pix_all_')
    vis_pix_metrics = evaluate_pixels(
        edges_df[test_mask], y_pix_test, pixel_pred_test,
        mask=vis_true_mask, prefix='pix_visible_')

    all_metrics = {
        'model': model_name,
        'train_time_s': train_time,
        'infer_time_s': infer_time,
        **vis_metrics,
        **all_pix_metrics,
        **vis_pix_metrics,
    }

    # Per view type
    for vt in ['SVI', 'WVI', 'DVI']:
        vt_mask = test_mask & (edges_df['view_type'] == vt)
        if vt_mask.sum() == 0:
            continue
        y_vis_vt = edges_df.loc[vt_mask, 'is_visible'].values
        vis_proba_vt = model.predict_visibility_proba(edges_df[vt_mask])
        vt_vis = evaluate_visibility(edges_df[vt_mask], y_vis_vt, vis_proba_vt,
                                      prefix=f'{vt}_vis_')
        all_metrics.update(vt_vis)

    # Print summary
    print(f"  {model_name} Results (test set):")
    print(f"    AUPRC: {all_metrics['vis_test_auprc']:.4f}")
    print(f"    AUROC: {all_metrics['vis_test_auroc']:.4f}")
    print(f"    F1: {all_metrics['vis_test_f1']:.4f}")
    print(f"    Pixel MAE (all): {all_metrics['pix_all_mae']:.2f}")
    print(f"    Pixel MAE (visible): {all_metrics['pix_visible_mae']:.2f}")
    print(f"    Train time: {all_metrics['train_time_s']:.2f}s")
    print(f"    Infer time: {all_metrics['infer_time_s']:.2f}s")

    return all_metrics


if __name__ == '__main__':
    print("Baseline models module loaded. Use run_pipeline.py to execute experiments.")
