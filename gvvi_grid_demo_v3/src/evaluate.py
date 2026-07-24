"""Classification evaluation metrics."""
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate(model, features, edge_index_local, edge_index_context,
             targets_cls, test_mask, class_names, device):
    print("=" * 60)
    print("EVALUATE: 10-Class GVVI Classification")
    print("=" * 60)

    x_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    ei_local = edge_index_local.to(device)
    ei_context = edge_index_context.to(device) if edge_index_context is not None else None
    model = model.to(device); model.eval()

    with torch.no_grad():
        logits = model(x_tensor, ei_local, ei_context)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_cls = logits.argmax(dim=-1).cpu().numpy()

    y_true = targets_cls[test_mask]
    y_pred = pred_cls[test_mask]

    acc = accuracy_score(y_true, y_pred)
    # Accuracy within ±1 class
    within_1 = (np.abs(y_true - y_pred) <= 1).mean()
    # Mean absolute class error
    mae_cls = np.abs(y_true.astype(float) - y_pred.astype(float)).mean()
    # Spearman rank correlation between true class and predicted class
    sp, _ = spearmanr(y_true, y_pred)
    # Weighted F1
    from sklearn.metrics import f1_score
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # Soft metrics: expected class from probability distribution
    class_indices = np.arange(len(class_names))
    pred_expected = (probs[test_mask] * class_indices).sum(axis=1)
    mae_soft = np.abs(y_true.astype(float) - pred_expected).mean()
    sp_soft, _ = spearmanr(y_true, pred_expected)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": round(float(acc), 6),
        "accuracy_within_1": round(float(within_1), 6),
        "mae_class": round(float(mae_cls), 6),
        "mae_class_soft": round(float(mae_soft), 6),
        "spearman": round(float(sp), 6),
        "spearman_soft": round(float(sp_soft), 6),
        "f1_macro": round(float(f1_macro), 6),
        "f1_weighted": round(float(f1_weighted), 6),
    }

    print(f"Accuracy: {acc:.4f}")
    print(f"Accuracy ±1: {within_1:.4f}")
    print(f"MAE (class): {mae_cls:.4f}")
    print(f"MAE (soft): {mae_soft:.4f}")
    print(f"Spearman: {sp:.4f}")
    print(f"Spearman (soft): {sp_soft:.4f}")
    print(f"F1 macro: {f1_macro:.4f}, weighted: {f1_weighted:.4f}")

    # Show top confused pairs
    errors = np.abs(y_true - y_pred)
    print(f"Error distribution: exact={(errors==0).mean():.3f}, off-by-1={(errors==1).mean():.3f}, "
          f"off-by-2+={(errors>=2).mean():.3f}")

    return metrics, pred_cls, probs, cm
