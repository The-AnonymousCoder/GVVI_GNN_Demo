"""Training with composite loss: BCE + SmoothL1 + GVVI MSE + PairwiseRanking."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.05, n_pairs=512):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs

    def forward(self, pred_full, target_full, mask):
        """Sample high-low pairs from train nodes only."""
        # pred_full is already masked (only train nodes)
        n = len(pred_full)
        if n < 10:
            return torch.tensor(0.0, device=pred_full.device)
        n_pairs = min(self.n_pairs, n // 2)
        # Shuffle and split into high/low candidates
        perm = torch.randperm(n, device=pred_full.device)
        high_idx = perm[:n_pairs]
        low_idx = perm[n_pairs:2*n_pairs]
        valid = target_full[high_idx] > target_full[low_idx]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred_full.device)
        diff = pred_full[high_idx][valid] - pred_full[low_idx][valid]
        loss = F.relu(self.margin - diff).mean()
        return loss


def train_model(model, features, edge_index_local, edge_index_context,
                targets, train_mask, val_mask, cfg, device, checkpoint_dir):
    print("=" * 60)
    print("TRAIN: Multi-Scale Residual GraphSAGE")
    print("=" * 60)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    lr, wd = cfg["training"]["lr"], cfg["training"]["weight_decay"]
    max_epochs = cfg["training"]["max_epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    grad_clip = cfg["training"]["gradient_clip"]
    lam_cls = cfg["training"]["lambda_cls"]
    lam_reg = cfg["training"]["lambda_reg"]
    lam_gvvi = cfg["training"]["lambda_gvvi"]
    lam_rank = cfg["training"]["lambda_rank"]

    w_s, w_w, w_d = 3.0/6.0, 2.0/6.0, 1.0/6.0

    y_s = torch.tensor(targets["gvvi_s"], dtype=torch.float32).to(device)
    y_w = torch.tensor(targets["gvvi_w"], dtype=torch.float32).to(device)
    y_d = torch.tensor(targets["gvvi_d"], dtype=torch.float32).to(device)
    y_gvvi = torch.tensor(targets["gvvi"], dtype=torch.float32).to(device)
    y = torch.stack([y_s, y_w, y_d], dim=1)

    x_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    ei_local = edge_index_local.to(device)
    ei_context = edge_index_context.to(device) if edge_index_context is not None else None

    t_mask = torch.tensor(train_mask, dtype=torch.bool).to(device)
    v_mask = torch.tensor(val_mask, dtype=torch.bool).to(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6)
    rank_loss_fn = PairwiseRankingLoss(margin=0.05, n_pairs=512)
    bce_fn = nn.BCELoss()
    smooth_l1 = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    ckpt_path = f"{checkpoint_dir}/best_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"Device: {device}, Features: {features.shape[1]}")
    print(f"Train: {t_mask.sum().item()}, Val: {v_mask.sum().item()}")
    print(f"Loss weights: cls={lam_cls}, reg={lam_reg}, gvvi={lam_gvvi}, rank={lam_rank}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds, p_nonzeros, p_values = model(x_tensor, ei_local, ei_context)

        # BCE for nonzero classification
        targets_nz = (y > 0).float()
        loss_cls_s = bce_fn(p_nonzeros[t_mask, 0], targets_nz[t_mask, 0])
        loss_cls_w = bce_fn(p_nonzeros[t_mask, 1], targets_nz[t_mask, 1])
        loss_cls_d = bce_fn(p_nonzeros[t_mask, 2], targets_nz[t_mask, 2])
        loss_cls = (loss_cls_s + loss_cls_w + loss_cls_d) / 3.0

        # SmoothL1 for positive values (only where true > 0)
        loss_reg_s = torch.tensor(0.0, device=device)
        loss_reg_w = torch.tensor(0.0, device=device)
        loss_reg_d = torch.tensor(0.0, device=device)
        for j in range(3):
            mask_pos = t_mask & (y[:, j] > 0)
            if mask_pos.sum() > 0:
                if j == 0:
                    loss_reg_s = smooth_l1(p_values[mask_pos, 0], y[mask_pos, 0])
                elif j == 1:
                    loss_reg_w = smooth_l1(p_values[mask_pos, 1], y[mask_pos, 1])
                else:
                    loss_reg_d = smooth_l1(p_values[mask_pos, 2], y[mask_pos, 2])
        loss_reg = w_s * loss_reg_s + w_w * loss_reg_w + w_d * loss_reg_d

        # GVVI composite loss
        pred_gvvi = w_s * preds[t_mask, 0] + w_w * preds[t_mask, 1] + w_d * preds[t_mask, 2]
        true_gvvi = w_s * y[t_mask, 0] + w_w * y[t_mask, 1] + w_d * y[t_mask, 2]
        loss_gvvi = F.mse_loss(pred_gvvi, true_gvvi)

        # Pairwise ranking loss (pred_gvvi is already train-only)
        if lam_rank > 0:
            loss_rank = rank_loss_fn(pred_gvvi, true_gvvi, None)
        else:
            loss_rank = torch.tensor(0.0, device=device)

        loss = lam_cls * loss_cls + lam_reg * loss_reg + lam_gvvi * loss_gvvi + lam_rank * loss_rank

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Epoch {epoch:3d}: NaN/Inf loss, skipping")
            patience_counter += 1
            if patience_counter >= patience: break
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_preds, v_pnz, v_pv = model(x_tensor, ei_local, ei_context)
            v_pred_gvvi = w_s * v_preds[v_mask, 0] + w_w * v_preds[v_mask, 1] + w_d * v_preds[v_mask, 2]
            v_true_gvvi = w_s * y[v_mask, 0] + w_w * y[v_mask, 1] + w_d * y[v_mask, 2]
            val_loss = F.mse_loss(v_pred_gvvi, v_true_gvvi)

        if torch.isnan(val_loss) or torch.isinf(val_loss):
            patience_counter += 1
            if patience_counter >= patience: break
            continue

        scheduler.step(val_loss)

        if epoch <= 5 or epoch % 50 == 0:
            print(f"Epoch {epoch:4d}: loss={loss.item():.6f}, val={val_loss.item():.6f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}, "
                  f"cls={loss_cls.item():.4f}, reg={loss_reg.item():.4f}, "
                  f"gvvi={loss_gvvi.item():.4f}, rank={loss_rank.item():.4f}")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss; best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best: {best_epoch}, val={best_val_loss:.6f})")
            break

    if epoch == max_epochs:
        print(f"Reached max epochs {max_epochs} (best: {best_epoch}, val={best_val_loss:.6f})")

    model.load_state_dict(torch.load(ckpt_path))
    print(f"Best model loaded from epoch {best_epoch}\n")
    return model, best_epoch
