"""Classification training with ordinal-aware loss."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropyLoss(nn.Module):
    """Cross-entropy + ordinal penalty for class distance."""
    def __init__(self, num_classes=10, ordinal_weight=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        if self.ordinal_weight <= 0:
            return ce_loss
        probs = F.softmax(logits, dim=-1)
        class_indices = torch.arange(self.num_classes, device=logits.device).float()
        pred_expected = (probs * class_indices).sum(dim=-1)
        ordinal_loss = F.mse_loss(pred_expected, targets.float())
        return ce_loss + self.ordinal_weight * ordinal_loss


def train_model(model, features, edge_index_local, edge_index_context,
                targets_cls, train_mask, val_mask, cfg, device, checkpoint_dir):
    print("=" * 60)
    print("TRAIN: 10-Class GVVI Classification")
    print("=" * 60)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    lr, wd = cfg["training"]["lr"], cfg["training"]["weight_decay"]
    max_epochs = cfg["training"]["max_epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    grad_clip = cfg["training"]["gradient_clip"]
    num_classes = cfg["classification"]["num_classes"]
    ord_w = cfg["classification"]["ordinal_weight"]

    y_cls = torch.tensor(targets_cls, dtype=torch.long).to(device)
    x_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    ei_local = edge_index_local.to(device)
    ei_context = edge_index_context.to(device) if edge_index_context is not None else None
    t_mask = torch.tensor(train_mask, dtype=torch.bool).to(device)
    v_mask = torch.tensor(val_mask, dtype=torch.bool).to(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6)
    loss_fn = OrdinalCrossEntropyLoss(num_classes, ord_w)

    best_val_loss = float("inf"); best_epoch = 0; patience_counter = 0
    ckpt_path = f"{checkpoint_dir}/best_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"Device: {device}, Features: {features.shape[1]}, Classes: {num_classes}")
    print(f"Train: {t_mask.sum().item()}, Val: {v_mask.sum().item()}")
    print(f"Class distribution (train): {np.bincount(targets_cls[train_mask], minlength=num_classes)}")

    for epoch in range(1, max_epochs + 1):
        model.train(); optimizer.zero_grad()
        logits = model(x_tensor, ei_local, ei_context)
        loss = loss_fn(logits[t_mask], y_cls[t_mask])
        if torch.isnan(loss) or torch.isinf(loss):
            patience_counter += 1
            if patience_counter >= patience: break
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(x_tensor, ei_local, ei_context)
            val_loss = F.cross_entropy(v_logits[v_mask], y_cls[v_mask])
            v_pred = v_logits[v_mask].argmax(dim=-1)
            v_acc = (v_pred == y_cls[v_mask]).float().mean()

        if torch.isnan(val_loss): patience_counter += 1
        else:
            scheduler.step(val_loss)
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss; best_epoch = epoch; patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

        if epoch <= 5 or epoch % 50 == 0:
            print(f"Epoch {epoch:4d}: loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, "
                  f"val_acc={v_acc.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at {epoch} (best: {best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt_path))
    print(f"Best model from epoch {best_epoch}\n")
    return model, best_epoch
