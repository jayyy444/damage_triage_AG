"""
train.py — trains MultiModalDamageClassifier using 5-fold cross-validation
splits (fold-1..5.txt, as shipped by QQB). VAL_FOLD is held out for validation /
early stopping, TEST_FOLD is held out for final reporting, and the remaining
folds are used for training. Class imbalance (damaged is the rare class)
is handled with a weighted random sampler.
"""
import os
import time
import json
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

import config
from dataset import QQBDataset, Augment
from model import MultiModalDamageClassifier

MODALITIES = {
    "all": ["sar", "sarftp", "opt", "optftp"],
    "sar": ["sar", "sarftp"],
    "opt": ["opt", "optftp"],
}


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification with logits."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=0.0), self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def best_f1(probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    idx = int(np.argmax(f1s))
    return float(f1s[idx]), float(precision[idx]), float(recall[idx])


def main(epochs=30, batch_size=16, lr=1e-4, val_fold=None, mode=None,
         test_fold=None, fusion=None, loss=None, **kwargs):
    val_fold = val_fold or config.VAL_FOLD
    test_fold = test_fold or getattr(config, "TEST_FOLD", "fold-2.txt")
    mode = mode or config.MODE
    fusion = fusion or getattr(config, "FUSION", "concat")
    loss_choice = loss or kwargs.get("loss_type") or getattr(config, "LOSS", "bce")

    keys = MODALITIES[mode]
    train_folds = [f for f in config.ALL_FOLDS if f != val_fold and f != test_fold]
    if not train_folds:
        train_folds = [f for f in config.ALL_FOLDS if f != val_fold]

    train_ds = QQBDataset(config.DATA_ROOT, train_folds)
    val_ds = QQBDataset(config.DATA_ROOT, [val_fold])
    now_start = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now_start}] train: {len(train_ds)}  val: {len(val_ds)}  test: {test_fold}  "
          f"mode: {mode}  fusion: {fusion}  loss: {loss_choice}", flush=True)

    labels = np.array(train_ds.labels) if len(train_ds.labels) > 0 else np.array([0, 1])
    class_counts = np.array([max(1, (labels == c).sum()) for c in (0, 1)])
    weight_per_class = 1.0 / class_counts
    sample_weights = weight_per_class[labels] if len(train_ds.labels) > 0 else np.array([1.0])
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sar_pretrain = config.SAR_PRETRAIN if os.path.exists(config.SAR_PRETRAIN) else None
    if mode in ("all", "sar") and sar_pretrain is None:
        print("note: no SAR-HUB weights found at", config.SAR_PRETRAIN,
              "-> SAR branch trains from random init (see README to download them)")
    model = MultiModalDamageClassifier(mode=mode, sar_pretrain=sar_pretrain, fusion=fusion).to(device)
    aug = Augment().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if loss_choice == "focal":
        criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    fusion_suffix = f"_{fusion}" if fusion != "concat" else ""
    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"model_{mode}_{val_fold.replace('.txt','')}{fusion_suffix}.pt"
    )

    best_auroc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_samples = max(1, len(train_ds))
        for images, label, _ in train_loader:
            raw = {k: images[k].to(device) for k in keys}
            augmented = aug(*[raw[k] for k in keys])
            batch = dict(zip(keys, augmented))
            label = label.to(device)

            opt.zero_grad()
            logits = model(**batch)
            loss_val = criterion(logits, label)
            loss_val.backward()
            opt.step()
            total_loss += loss_val.item() * label.size(0)
        sched.step()
        train_loss = total_loss / n_samples

        model.eval()
        probs, gts = [], []
        with torch.no_grad():
            for images, label, _ in val_loader:
                batch = {k: images[k].to(device) for k in keys}
                logits = model(**batch)
                probs.append(torch.sigmoid(logits).cpu().item())
                gts.append(label.item())
        probs, gts = np.array(probs), np.array(gts)
        auroc = roc_auc_score(gts, probs) if len(set(gts.tolist())) > 1 else float("nan")
        f1, prec, rec = best_f1(probs, gts) if len(probs) > 0 else (0.0, 0.0, 0.0)

        elapsed = time.time() - t0
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now_str}] epoch {epoch:02d}  train_loss {train_loss:.4f}  val_auroc {auroc:.4f}  "
              f"best_f1 {f1:.4f}  precision {prec:.4f}  recall {rec:.4f}  ({elapsed:.1f}s)", flush=True)

        if not np.isnan(auroc) and auroc >= best_auroc:
            best_auroc = auroc
            torch.save({"state_dict": model.state_dict(), "mode": mode, "fusion": fusion}, ckpt_path)

    if not os.path.exists(ckpt_path):
        torch.save({"state_dict": model.state_dict(), "mode": mode, "fusion": fusion}, ckpt_path)

    print(f"Best val AUROC {best_auroc:.4f} -> checkpoint saved to {ckpt_path}")

    # Evaluate best checkpoint ONCE on test_fold
    saved_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(saved_ckpt["state_dict"])
    model.eval()

    test_ds = QQBDataset(config.DATA_ROOT, [test_fold])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    test_probs, test_gts = [], []
    with torch.no_grad():
        for images, label, _ in test_loader:
            batch = {k: images[k].to(device) for k in keys}
            logits = model(**batch)
            test_probs.append(torch.sigmoid(logits).cpu().item())
            test_gts.append(label.item())

    test_probs, test_gts = np.array(test_probs), np.array(test_gts)
    if len(set(test_gts.tolist())) > 1:
        test_auroc = float(roc_auc_score(test_gts, test_probs))
        test_pr_auc = float(average_precision_score(test_gts, test_probs))
        test_f1_val, test_prec, test_rec = best_f1(test_probs, test_gts)
    else:
        test_auroc = float("nan")
        test_pr_auc = float("nan")
        test_f1_val, test_prec, test_rec = 0.0, 0.0, 0.0

    print(f"Test Fold ({test_fold}) -> AUROC: {test_auroc:.4f}, PR-AUC: {test_pr_auc:.4f}, F1: {test_f1_val:.4f}")

    metrics_path = os.path.join(config.CHECKPOINT_DIR, f"metrics_{mode}_{fusion}.json")
    metrics_data = {
        "mode": mode,
        "fusion": fusion,
        "loss": loss_choice,
        "val_fold": val_fold,
        "test_fold": test_fold,
        "best_val_auroc": float(best_auroc),
        "test_auroc": test_auroc,
        "test_pr_auc": test_pr_auc,
        "test_f1": test_f1_val,
        "test_precision": test_prec,
        "test_recall": test_rec,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    return ckpt_path, best_auroc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_fold", default=None, help="e.g. fold-1.txt (default: config.VAL_FOLD)")
    ap.add_argument("--test_fold", default=None, help="e.g. fold-2.txt (default: config.TEST_FOLD)")
    ap.add_argument("--mode", default=None, choices=["all", "sar", "opt"])
    ap.add_argument("--fusion", default=None, choices=["concat", "gated"])
    ap.add_argument("--loss", default=None, choices=["bce", "focal"])
    args = ap.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.val_fold, args.mode,
         test_fold=args.test_fold, fusion=args.fusion, loss=args.loss)
