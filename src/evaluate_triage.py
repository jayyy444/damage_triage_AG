"""
evaluate_triage.py — comprehensive evaluation of triage calibration and classification
on a held-out fold (default: config.TEST_FOLD).

Computes:
  - metrics.json: AUROC, PR-AUC, F1 at best threshold, ECE (10 bins), Brier score
  - confusion_matrix.png
  - reliability_diagram.png
  - triage_rates.json + triage_rates.png: per-level actual damaged rates + monotonic check
  - temperature scaling: fits temperature on VAL fold and evaluates calibration
"""
import os
import json
import bisect
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

import config
from dataset import QQBDataset
import infer

MODALITIES = {
    "all": ["sar", "sarftp", "opt", "optftp"],
    "sar": ["sar", "sarftp"],
    "opt": ["opt", "optftp"],
}


def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []
    n = len(probs)
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= low) & (probs <= high) if i == n_bins - 1 else (probs >= low) & (probs < high)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(labels[mask]))
            bin_conf = float(np.mean(probs[mask]))
            ece += (count / n) * abs(bin_acc - bin_conf)
            bin_data.append({"bin_idx": i, "low": low, "high": high, "count": count,
                             "acc": bin_acc, "conf": bin_conf})
        else:
            bin_data.append({"bin_idx": i, "low": low, "high": high, "count": 0,
                             "acc": None, "conf": None})
    return float(ece), bin_data


def fit_temperature(val_logits, val_labels):
    """Fit a single positive scalar temperature on validation logits via L-BFGS."""
    temp_param = nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([temp_param], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    labels_t = torch.as_tensor(val_labels, dtype=torch.float32)

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits_t / temp_param.clamp(min=0.05), labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return float(temp_param.clamp(min=0.05).detach().item())


def extract_logits_labels(model, fold, mode, data_root=None, device="cpu"):
    data_root = data_root or config.DATA_ROOT
    ds = QQBDataset(data_root, [fold])
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    keys = MODALITIES[mode]
    logits, gts, bids = [], [], []
    with torch.no_grad():
        for images, label, bid in loader:
            batch = {k: images[k].to(device) for k in keys}
            out = model(**batch)
            logits.append(out.cpu().item())
            gts.append(label.item())
            bids.append(bid[0])
    return np.array(logits), np.array(gts), bids


def evaluate(ckpt_path=None, test_fold=None, val_fold=None, out_dir=None, data_root=None):
    test_fold = test_fold or config.TEST_FOLD
    val_fold = val_fold or config.VAL_FOLD
    out_dir = out_dir or os.path.join(config.OUTPUTS_DIR, "eval")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = ckpt_path or infer._default_ckpt(test_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mode = infer.load_model(ckpt_path, device)

    # 1. Test fold predictions (uncalibrated)
    test_logits, test_gts, _ = extract_logits_labels(model, test_fold, mode, data_root, device)
    uncal_probs = 1.0 / (1.0 + np.exp(-test_logits))

    # 2. Temperature scaling using val_fold
    val_logits, val_gts, _ = extract_logits_labels(model, val_fold, mode, data_root, device)
    temp = fit_temperature(val_logits, val_gts)
    cal_probs = 1.0 / (1.0 + np.exp(-(test_logits / temp)))

    # Save temperature into a copy of checkpoint
    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    ckpt_calibrated = dict(ckpt_raw)
    ckpt_calibrated["temperature"] = temp
    cal_ckpt_path = os.path.join(out_dir, f"calibrated_{os.path.basename(ckpt_path)}")
    torch.save(ckpt_calibrated, cal_ckpt_path)

    # Metrics
    has_both = len(set(test_gts.tolist())) > 1
    auroc = float(roc_auc_score(test_gts, uncal_probs)) if has_both else float("nan")
    pr_auc = float(average_precision_score(test_gts, uncal_probs)) if has_both else float("nan")

    if has_both:
        precisions, recalls, thresholds = precision_recall_curve(test_gts, uncal_probs)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = int(np.argmax(f1s))
        best_f1 = float(f1s[best_idx])
        best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    else:
        best_f1 = 0.0
        best_th = 0.5

    ece_uncal, bin_data = compute_ece(uncal_probs, test_gts, n_bins=10)
    ece_cal, _ = compute_ece(cal_probs, test_gts, n_bins=10)
    brier = float(brier_score_loss(test_gts, uncal_probs)) if len(test_gts) > 0 else 0.0

    metrics = {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_threshold": best_th,
        "ece": ece_uncal,
        "ece_calibrated": ece_cal,
        "brier_score": brier,
        "temperature": temp,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    preds_binary = (uncal_probs >= best_th).astype(int)
    cm = confusion_matrix(test_gts, preds_binary, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(config.CLASSES)
    ax.set_yticklabels(config.CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (th={best_th:.2f})")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    confs = [b["conf"] for b in bin_data if b["conf"] is not None]
    accs = [b["acc"] for b in bin_data if b["acc"] is not None]
    if confs:
        ax.plot(confs, accs, "s-", color="crimson", label=f"Uncal ECE={ece_uncal:.3f}")
    _, cal_bin_data = compute_ece(cal_probs, test_gts, n_bins=10)
    cal_confs = [b["conf"] for b in cal_bin_data if b["conf"] is not None]
    cal_accs = [b["acc"] for b in cal_bin_data if b["acc"] is not None]
    if cal_confs:
        ax.plot(cal_confs, cal_accs, "o-", color="navy", label=f"Cal ECE={ece_cal:.3f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of damaged")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("Reliability Diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability_diagram.png"), dpi=150)
    plt.close(fig)

    # Triage rates: per-level actual damaged rates
    level_indices = [bisect.bisect_right(config.TRIAGE_BINS, p) for p in cal_probs]
    level_rates = {}
    valid_rates_for_monotonic = []
    for idx, lvl in enumerate(config.TRIAGE_LEVELS):
        mask = [i for i, li in enumerate(level_indices) if li == idx]
        count = len(mask)
        if count > 0:
            actual_rate = round(float(np.mean(test_gts[mask])), 4)
            valid_rates_for_monotonic.append(actual_rate)
        else:
            actual_rate = None
        level_rates[lvl] = {"count": count, "damaged_rate": actual_rate}

    monotonic = True
    if len(valid_rates_for_monotonic) > 1:
        monotonic = all(valid_rates_for_monotonic[i] <= valid_rates_for_monotonic[i + 1]
                        for i in range(len(valid_rates_for_monotonic) - 1))

    triage_results = {
        "levels": level_rates,
        "monotonic": monotonic,
    }
    with open(os.path.join(out_dir, "triage_rates.json"), "w") as f:
        json.dump(triage_results, f, indent=2)

    # Triage rates bar plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_positions = np.arange(len(config.TRIAGE_LEVELS))
    heights = [level_rates[lvl]["damaged_rate"] if level_rates[lvl]["damaged_rate"] is not None else 0.0
               for lvl in config.TRIAGE_LEVELS]
    colors = ["#2ca02c", "#8c564b", "#ff7f0e", "#d62728"]
    bars = ax.bar(x_positions, heights, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(config.TRIAGE_LEVELS, rotation=15)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Actual Damaged Rate")
    ax.set_title(f"Triage Levels vs Actual Damage Rate (Monotonic: {monotonic})")
    for b, lvl in zip(bars, config.TRIAGE_LEVELS):
        rate = level_rates[lvl]["damaged_rate"]
        cnt = level_rates[lvl]["count"]
        txt = f"Rate: {rate:.2f}\n(N={cnt})" if rate is not None else f"empty\n(N=0)"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, txt,
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "triage_rates.png"), dpi=150)
    plt.close(fig)

    print(f"Evaluation complete. Results saved -> {out_dir}")
    print(f"  AUROC: {auroc:.4f}  PR-AUC: {pr_auc:.4f}  Best F1: {best_f1:.4f}")
    print(f"  ECE before: {ece_uncal:.4f} -> after T={temp:.3f}: {ece_cal:.4f}")
    print(f"  Triage rates monotonic: {monotonic}")
    return metrics, triage_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fold", default=None, help="test fold (default: config.TEST_FOLD)")
    ap.add_argument("--val_fold", default=None, help="val fold for temperature scaling")
    ap.add_argument("--out", default=None, help="output directory (default: outputs/eval/)")
    ap.add_argument("--data_root", default=None)
    args = ap.parse_args()
    evaluate(ckpt_path=args.checkpoint, test_fold=args.fold, val_fold=args.val_fold,
             out_dir=args.out, data_root=args.data_root)
