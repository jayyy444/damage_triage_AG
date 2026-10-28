"""
run_cv.py — 5-fold cross-validation runner.
For each of the 5 folds as val (and the next fold as test), trains and evaluates,
then writes outputs/eval/cv_results.csv and prints mean ± std for AUROC, PR-AUC, F1.
"""
import os
import argparse
import pandas as pd
import numpy as np

import config
import train


def run_cv(epochs=30, mode=None, fusion=None, out_dir=None):
    mode = mode or config.MODE
    fusion = fusion or getattr(config, "FUSION", "concat")
    out_dir = out_dir or os.path.join(config.OUTPUTS_DIR, "eval")
    os.makedirs(out_dir, exist_ok=True)

    folds = config.ALL_FOLDS
    n_folds = len(folds)
    results = []

    print(f"Starting 5-fold CV (mode={mode}, fusion={fusion}, epochs={epochs})...")
    for i in range(n_folds):
        val_fold = folds[i]
        test_fold = folds[(i + 1) % n_folds]
        print(f"\n--- CV Fold {i + 1}/{n_folds}: Val={val_fold}, Test={test_fold} ---")

        ckpt_path, _ = train.main(
            epochs=epochs,
            mode=mode,
            val_fold=val_fold,
            test_fold=test_fold,
            fusion=fusion,
        )

        metrics_path = os.path.join(config.CHECKPOINT_DIR, f"metrics_{mode}_{fusion}.json")
        import json
        with open(metrics_path) as f:
            m = json.load(f)

        row = {
            "fold_idx": i + 1,
            "val_fold": val_fold,
            "test_fold": test_fold,
            "auroc": round(float(m.get("test_auroc", 0.0)), 4),
            "pr_auc": round(float(m.get("test_pr_auc", 0.0)), 4),
            "f1": round(float(m.get("test_f1", 0.0)), 4),
            "precision": round(float(m.get("test_precision", 0.0)), 4),
            "recall": round(float(m.get("test_recall", 0.0)), 4),
        }
        results.append(row)
        print(f"Fold {i + 1} test metrics: AUROC={row['auroc']}, PR-AUC={row['pr_auc']}, F1={row['f1']}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "cv_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCV results saved -> {csv_path}")

    # Summary statistics
    aurocs = df["auroc"].dropna()
    pr_aucs = df["pr_auc"].dropna()
    f1s = df["f1"].dropna()

    print("\n" + "=" * 50)
    print(f"Cross-Validation Summary (5 folds, mode={mode}, fusion={fusion}):")
    print(f"  AUROC:  {aurocs.mean():.4f} +/- {aurocs.std():.4f}")
    print(f"  PR-AUC: {pr_aucs.mean():.4f} +/- {pr_aucs.std():.4f}")
    print(f"  F1:     {f1s.mean():.4f} +/- {f1s.std():.4f}")
    print("=" * 50)

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--mode", default=None, choices=["all", "sar", "opt"])
    ap.add_argument("--fusion", default=None, choices=["concat", "gated"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_cv(epochs=args.epochs, mode=args.mode, fusion=args.fusion, out_dir=args.out)
