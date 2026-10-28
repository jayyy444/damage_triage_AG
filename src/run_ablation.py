"""
run_ablation.py — runs modality and fusion ablations:
sar, opt, all (and optionally --fusion gated) with a shared fold setup,
writes outputs/eval/ablation.csv and prints a markdown table to stdout.
"""
import os
import json
import argparse
import pandas as pd

import config
import train


def format_markdown_table(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run_ablation(epochs=30, fusion="concat", val_fold=None, test_fold=None, out_dir=None):
    val_fold = val_fold or config.VAL_FOLD
    test_fold = test_fold or config.TEST_FOLD
    out_dir = out_dir or os.path.join(config.OUTPUTS_DIR, "eval")
    os.makedirs(out_dir, exist_ok=True)

    experiments = [
        ("sar", "concat"),
        ("opt", "concat"),
        ("all", "concat"),
    ]
    if fusion == "gated":
        experiments.append(("all", "gated"))

    results = []
    print(f"Starting ablation study (val={val_fold}, test={test_fold}, epochs={epochs})...")
    for mode, fus in experiments:
        print(f"\n--- Running ablation: mode={mode}, fusion={fus} ---")
        train.main(
            epochs=epochs,
            mode=mode,
            fusion=fus,
            val_fold=val_fold,
            test_fold=test_fold,
        )
        metrics_path = os.path.join(config.CHECKPOINT_DIR, f"metrics_{mode}_{fus}.json")
        with open(metrics_path) as f:
            m = json.load(f)

        results.append({
            "mode": mode,
            "fusion": fus,
            "auroc": round(float(m.get("test_auroc", 0.0)), 4),
            "pr_auc": round(float(m.get("test_pr_auc", 0.0)), 4),
            "f1": round(float(m.get("test_f1", 0.0)), 4),
            "precision": round(float(m.get("test_precision", 0.0)), 4),
            "recall": round(float(m.get("test_recall", 0.0)), 4),
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAblation CSV saved -> {csv_path}\n")

    md_table = format_markdown_table(df)
    print("### Ablation Results\n")
    print(md_table)
    print("\n")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--fusion", default="concat", choices=["concat", "gated"])
    ap.add_argument("--val_fold", default=None)
    ap.add_argument("--test_fold", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_ablation(
        epochs=args.epochs,
        fusion=args.fusion,
        val_fold=args.val_fold,
        test_fold=args.test_fold,
        out_dir=args.out,
    )
