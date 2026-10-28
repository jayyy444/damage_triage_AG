"""
One-command pipeline: build manifest -> train -> infer -> Grad-CAM
heatmaps (SAR + optical) for top-priority buildings -> aggregate -> PDF
report.

Usage:
    python src/run_pipeline.py --data_root /path/to/extracted/qqb_dataset

Re-running is safe/cheap: it skips training if a checkpoint already
exists for this mode+val_fold. Use --force_train to redo it anyway.
"""
import os
import json
import argparse

import config
import dataset
import train
import infer
import explain
import aggregate as aggregate_mod
import report as report_mod


def main(data_root, force_train=False, epochs=30, mode="all", fusion=None):
    config.DATA_ROOT = data_root  # allow override without editing config.py
    fusion = fusion or getattr(config, "FUSION", "concat")

    manifest_path = os.path.join(data_root, "labels.json")
    if not os.path.exists(manifest_path):
        print("== Step 1/5: build manifest ==")
        dataset.build_manifest(data_root, out_path=manifest_path)
    else:
        print("== Step 1/5: build manifest == (labels.json exists, skipping)")

    fusion_suffix = f"_{fusion}" if fusion != "concat" else ""
    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"model_{mode}_{config.VAL_FOLD.replace('.txt','')}{fusion_suffix}.pt"
    )
    if force_train or not os.path.exists(ckpt_path):
        print("== Step 2/5: train ==")
        ckpt_path, _ = train.main(epochs=epochs, mode=mode, val_fold=config.VAL_FOLD,
                                  test_fold=config.TEST_FOLD, fusion=fusion)
    else:
        print("== Step 2/5: train == (checkpoint exists, skipping — use --force_train to redo)")

    print("== Step 3/5: infer ==")
    predictions, review_queue = infer.run_inference(ckpt_path=ckpt_path, fold=config.TEST_FOLD)

    print("== Step 4/5: grad-cam heatmaps ==")
    top = sorted([p for p in predictions if p["predicted_label"] == "damaged"],
                 key=lambda p: p["probability_damaged"], reverse=True)[:20]
    explain.generate_heatmaps_for([p["building_id"] for p in top], ckpt_path, fold_root=data_root)

    print("== Step 5/5: aggregate + report ==")
    summary = aggregate_mod.aggregate(predictions)
    with open(os.path.join(config.OUTPUTS_DIR, "triage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    report_mod.generate_report(predictions, review_queue, summary,
                                heatmap_dir=os.path.join(config.OUTPUTS_DIR, "heatmaps"))

    print("\nTriage breakdown:")
    for lvl, cnt in summary.get("triage_counts", {}).items():
        print(f"  {lvl}: {cnt}")
    print(f"  flagged for review: {summary.get('review_count', 0)}")

    print("\nDone. Check outputs/ for predictions.json, review_queue.json, "
          "triage_summary.json, heatmaps/, reports/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="path to extracted QQB dataset (contains *_SAR.mat, fold-*.txt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--mode", default="all", choices=["all", "sar", "opt"])
    ap.add_argument("--fusion", default=None, choices=["concat", "gated"])
    ap.add_argument("--force_train", action="store_true")
    args = ap.parse_args()
    main(args.data_root, args.force_train, args.epochs, args.mode, fusion=args.fusion)
