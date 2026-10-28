"""
tests/smoke_test.py — synthetic smoke test for DamageTriage.

Validates the full pipeline without real data:
- Monkeypatches QQBDataset to yield synthetic tensors for all 4 modalities
- Tests mode in ["all", "sar", "opt"] and fusion in ["concat", "gated"]
- Runs 1 epoch of train, inference, triage-level assignment, aggregate, and evaluate_triage
- Asserts every prediction has triage_level in TRIAGE_LEVELS
- Asserts triage_counts sums to N
- Asserts loading an old-format checkpoint (without "fusion" key) works seamlessly
"""
import os
import sys
import shutil
import tempfile
import torch
import numpy as np

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import config
import dataset
import model as model_mod
import train
import infer
import aggregate
import evaluate_triage
import run_ablation
import run_cv


def setup_fake_environment():
    temp_dir = tempfile.mkdtemp(prefix="damagetriage_smoke_")
    data_dir = os.path.join(temp_dir, "data")
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    out_dir = os.path.join(temp_dir, "outputs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Create synthetic folds (6 buildings per fold, 3 intact and 3 damaged)
    b_id = 1000
    for fold in config.ALL_FOLDS:
        fold_file = os.path.join(data_dir, fold)
        with open(fold_file, "w") as f:
            for i in range(6):
                label = i % 2  # balanced labels
                f.write(f"b_{b_id},{label}\n")
                b_id += 1

    # Override config paths
    config.DATA_ROOT = data_dir
    config.CHECKPOINT_DIR = ckpt_dir
    config.OUTPUTS_DIR = out_dir

    # Monkeypatch QQBDataset.__getitem__ to return synthetic tensors without needing .mat files
    def fake_getitem(self, idx):
        bid = self.ids[idx]
        label = np.float32(self.labels[idx])
        # Deterministic synthetic tensors for each building id
        rng = np.random.RandomState(abs(hash(bid)) % 1000000)
        img_shape = (3, config.IMG_SIZE, config.IMG_SIZE)
        images = {
            "sar": torch.from_numpy(rng.rand(*img_shape).astype(np.float32)),
            "sarftp": torch.from_numpy((rng.rand(*img_shape) > 0.5).astype(np.float32)),
            "opt": torch.from_numpy(rng.rand(*img_shape).astype(np.float32)),
            "optftp": torch.from_numpy((rng.rand(*img_shape) > 0.5).astype(np.float32)),
        }
        return images, label, bid

    dataset.QQBDataset.__getitem__ = fake_getitem
    return temp_dir


def run_smoke_tests():
    temp_dir = setup_fake_environment()
    print(f"Running smoke test in temporary sandbox: {temp_dir}")
    try:
        modes = ["all", "sar", "opt"]
        fusions = ["concat", "gated"]

        total_runs = len(modes) * len(fusions)
        current = 0

        for m in modes:
            for f in fusions:
                current += 1
                print(f"\n==========================================")
                print(f"[{current}/{total_runs}] Testing Mode='{m}' | Fusion='{f}'")
                print(f"==========================================")

                # 1. Train 1 epoch
                ckpt_path, best_auroc = train.main(
                    epochs=1,
                    batch_size=4,
                    lr=1e-3,
                    val_fold="fold-1.txt",
                    test_fold="fold-2.txt",
                    mode=m,
                    fusion=f,
                )
                assert os.path.exists(ckpt_path), f"Checkpoint not saved: {ckpt_path}"
                print(f"[OK] Checkpoint saved: {ckpt_path}")

                # 2. Check metrics JSON was saved
                metrics_json = os.path.join(config.CHECKPOINT_DIR, f"metrics_{m}_{f}.json")
                assert os.path.exists(metrics_json), f"Metrics JSON missing: {metrics_json}"
                print(f"[OK] Metrics file exists: {metrics_json}")

                # 3. Inference
                predictions, review_queue = infer.run_inference(ckpt_path=ckpt_path, fold="fold-2.txt")
                assert len(predictions) > 0, "No predictions returned from inference"
                print(f"[OK] Inference produced {len(predictions)} predictions")

                # 4. Check triage level keys and types
                for p in predictions:
                    assert "triage_level" in p, f"Missing triage_level in {p}"
                    assert p["triage_level"] in config.TRIAGE_LEVELS, f"Invalid triage level: {p['triage_level']}"
                    assert "needs_review" in p, f"Missing needs_review in {p}"
                    assert isinstance(p["needs_review"], bool), f"needs_review should be bool, got {type(p['needs_review'])}"
                    assert "probability_damaged" in p
                    assert "confidence" in p
                    assert "building_id" in p
                print("[OK] All predictions have valid triage_level and schema")

                # 5. Check review queue
                for r in review_queue:
                    assert r["needs_review"] is True
                print(f"[OK] Review queue has {len(review_queue)} flagged buildings")

                # 6. Aggregate
                summary = aggregate.aggregate(predictions)
                assert "triage_counts" in summary, "triage_counts missing in summary"
                assert "review_count" in summary, "review_count missing in summary"

                triage_sum = sum(summary["triage_counts"].values())
                assert triage_sum == len(predictions), (
                    f"Triage counts sum ({triage_sum}) does not equal total predictions ({len(predictions)})"
                )
                for lvl in config.TRIAGE_LEVELS:
                    assert lvl in summary["triage_counts"], f"Level {lvl} missing in triage_counts"
                print(f"[OK] Aggregate verified: triage_counts sum to {triage_sum}")

                # 7. Evaluate Triage (calibration, ECE, temperature scaling, rates)
                eval_out = os.path.join(config.OUTPUTS_DIR, f"eval_{m}_{f}")
                metrics, triage_rates = evaluate_triage.evaluate(
                    ckpt_path=ckpt_path,
                    test_fold="fold-2.txt",
                    val_fold="fold-1.txt",
                    out_dir=eval_out,
                )
                assert os.path.exists(os.path.join(eval_out, "metrics.json")), "metrics.json missing"
                assert os.path.exists(os.path.join(eval_out, "confusion_matrix.png")), "confusion_matrix.png missing"
                assert os.path.exists(os.path.join(eval_out, "reliability_diagram.png")), "reliability_diagram.png missing"
                assert os.path.exists(os.path.join(eval_out, "triage_rates.json")), "triage_rates.json missing"
                assert os.path.exists(os.path.join(eval_out, "triage_rates.png")), "triage_rates.png missing"
                assert "monotonic" in triage_rates, "monotonic check missing in triage_rates"
                print(f"[OK] evaluate_triage artifacts generated in {eval_out}")

        # 8. Test OLD-format checkpoint loading (state_dict and mode only, no "fusion" or "temperature" keys)
        print("\n==========================================")
        print("Testing backwards compatibility with OLD-format checkpoint...")
        print("==========================================")
        old_model = model_mod.MultiModalDamageClassifier(mode="all", fusion="concat")
        old_ckpt = {
            "state_dict": old_model.state_dict(),
            "mode": "all"
        }
        old_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "legacy_model_fold-1.pt")
        torch.save(old_ckpt, old_ckpt_path)

        # Load via infer.load_model
        loaded_model, loaded_mode = infer.load_model(old_ckpt_path, device="cpu")
        assert loaded_mode == "all"
        assert loaded_model.fusion == "concat"
        assert getattr(loaded_model, "temperature", None) is None
        print("[OK] infer.load_model successfully loaded legacy checkpoint")

        # Run inference using legacy checkpoint
        old_preds, old_review = infer.run_inference(ckpt_path=old_ckpt_path, fold="fold-2.txt")
        assert len(old_preds) > 0
        for p in old_preds:
            assert p["triage_level"] in config.TRIAGE_LEVELS
        print("[OK] Inference with legacy checkpoint executed successfully")

        # 9. Test run_ablation and run_cv programmatically
        print("\n==========================================")
        print("Testing run_ablation programmatically (1 epoch)...")
        print("==========================================")
        ablation_df = run_ablation.run_ablation(
            epochs=1,
            fusion="gated",
            val_fold="fold-1.txt",
            test_fold="fold-2.txt",
            out_dir=os.path.join(config.OUTPUTS_DIR, "eval_ablation"),
        )
        assert len(ablation_df) == 4, f"Expected 4 ablation rows, got {len(ablation_df)}"
        print("[OK] run_ablation executed and returned expected dataframe")

        print("\n==========================================")
        print("Testing run_cv programmatically (1 epoch)...")
        print("==========================================")
        cv_df = run_cv.run_cv(
            epochs=1,
            mode="sar",
            fusion="concat",
            out_dir=os.path.join(config.OUTPUTS_DIR, "eval_cv"),
        )
        assert len(cv_df) == 5, f"Expected 5 CV rows, got {len(cv_df)}"
        print("[OK] run_cv executed and returned expected dataframe")

        print("\n" + "*" * 60)
        print("ALL SMOKE TESTS PASSED SUCCESSFULLY!")
        print("*" * 60 + "\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_smoke_tests()
