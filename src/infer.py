"""
infer.py — runs the trained model over a fold (default: config.TEST_FOLD)
and writes outputs/predictions.json + outputs/review_queue.json, mirroring
the interface of the original mexico-earthquake build's infer.py so the
rest of the pipeline (gradcam/aggregate/report) doesn't need to change.
"""
import os
import json
import bisect
import argparse

import torch
from torch.utils.data import DataLoader

import config
from dataset import QQBDataset
from model import MultiModalDamageClassifier

MODALITIES = {
    "all": ["sar", "sarftp", "opt", "optftp"],
    "sar": ["sar", "sarftp"],
    "opt": ["opt", "optftp"],
}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    fusion = ckpt.get("fusion", "concat")
    model = MultiModalDamageClassifier(mode=ckpt["mode"], fusion=fusion).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.temperature = ckpt.get("temperature", None)
    model.eval()
    return model, ckpt["mode"]


def _default_ckpt(fold=None, mode=None, fusion=None):
    mode = mode or config.MODE
    fusion = fusion or getattr(config, "FUSION", "concat")
    suffix = f"_{fusion}" if fusion != "concat" else ""
    target_val = os.path.join(config.CHECKPOINT_DIR, f"model_{mode}_{config.VAL_FOLD.replace('.txt','')}{suffix}.pt")
    if os.path.exists(target_val):
        return target_val
    target_fold = os.path.join(
        config.CHECKPOINT_DIR,
        f"model_{mode}_{(fold or config.VAL_FOLD).replace('.txt','')}{suffix}.pt"
    )
    if os.path.exists(target_fold):
        return target_fold
    return target_val


def run_inference(ckpt_path=None, fold=None):
    fold = fold or config.TEST_FOLD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path or _default_ckpt(fold)
    model, mode = load_model(ckpt_path, device)
    keys = MODALITIES[mode]

    ds = QQBDataset(config.DATA_ROOT, [fold])
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    predictions, review_queue = [], []
    with torch.no_grad():
        for images, label, bid in loader:
            batch = {k: images[k].to(device) for k in keys}
            logits = model(**batch)
            if hasattr(model, "temperature") and model.temperature is not None:
                logits = logits / model.temperature
            prob = torch.sigmoid(logits).item()
            pred_label = config.CLASSES[int(prob >= 0.5)]

            level_idx = bisect.bisect_right(config.TRIAGE_BINS, prob)
            triage_level = config.TRIAGE_LEVELS[level_idx]
            needs_review = triage_level in config.REVIEW_LEVELS

            entry = {
                "building_id": bid[0],
                "true_label": config.CLASSES[int(label.item())],
                "predicted_label": pred_label,
                "probability_damaged": round(prob, 4),
                "confidence": round(max(prob, 1 - prob), 4),
                "triage_level": triage_level,
                "needs_review": needs_review,
            }
            predictions.append(entry)
            if needs_review:
                review_queue.append(entry)

    review_queue.sort(key=lambda p: abs(p["probability_damaged"] - 0.5))

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUTS_DIR, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    with open(os.path.join(config.OUTPUTS_DIR, "review_queue.json"), "w") as f:
        json.dump(review_queue, f, indent=2)

    print(f"{len(predictions)} predictions -> outputs/predictions.json")
    print(f"{len(review_queue)} flagged for review -> outputs/review_queue.json")
    return predictions, review_queue


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fold", default=None)
    args = ap.parse_args()
    run_inference(args.checkpoint, args.fold)
