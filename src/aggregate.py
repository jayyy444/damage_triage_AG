"""
aggregate.py — summary statistics across all predictions.

Produces overall + triage level breakdown + top-priority severity/confidence
summary for disaster response reporting.
"""
import os
import json
import bisect

import config


def aggregate(predictions):
    total = len(predictions)
    damaged = [p for p in predictions if p["predicted_label"] == "damaged"]
    review = [p for p in predictions if abs(p["probability_damaged"] - 0.5) < config.REVIEW_MARGIN]

    triage_counts = {lvl: 0 for lvl in config.TRIAGE_LEVELS}
    review_count = 0
    for p in predictions:
        lvl = p.get("triage_level")
        if lvl is None and "probability_damaged" in p:
            idx = bisect.bisect_right(config.TRIAGE_BINS, p["probability_damaged"])
            lvl = config.TRIAGE_LEVELS[idx]
        if lvl in triage_counts:
            triage_counts[lvl] += 1
        if p.get("needs_review", lvl in config.REVIEW_LEVELS):
            review_count += 1

    summary = {
        "total_buildings": total,
        "predicted_damaged": len(damaged),
        "predicted_intact": total - len(damaged),
        "damaged_pct": round(100 * len(damaged) / max(1, total), 1),
        "flagged_for_review": len(review),
        "review_count": review_count,
        "triage_counts": triage_counts,
        "avg_confidence": round(sum(p["confidence"] for p in predictions) / max(1, total), 4) if total > 0 else 0.0,
        "top_priority_building_ids": [
            p["building_id"] for p in
            sorted(damaged, key=lambda p: p["probability_damaged"], reverse=True)[:20]
        ],
    }
    return summary


if __name__ == "__main__":
    with open(os.path.join(config.OUTPUTS_DIR, "predictions.json")) as f:
        predictions = json.load(f)
    summary = aggregate(predictions)
    out_path = os.path.join(config.OUTPUTS_DIR, "triage_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {out_path}")
