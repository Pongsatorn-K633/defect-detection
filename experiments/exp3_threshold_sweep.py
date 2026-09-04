"""Experiment 3 -- threshold sweep.

Recall vs false-call rate for the ROI-cropped configuration, reported per ROI
as well as per part. The headline number is the false-call rate at 99% recall:
in a 100%-inspection manual station, false calls are what actually costs the
operator time.

    python -m experiments.exp3_threshold_sweep --sku 8983864641
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from experiments.common import (golden_paths, load_all, load_labels, out_dir,
                                print_table, test_paths)
from inspector.config import PipelineConfig
from inspector.inspect import inspect_image
from inspector.register import register_sku
from inspector.roi import SKUConfig

TARGET_RECALL = 0.99


def sweep(scores: np.ndarray, labels: np.ndarray, n: int = 200):
    """Recall and false-call rate across candidate thresholds."""
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    if len(scores) == 0 or (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return []
    lo, hi = float(scores.min()), float(scores.max())
    span = max(hi - lo, 1e-6)
    grid = np.linspace(lo - 0.02 * span, hi + 0.02 * span, n)

    rows = []
    for t in grid:
        flagged = scores > t
        tp = int((flagged & (labels == 1)).sum())
        fp = int((flagged & (labels == 0)).sum())
        recall = tp / max((labels == 1).sum(), 1)
        false_call = fp / max((labels == 0).sum(), 1)
        rows.append({"threshold": float(t), "recall": float(recall),
                     "false_call_rate": float(false_call),
                     "tp": tp, "fp": fp})
    return rows


def at_recall(rows, target: float):
    """Lowest false-call rate among thresholds meeting the recall target."""
    ok = [r for r in rows if r["recall"] >= target]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["false_call_rate"], -r["threshold"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="8983864641")
    ap.add_argument("--segmenter", default="birefnet")
    ap.add_argument("--recall", type=float, default=TARGET_RECALL)
    args = ap.parse_args()

    labels = load_labels(args.sku)
    if not labels:
        print(f"Experiment 3 needs labels at data/labels/{args.sku}.csv")
        print("Format: image,label   (1 = defective, 0 = good)")
        print("Without ground truth a recall/false-call curve cannot be "
              "computed, and guessing one would be worse than reporting none.")
        return 1

    cfg = PipelineConfig()
    cfg.segment.method = args.segmenter

    gpaths, tpaths = golden_paths(args.sku), test_paths(args.sku)
    try:
        sku_cfg = SKUConfig.load(args.sku)
    except FileNotFoundError:
        sku_cfg = SKUConfig(sku=args.sku)

    print(f"Experiment 3 -- threshold sweep, SKU {args.sku}\n")
    reg = register_sku(args.sku, load_all(gpaths), cfg,
                       [p.name for p in gpaths], sku_cfg.rois)
    reg["config"] = cfg

    per_roi: dict[str, list[float]] = {}
    part_scores, y, names = [], [], []
    for p in tpaths:
        if p.name not in labels:
            continue
        res = inspect_image(load_image_cached(p), reg, sku_cfg, cfg,
                            source=p.name)
        names.append(p.name)
        y.append(labels[p.name])
        part_scores.append(res.max_score)
        for r in res.rois:
            per_roi.setdefault(r.id, []).append(r.score)

    y = np.array(y, int)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        print("Need both good and defective labelled images to sweep.")
        return 1

    d = out_dir("exp3", args.sku)
    summary = []

    print("--- per-ROI ---")
    for rid, scores in sorted(per_roi.items()):
        rows = sweep(np.array(scores, float), y)
        best = at_recall(rows, args.recall)
        summary.append({
            "scope": rid,
            "threshold": best["threshold"] if best else float("nan"),
            "recall": best["recall"] if best else float("nan"),
            "false_call_rate": best["false_call_rate"] if best else float("nan"),
        })
        with open(d / f"sweep_{rid}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["threshold", "recall",
                                               "false_call_rate", "tp", "fp"])
            w.writeheader()
            w.writerows(rows)

    rows = sweep(np.array(part_scores, float), y)
    best = at_recall(rows, args.recall)
    summary.append({
        "scope": "PART (any ROI)",
        "threshold": best["threshold"] if best else float("nan"),
        "recall": best["recall"] if best else float("nan"),
        "false_call_rate": best["false_call_rate"] if best else float("nan"),
    })
    with open(d / "sweep_part.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["threshold", "recall",
                                           "false_call_rate", "tp", "fp"])
        w.writeheader()
        w.writerows(rows)

    print_table(summary, ["scope", "threshold", "recall", "false_call_rate"],
                {"scope": 22, "threshold": 12, "recall": 10,
                 "false_call_rate": 18})

    part = summary[-1]
    if np.isfinite(part["false_call_rate"]):
        print(f"\nAt {100 * args.recall:.0f}% recall the part-level false-call "
              f"rate is {100 * part['false_call_rate']:.1f}% "
              f"(threshold {part['threshold']:.2f}).")
    else:
        print(f"\nNo threshold reaches {100 * args.recall:.0f}% recall on this "
              f"data -- the distributions overlap.")

    _plot(d, rows)
    print(f"wrote {d}")
    return 0


def load_image_cached(p):
    from inspector.pipeline import load_image
    return load_image(p)


def _plot(d, rows):
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, [r["recall"] for r in rows], label="recall")
    ax.plot(t, [r["false_call_rate"] for r in rows], label="false-call rate")
    ax.set_xlabel("threshold")
    ax.set_ylabel("rate")
    ax.set_title("Part-level threshold sweep")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "threshold_sweep.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
