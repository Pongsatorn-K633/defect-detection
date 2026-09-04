"""Experiment 5 -- how many reference images?

Sweeps k = 1, 2, 4, 8, 16, 32 golden images. For each k it refits and reports
recall AND false-call rate on the same axes, plus the retained component count
r and the fit time.

The expectation worth testing: more golden images widen the subspace, which
helps with lighting and specular variation (fewer false calls) but also lets
the subspace reconstruct subtle defects (lower recall). If so there is an
optimum rather than "more is always better".

With no labels available the script still reports the quantities that do not
need them -- r, fit time, and the golden/test score distributions -- and says
which numbers it is withholding and why.

    python -m experiments.exp5_reference_count_sweep --sku 8983864641
"""
from __future__ import annotations

import argparse
import csv
import random

import numpy as np

from experiments.common import (golden_paths, load_all, load_labels, out_dir,
                                print_table, roc_auc, test_paths)
from inspector import subspace as sub_mod
from inspector.config import PipelineConfig
from inspector.inspect import inspect_image
from inspector.pipeline import load_image
from inspector.register import register_sku
from inspector.roi import SKUConfig


def evaluate(sku, gpaths, gimgs, tpaths, timgs, cfg, sku_cfg, labels, seed=0):
    reg = register_sku(sku, gimgs, cfg, [p.name for p in gpaths], sku_cfg.rois)
    reg["config"] = cfg

    scores, verdicts = [], []
    for p, img in zip(tpaths, timgs):
        res = inspect_image(img, reg, sku_cfg, cfg, source=p.name)
        scores.append(res.max_score)
        verdicts.append(res.verdict)

    row = {
        "k": len(gimgs),
        "r": reg["model"]["r"],
        "n_patches": reg["n_valid_patches"],
        "fit_s": round(reg["fit_seconds"], 1),
        "golden_max": max(reg["golden_scores"].get("whole_part", [np.nan])),
    }

    s = np.array(scores, float)
    valid = np.array([v != "INVALID" for v in verdicts])
    row["test_mean"] = float(np.nanmean(s[valid])) if valid.any() else np.nan

    if labels:
        y = np.array([labels.get(p.name, -1) for p in tpaths])
        keep = valid & (y >= 0)
        if keep.any() and (y[keep] == 1).any() and (y[keep] == 0).any():
            row["auroc"] = roc_auc(s[keep], y[keep])
            thr = sku_cfg.threshold
            flagged = s[keep] > thr
            row["recall"] = float((flagged & (y[keep] == 1)).sum()
                                  / max((y[keep] == 1).sum(), 1))
            row["false_call"] = float((flagged & (y[keep] == 0)).sum()
                                      / max((y[keep] == 0).sum(), 1))
    return row, reg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="8983864641")
    ap.add_argument("--segmenter", default="birefnet")
    ap.add_argument("--ks", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.segment.method = args.segmenter

    gpaths, tpaths = golden_paths(args.sku), test_paths(args.sku)
    if not gpaths:
        raise SystemExit(f"no golden images for SKU {args.sku}")
    labels = load_labels(args.sku)

    try:
        sku_cfg = SKUConfig.load(args.sku)
    except FileNotFoundError:
        sku_cfg = SKUConfig(sku=args.sku)

    gimgs_all = load_all(gpaths)
    timgs = load_all(tpaths)

    ks = [k for k in sorted(set(args.ks)) if k <= len(gpaths)]
    skipped = [k for k in sorted(set(args.ks)) if k > len(gpaths)]

    print(f"Experiment 5 -- reference count sweep, SKU {args.sku}")
    print(f"{len(gpaths)} golden images available, {len(tpaths)} test images, "
          f"{len(labels)} labelled")
    if skipped:
        print(f"skipping k={skipped}: only {len(gpaths)} golden images exist. "
              f"The spec's expected useful range is 8-16, so this sweep "
              f"cannot yet reach the interesting part of the curve.")
    print()

    rng = random.Random(args.seed)
    rows = []
    for k in ks:
        idx = list(range(len(gpaths)))
        rng.shuffle(idx)
        idx = sorted(idx[:k])
        try:
            row, _ = evaluate(args.sku, [gpaths[i] for i in idx],
                              [gimgs_all[i] for i in idx], tpaths, timgs,
                              cfg, sku_cfg, labels)
        except (ValueError, RuntimeError) as exc:
            print(f"k={k}: could not fit ({exc})")
            continue
        rows.append(row)
        print(f"k={k:<3} r={row['r']:<5} fit {row['fit_s']:>6.1f}s  "
              f"golden_max {row['golden_max']:.2f}  "
              f"test_mean {row['test_mean']:.2f}")

    cols = ["k", "r", "n_patches", "fit_s", "golden_max", "test_mean"]
    if labels:
        cols += ["auroc", "recall", "false_call"]
    print("\n--- sweep ---")
    print_table(rows, cols, {c: 12 for c in cols})

    if not labels:
        print("\nNo labels at data/labels/%s.csv, so recall and false-call "
              "rate are not reported." % args.sku)
        print("What the numbers above still show: r grows with k (a wider "
              "subspace), and the gap between golden_max and test_mean is the "
              "separation any threshold has to work with.")

    if labels and len(rows) >= 3 and all("recall" in r for r in rows):
        best = max(rows, key=lambda r: r.get("auroc", 0))
        print(f"\nOptimum on this data: k={best['k']} "
              f"(AUROC {best.get('auroc', float('nan')):.3f})")

    # Coverage vs count: the spec's claim is that spanning lighting and
    # highlight positions matters more than raw image count.
    if len(gpaths) >= 4:
        print("\n--- coverage vs count ---")
        k = min(len(gpaths) // 2, 8)
        reg_all = register_sku(args.sku, gimgs_all, cfg,
                               [p.name for p in gpaths], sku_cfg.rois)
        per_img = []
        for r in reg_all["results"]:
            import cv2
            v = cv2.cvtColor(r.aligned_image, cv2.COLOR_BGR2HSV)[:, :, 2]
            per_img.append(float(v.mean()))
        order = np.argsort(per_img)
        similar = sorted(order[:k])                    # closest in brightness
        varied = sorted(np.linspace(0, len(order) - 1, k).astype(int))
        varied = sorted({int(order[i]) for i in varied})

        for name, idx in (("near-duplicate", similar), ("varied", varied)):
            row, _ = evaluate(args.sku, [gpaths[i] for i in idx],
                              [gimgs_all[i] for i in idx], tpaths, timgs,
                              cfg, sku_cfg, labels)
            row["set"] = name
            print(f"{name:<16} k={row['k']:<3} r={row['r']:<5} "
                  f"golden_max {row['golden_max']:.2f}  "
                  f"test_mean {row['test_mean']:.2f}"
                  + (f"  AUROC {row['auroc']:.3f}" if "auroc" in row else ""))
            rows.append(row)

    d = out_dir("exp5", args.sku)
    with open(d / "sweep.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    _plot(d, [r for r in rows if "set" not in r], bool(labels))
    print(f"\nwrote {d}")
    return 0


def _plot(d, rows, has_labels):
    if len(rows) < 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if has_labels and all("recall" in r for r in rows):
        # Both on the same axes, which is the point of the experiment.
        ax.plot(ks, [r["recall"] for r in rows], "o-", label="recall")
        ax.plot(ks, [r["false_call"] for r in rows], "s-",
                label="false-call rate")
        ax.set_ylabel("rate")
    else:
        ax.plot(ks, [r["test_mean"] for r in rows], "o-", label="test mean score")
        ax.plot(ks, [r["golden_max"] for r in rows], "s-", label="golden max score")
        ax.set_ylabel("score")
    ax.set_xlabel("k (golden images)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ks, [r["r"] for r in rows], "^--", color="grey",
             label="components r")
    ax2.set_ylabel("retained components r")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    ax.set_title("Reference count sweep")
    fig.tight_layout()
    fig.savefig(d / "reference_sweep.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
