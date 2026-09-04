"""Experiment 2 -- does the ROI crop actually help?

The go/no-go for the whole approach. Three scoring configurations on the same
images:

  1. full frame, no segmentation      -- the naive baseline
  2. segmented + aligned, whole part
  3. segmented + aligned + ROI cropped

Reports AUROC and good/defective score separation for each, and -- because a
missing tab is exactly the "logical anomaly" SubspaceAD is weak on -- reports
each ROI separately rather than only in aggregate.

Needs labels at data/labels/<sku>.csv (image,label; 1 = defective). Without
them it prints the score distributions and stops short of inventing accuracy
numbers.

    python -m experiments.exp2_roi_vs_fullframe --sku 8983864641
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.common import (golden_paths, load_all, load_labels, out_dir,
                                print_table, roc_auc, separation, test_paths)
from inspector import features as feat_mod
from inspector import subspace as sub_mod
from inspector.config import PipelineConfig
from inspector.inspect import inspect_image
from inspector.pipeline import load_image
from inspector.register import register_sku
from inspector.roi import SKUConfig, patch_grid_box


def _full_frame_scores(gimgs, timgs, cfg):
    """Configuration 1: no segmentation, no alignment, every patch scored.

    This is the baseline the method has to beat -- if it does not, the
    segmentation and alignment machinery is not earning its complexity.
    """
    gfeat = [feat_mod.extract_patch_features(im, cfg.features) for im in gimgs]
    stacked = np.concatenate([f.reshape(-1, f.shape[-1]) for f in gfeat])
    model = sub_mod.fit_subspace(stacked, cfg.subspace.tau,
                                 cfg.subspace.max_components)
    out = []
    for im in timgs:
        f = feat_mod.extract_patch_features(im, cfg.features)
        s = sub_mod.residual_score(f, model)
        out.append(sub_mod.aggregate(s, np.ones(s.shape, bool),
                                     cfg.subspace.aggregation))
    return out, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="8983864641")
    ap.add_argument("--segmenter", default="birefnet")
    ap.add_argument("--backbone", default="vitl14")
    ap.add_argument("--input-size", type=int, default=672)
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.segment.method = args.segmenter
    cfg.features.backbone = args.backbone
    cfg.features.input_size = args.input_size

    gpaths, tpaths = golden_paths(args.sku), test_paths(args.sku)
    if not gpaths or not tpaths:
        raise SystemExit("need both golden and test images")
    labels = load_labels(args.sku)

    print(f"Experiment 2 -- ROI vs whole part vs full frame, SKU {args.sku}")
    print(f"{len(gpaths)} golden, {len(tpaths)} test, "
          f"{len(labels)} labelled\n")

    gimgs, timgs = load_all(gpaths), load_all(tpaths)

    try:
        sku_cfg = SKUConfig.load(args.sku)
    except FileNotFoundError:
        sku_cfg = SKUConfig(sku=args.sku)

    reg = register_sku(args.sku, gimgs, cfg, [p.name for p in gpaths],
                       sku_cfg.rois)
    reg["config"] = cfg

    # Configuration 3 (and 2, via the whole-part ROI): the real pipeline.
    per_image = {}
    for path, img in zip(tpaths, timgs):
        res = inspect_image(img, reg, sku_cfg, cfg, source=path.name)
        entry = {"verdict": res.verdict, "reason": res.reason}
        for r in res.rois:
            entry[r.id] = r.score
        # Whole part with segmentation + alignment = configuration 2.
        if res.score_map is not None and res.valid_grid is not None:
            entry["whole_part"] = sub_mod.aggregate(
                res.score_map, res.valid_grid, cfg.subspace.aggregation)
        per_image[path.name] = entry

    print("\n--- configuration 1: full frame, no segmentation ---")
    full, _ = _full_frame_scores(gimgs, timgs, cfg)
    for p, s in zip(tpaths, full):
        per_image.setdefault(p.name, {})["full_frame"] = s

    rows = []
    for p in tpaths:
        e = per_image.get(p.name, {})
        row = {"image": p.name, "label": labels.get(p.name, "?"),
               "verdict": e.get("verdict", "-")}
        row.update({k: v for k, v in e.items()
                    if k not in ("verdict", "reason")})
        rows.append(row)

    score_cols = sorted({k for r in rows for k in r
                         if k not in ("image", "label", "verdict")})
    print("\n--- per-image scores ---")
    print_table(rows, ["image", "label", "verdict"] + score_cols,
                {"image": 16, "label": 7, "verdict": 10})

    if not labels:
        print("\nNo labels found at data/labels/%s.csv -- reporting scores "
              "only." % args.sku)
        print("Add an `image,label` CSV (1 = defective) to get AUROC and the "
              "go/no-go separation numbers this experiment exists to produce.")
        return 0

    y = np.array([labels.get(p.name, -1) for p in tpaths])
    keep = y >= 0
    print("\n--- configuration comparison (labelled images only) ---")
    comp = []
    for col in score_cols:
        s = np.array([per_image.get(p.name, {}).get(col, np.nan)
                      for p in tpaths], float)
        comp.append({"scoring": col,
                     "auroc": roc_auc(s[keep], y[keep]),
                     "separation": separation(s[keep], y[keep]),
                     "n": int(np.isfinite(s[keep]).sum())})
    comp.sort(key=lambda r: (-r["auroc"] if np.isfinite(r["auroc"]) else 0))
    print_table(comp, ["scoring", "auroc", "separation", "n"],
                {"scoring": 24, "auroc": 10, "separation": 12, "n": 5})

    best = comp[0] if comp else None
    if best and np.isfinite(best["auroc"]):
        print(f"\nBest configuration: {best['scoring']} "
              f"(AUROC {best['auroc']:.3f})")
        roi_best = [c for c in comp if c["scoring"] not in
                    ("full_frame", "whole_part")]
        base = [c for c in comp if c["scoring"] == "full_frame"]
        if roi_best and base and np.isfinite(base[0]["auroc"]):
            gain = roi_best[0]["auroc"] - base[0]["auroc"]
            print(f"ROI-cropped beats the naive full-frame baseline by "
                  f"{gain:+.3f} AUROC.")
            if gain <= 0:
                print("ROI cropping is NOT separating the defect. Per the "
                      "spec this is the go/no-go: the method may be wrong "
                      "for this defect class and classical CV should be "
                      "considered for these checks.")

    d = out_dir("exp2", args.sku)
    import csv
    with open(d / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        keys = ["image", "label", "verdict"] + score_cols
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {d / 'scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
