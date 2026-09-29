"""Experiment 1 -- does alignment hold?

Build this first; nothing downstream means anything if it fails. Every image
of a SKU is aligned into the canonical frame and compared against the
reference silhouette.

Pass criterion: mean IoU > 0.95 and centroid drift < 5 px on the canvas.

    python -m experiments.exp1_alignment_repeatability --sku 8983864641
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from experiments.common import (gather, golden_paths, load_all, out_dir,
                                print_table, test_paths)
from inspector import viz
from inspector.config import PipelineConfig
from inspector.pipeline import build_reference, load_image, preprocess

PASS_IOU = 0.95
PASS_DRIFT = 5.0


def run(sku: str, cfg: PipelineConfig, include_test: bool = True) -> dict:
    gpaths = golden_paths(sku)
    if not gpaths:
        raise SystemExit(f"no golden images for SKU {sku}")

    ref_stats, ref_img, ref_mask, results = build_reference(
        load_all(gpaths), cfg, [p.name for p in gpaths]
    )

    rows = []
    for r in results:
        rows.append({
            "image": r.source, "set": "golden",
            "iou": r.info.get("iou", float("nan")),
            "drift_px": r.info.get("centroid_drift_px", float("nan")),
            "ecc": r.info.get("ecc_ok", "-"),
            "tilt_px": r.info.get("tilt_px", float("nan")),
            "ok": r.ok,
        })

    if include_test:
        for p in test_paths(sku):
            r = preprocess(load_image(p),
                           cfg, ref_stats=ref_stats, ref_mask=ref_mask,
                           ref_image=ref_img, source=p.name)
            rows.append({
                "image": p.name, "set": "test",
                "iou": r.info.get("iou", float("nan")),
                "drift_px": r.info.get("centroid_drift_px", float("nan")),
                "ecc": r.info.get("ecc_ok", "-"),
                "tilt_px": r.info.get("tilt_px", float("nan")),
                "ok": r.ok,
            })

    return {"rows": rows, "ref_mask": ref_mask, "ref_image": ref_img,
            "results": results, "ref_stats": ref_stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="8983864641")
    ap.add_argument("--segmenter", default="birefnet",
                    choices=["classical", "birefnet", "auto"])
    ap.add_argument("--no-ecc", action="store_true")
    ap.add_argument("--ecc-on", default="mask", choices=["mask", "intensity"])
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.segment.method = args.segmenter
    cfg.align.refine_homography = not args.no_ecc
    cfg.align.ecc_on = args.ecc_on

    print(f"Experiment 1 -- alignment repeatability, SKU {args.sku}")
    print(f"segmenter={args.segmenter}  ecc={'off' if args.no_ecc else args.ecc_on}\n")

    res = run(args.sku, cfg)
    rows = res["rows"]
    print_table(rows, ["image", "set", "iou", "drift_px", "ecc", "ok"],
                {"image": 20, "set": 9, "iou": 9, "drift_px": 11, "ecc": 8})

    # Only images that passed the validity gate count: a part clipped by the
    # frame edge is an invalid photo, and including it would mix a photography
    # failure into an alignment measurement.
    valid = [r for r in rows if r["ok"] and np.isfinite(r["iou"])]
    golden = [r for r in valid if r["set"] == "golden"]

    print()
    for name, subset in (("golden", golden), ("all valid", valid)):
        if not subset:
            continue
        iou = np.array([r["iou"] for r in subset])
        drift = np.array([r["drift_px"] for r in subset])
        print(f"{name:<12} n={len(subset):<3} mean IoU {iou.mean():.4f}  "
              f"min {iou.min():.4f}  mean drift {drift.mean():.2f} px  "
              f"max {drift.max():.2f} px")

    if golden:
        iou = np.array([r["iou"] for r in golden])
        drift = np.array([r["drift_px"] for r in golden])
        ok = iou.mean() > PASS_IOU and drift.max() < PASS_DRIFT
        print(f"\nPass criterion (mean IoU > {PASS_IOU}, max drift < "
              f"{PASS_DRIFT} px): {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("Alignment is the foundation -- fix this before trusting any "
                  "score. Every downstream number is meaningless otherwise.")

    d = out_dir("exp1", args.sku)
    stack = np.mean([(r.aligned_part_mask > 0).astype(np.float32)
                     for r in res["results"]
                     if r.aligned_part_mask is not None], axis=0)
    cv2.imwrite(str(d / "aligned_agreement.png"),
                viz.label(cv2.applyColorMap((stack * 255).astype(np.uint8),
                                            cv2.COLORMAP_VIRIDIS),
                          "aligned mask agreement"))
    cv2.imwrite(str(d / "reference.png"), res["ref_image"])
    cv2.imwrite(str(d / "aligned_grid.png"),
                viz.grid([(r.aligned_image, r.source) for r in res["results"]],
                         cols=4))
    print(f"\nwrote {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
