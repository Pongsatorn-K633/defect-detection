"""Experiment 4 -- specular sensitivity.

The expected dominant failure mode. Bare and painted steel both throw moving
specular hotspots that can out-score a real defect, so a set of GOOD parts
photographed with the light and camera deliberately moved is scored against
the subspace and the false-fail count reported.

Put those photos in data/specular/<sku>/. They must all be good parts; the
whole point is that every fail here is a false one.

    python -m experiments.exp4_specular_sensitivity --sku 8983864641 --threshold 6
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.common import (gather, golden_paths, load_all, out_dir,
                                print_table)
from inspector.config import DATA_DIR, PipelineConfig
from inspector.inspect import inspect_image
from inspector.pipeline import load_image
from inspector.register import register_sku
from inspector.roi import SKUConfig


def specular_paths(sku: str):
    return gather(DATA_DIR / "specular" / sku)


def brightness_stats(image) -> dict:
    """Where and how strong the highlights are in this shot."""
    import cv2

    v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    hot = v > 240
    ys, xs = np.nonzero(hot)
    return {
        "clipped_pct": 100.0 * hot.mean(),
        "hot_centroid_x": float(xs.mean()) if len(xs) else float("nan"),
        "hot_centroid_y": float(ys.mean()) if len(ys) else float("nan"),
        "mean_v": float(v.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="8983864641")
    ap.add_argument("--segmenter", default="birefnet")
    ap.add_argument("--threshold", type=float, default=None,
                    help="threshold from Experiment 3; defaults to the SKU's")
    ap.add_argument("--position-norm", dest="posnorm", default=None,
                    choices=["on", "off"])
    args = ap.parse_args()

    spaths = specular_paths(args.sku)
    if not spaths:
        print(f"Experiment 4 needs good-part photos with varied lighting in "
              f"{DATA_DIR / 'specular' / args.sku}")
        print("Photograph one good part ~20x, deliberately moving the light "
              "or camera so highlights land in different places.")
        print("Every fail on that set is a false fail -- that is the number "
              "this experiment produces.")
        return 1

    cfg = PipelineConfig()
    cfg.segment.method = args.segmenter
    if args.posnorm:
        cfg.subspace.position_normalize = args.posnorm == "on"

    gpaths = golden_paths(args.sku)
    try:
        sku_cfg = SKUConfig.load(args.sku)
    except FileNotFoundError:
        sku_cfg = SKUConfig(sku=args.sku)
    if args.threshold is not None:
        sku_cfg.threshold = args.threshold

    print(f"Experiment 4 -- specular sensitivity, SKU {args.sku}")
    print(f"{len(spaths)} good-part shots, threshold {sku_cfg.threshold}, "
          f"position normalisation "
          f"{'on' if cfg.subspace.position_normalize else 'off'}\n")

    reg = register_sku(args.sku, load_all(gpaths), cfg,
                       [p.name for p in gpaths], sku_cfg.rois)
    reg["config"] = cfg

    rows = []
    for p in spaths:
        img = load_image(p)
        res = inspect_image(img, reg, sku_cfg, cfg, source=p.name)
        b = brightness_stats(img)
        rows.append({
            "image": p.name,
            "verdict": res.verdict,
            "max_score": res.max_score,
            "clipped_pct": b["clipped_pct"],
            "failed": ", ".join(r.id for r in res.failed_rois()) or "-",
        })

    print_table(rows, ["image", "verdict", "max_score", "clipped_pct",
                       "failed"],
                {"image": 20, "verdict": 10, "max_score": 12,
                 "clipped_pct": 13, "failed": 24})

    scored = [r for r in rows if r["verdict"] != "INVALID"]
    false_fail = [r for r in scored if r["verdict"] == "FAIL"]
    invalid = [r for r in rows if r["verdict"] == "INVALID"]

    print(f"\n{len(false_fail)} / {len(scored)} good parts falsely failed "
          f"({100 * len(false_fail) / max(len(scored), 1):.1f}%)")
    if invalid:
        print(f"{len(invalid)} rejected as invalid photos (not counted as "
              f"false fails)")

    if scored:
        s = np.array([r["max_score"] for r in scored], float)
        c = np.array([r["clipped_pct"] for r in scored], float)
        if np.isfinite(s).sum() > 2 and c.std() > 0:
            corr = float(np.corrcoef(s[np.isfinite(s)], c[np.isfinite(s)])[0, 1])
            print(f"correlation between clipped-highlight area and score: "
                  f"{corr:+.2f}")
            if corr > 0.5:
                print("Score tracks highlight area -- specular reflection is "
                      "driving the failures, as expected.")

    if false_fail:
        print("\nFixes in priority order (per the spec's risk list):")
        print("  1. more golden images covering highlight positions")
        print("  2. per-position normalisation (--position-norm on)")
        print("  3. a polarising filter on the lens")
        print("  4. a diffused light source at the station")

    d = out_dir("exp4", args.sku)
    import csv
    with open(d / "specular.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {d / 'specular.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
