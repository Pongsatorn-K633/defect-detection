"""Stage 2 -- inspection. Score one part against its SKU's normal subspace.

    python -m inspector.inspect --sku 8983864641 --image test.jpg --out result/
    python -m inspector.inspect --sku 8983864641 --dir test_images/8983864641 --csv r.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import features as feat_mod
from . import geometry as geom_mod
from . import subspace as sub_mod
from . import viz
from .config import OUTPUT_DIR, TEST_DIR, PipelineConfig
from .pipeline import PreprocessResult, load_image, preprocess
from .register import load_registration
from .roi import ROI, SKUConfig, patch_grid_box

log = logging.getLogger(__name__)


@dataclass
class ROIResult:
    id: str
    type: str
    score: float
    threshold: float
    passed: bool
    detail: str = ""
    measured: dict = field(default_factory=dict)


@dataclass
class InspectionResult:
    source: str = ""
    # RETAKE is deliberately distinct from FAIL: the photo is unusable, but
    # nothing has been learned about the part. Reporting it as FAIL would send
    # the operator to scrap a part that was never inspected.
    verdict: str = "PASS"          # PASS | FAIL | RETAKE
    rois: list[ROIResult] = field(default_factory=list)
    score_map: np.ndarray | None = None
    valid_grid: np.ndarray | None = None
    pre: PreprocessResult | None = None
    inference_seconds: float = 0.0
    reason: str = ""
    reason_th: str = ""            # operator-facing wording in Thai

    @property
    def max_score(self) -> float:
        vals = [r.score for r in self.rois if np.isfinite(r.score)]
        return float(max(vals)) if vals else float("nan")

    def failed_rois(self) -> list[ROIResult]:
        return [r for r in self.rois if not r.passed]


def inspect_image(
    image: np.ndarray,
    reg: dict,
    sku_cfg: SKUConfig,
    cfg: PipelineConfig | None = None,
    source: str = "",
) -> InspectionResult:
    """Score one part. Runs the identical preprocessing path as registration."""
    cfg = cfg or reg["config"]
    t0 = time.time()
    out = InspectionResult(source=source)

    pre = preprocess(image, cfg, ref_stats=reg["ref_stats"],
                     ref_mask=reg["ref_mask"], ref_image=reg["ref_image"],
                     source=source)
    out.pre = pre
    if not pre.ok:
        # An unusable photo is not a defective part. Reporting a clipped or
        # unsegmentable frame as FAIL would send the operator to scrap a good
        # part instead of retaking the picture, so it gets its own verdict.
        out.verdict = "RETAKE"
        out.reason = pre.reject_reason
        out.reason_th = pre.reject_reason_th
        out.inference_seconds = time.time() - t0
        return out

    fmap = feat_mod.extract_patch_features(pre.aligned_image, cfg.features)
    grid_h, grid_w = fmap.shape[:2]
    valid = feat_mod.patch_mask_from_pixel_mask(
        pre.aligned_valid_mask, grid_h, grid_w, cfg.features.patch_coverage
    )

    pos_stats = reg.get("position_stats")
    # Position statistics are tied to the grid they were fitted on; using them
    # against a different grid would silently misalign every position.
    if pos_stats and tuple(reg["grid"]) != (grid_h, grid_w):
        log.warning("grid %s != registered grid %s; position normalisation off",
                    (grid_h, grid_w), tuple(reg["grid"]))
        pos_stats = None

    maps = sub_mod.score_maps(fmap, reg["model"], cfg.subspace, pos_stats)
    out.score_map = maps["score"]
    out.valid_grid = valid

    canvas = reg["canvas_size"]
    default_thr = sku_cfg.threshold if sku_cfg else cfg.threshold

    for roi in (sku_cfg.effective_rois() if sku_cfg else []):
        thr = roi.threshold if roi.threshold is not None else default_thr

        if roi.type == "geometry":
            nominal = reg["geometry_nominals"].get(roi.id)
            if not nominal:
                out.rois.append(ROIResult(roi.id, roi.type, float("nan"), thr,
                                          True, "no nominal recorded"))
                continue
            measured = geom_mod.measure_roi_geometry(pre.aligned_part_mask,
                                                     roi.box)
            tol = roi.tolerance or {
                "area_pct": cfg.geometry.area_pct,
                "fill_ratio_pct": cfg.geometry.fill_ratio_pct,
                "corner_angle_deg": cfg.geometry.corner_angle_deg,
            }
            violations = geom_mod.geometry_violations(measured, nominal, tol)
            score = geom_mod.geometry_score(measured, nominal, tol)
            out.rois.append(ROIResult(
                roi.id, roi.type, score, 1.0, not violations,
                "; ".join(violations), measured
            ))
            continue

        gy0, gy1, gx0, gx1 = patch_grid_box(roi.box, canvas, grid_h, grid_w)
        sub = maps["score"][gy0:gy1, gx0:gx1]
        vsub = valid[gy0:gy1, gx0:gx1]
        if not vsub.any():
            out.rois.append(ROIResult(roi.id, roi.type, float("nan"), thr, True,
                                      "no valid patches in ROI"))
            continue
        score = sub_mod.aggregate(sub, vsub, cfg.subspace.aggregation)
        out.rois.append(ROIResult(roi.id, roi.type, score, thr, score <= thr,
                                  f"{int(vsub.sum())} valid patches"))

    out.verdict = "FAIL" if any(not r.passed for r in out.rois) else "PASS"
    out.inference_seconds = time.time() - t0
    return out


def render_result(res: InspectionResult, reg: dict, sku_cfg: SKUConfig,
                  out_dir: Path) -> Path:
    """Heatmap overlay plus the intermediate stages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(res.source).stem or "result"

    if res.pre is None or res.pre.aligned_image is None:
        return out_dir

    tiles = [(res.pre.image, "raw")]
    if res.pre.part_mask is not None:
        tiles.append((viz.overlay_mask(res.pre.image, res.pre.part_mask),
                      "part mask"))
    if res.pre.ink_mask is not None:
        tiles.append((viz.overlay_mask(res.pre.image, res.pre.ink_mask,
                                       (255, 0, 0)), "ink mask"))
    tiles.append((res.pre.aligned_image, "aligned"))

    if res.score_map is not None:
        thr = sku_cfg.threshold if sku_cfg else 6.0
        masked = np.where(res.valid_grid, res.score_map, 0.0)
        # Fixed colour scale anchored on the threshold, never on this image's
        # own min/max -- a per-image rescale would make every part, good or
        # bad, light up identically.
        heat = viz.heatmap_overlay(res.pre.aligned_image, masked,
                                   vmin=0.0, vmax=max(thr * 2.0, 1e-6))
        scores = {r.id: r.score for r in res.rois}
        heat = viz.draw_rois(heat, [r.to_dict() for r in
                                    (sku_cfg.effective_rois() if sku_cfg else [])],
                             scores, thr)
        tiles.append((heat, f"{res.verdict}  max {res.max_score:.2f}"))

    cv2.imwrite(str(out_dir / f"{stem}_stages.png"), viz.grid(tiles, cols=5))
    return out_dir / f"{stem}_stages.png"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Inspect part(s) against a SKU model")
    ap.add_argument("--sku", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    reg = load_registration(args.sku)
    try:
        sku_cfg = SKUConfig.load(args.sku)
    except FileNotFoundError:
        sku_cfg = SKUConfig(sku=args.sku, canvas_size=reg["canvas_size"])
    if args.threshold is not None:
        sku_cfg.threshold = args.threshold

    if args.image:
        paths = [Path(args.image)]
    elif args.dir:
        paths = [p for p in sorted(Path(args.dir).glob("*"))
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    else:
        folder = TEST_DIR / args.sku
        paths = [p for p in sorted(folder.glob("*"))
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not paths:
        print("no images to inspect")
        return 1

    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "inspect" / args.sku
    rows = []
    print(f"{'image':<18}{'verdict':<10}{'max':>8}  detail")
    for p in paths:
        res = inspect_image(load_image(p), reg, sku_cfg, source=p.name)
        render_result(res, reg, sku_cfg, out_dir)
        failed = ", ".join(f"{r.id}={r.score:.2f}" for r in res.failed_rois())
        print(f"{p.name:<18}{res.verdict:<10}{res.max_score:>8.2f}  "
              f"{res.reason or failed}")
        if res.reason_th:
            print(f"{'':<18}{'':<10}{'':>8}  -> {viz.console_safe(res.reason_th)}")
        row = {"image": p.name, "verdict": res.verdict,
               "max_score": res.max_score,
               "inference_s": round(res.inference_seconds, 3),
               "reason": res.reason}
        for r in res.rois:
            row[f"roi_{r.id}"] = r.score
        rows.append(row)

    if args.csv and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    print(f"visualisations in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
