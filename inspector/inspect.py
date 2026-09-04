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
    # Four outcomes, and the distinctions matter operationally:
    #   PASS    every sheet check is within limits
    #   FAIL    a sheet check failed -- the only ground for rejection
    #   REVIEW  sheet checks pass, but something outside them scores high.
    #           A person decides; the machine does not reject off-sheet.
    #   RETAKE  the photo is unusable, so nothing was learned about the part
    verdict: str = "PASS"          # PASS | FAIL | REVIEW | RETAKE
    rois: list[ROIResult] = field(default_factory=list)
    score_map: np.ndarray | None = None
    valid_grid: np.ndarray | None = None
    pre: PreprocessResult | None = None
    inference_seconds: float = 0.0
    reason: str = ""
    reason_th: str = ""            # operator-facing wording in Thai
    # Set when something outside every sheet ROI scores high: the score, and
    # where it is on the canonical canvas so the operator can be shown where
    # to look.
    review_score: float = 0.0
    review_threshold: float = 0.0
    review_xy: tuple[int, int] | None = None

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

    # Off-sheet anomalies: surface them to a person rather than rejecting on
    # them. A FAIL already sends the part back, so only a passing part is
    # worth reviewing.
    if out.verdict == "PASS" and cfg.review_enabled:
        review = _off_roi_review(maps["score"], valid, sku_cfg, reg, cfg,
                                 grid_h, grid_w)
        if review:
            out.verdict = "REVIEW"
            out.reason = (f"something outside the inspection-sheet checks "
                          f"scores {review['score']:.1f} "
                          f"(review bar {review['threshold']:.1f}) -- "
                          f"not grounds for rejection, please look at the part")
            out.reason_th = "แนะนำให้ตรวจสอบอีกครั้ง"
            out.review_score = review["score"]
            out.review_threshold = review["threshold"]
            out.review_xy = review["xy"]

    out.inference_seconds = time.time() - t0
    return out


def _off_roi_review(score_map, valid, sku_cfg, reg, cfg, grid_h, grid_w):
    """Is anything outside every sheet ROI anomalous enough to show a human?

    The bar is a multiple of the worst score the GOLDEN parts produced in the
    same off-ROI region, so it adapts to how busy a given part is rather than
    being one global number that means different things on different SKUs.
    Returns None when there is nothing worth reviewing.
    """
    rois = sku_cfg.effective_rois() if sku_cfg else []
    if not rois:
        return None

    covered = np.zeros((grid_h, grid_w), bool)
    for roi in rois:
        gy0, gy1, gx0, gx1 = patch_grid_box(roi.box, reg["canvas_size"],
                                            grid_h, grid_w)
        covered[gy0:gy1, gx0:gx1] = True
    off = valid & ~covered
    if not off.any():
        return None

    golden_off = reg.get("golden_scores", {}).get("outside_rois") or []
    if not golden_off:
        # No baseline recorded (an older model): staying silent is better than
        # reviewing on a number we cannot justify.
        return None

    threshold = max(cfg.review_sigma * float(max(golden_off)), cfg.review_floor)
    score = sub_mod.aggregate(score_map, off, cfg.subspace.aggregation)
    if not np.isfinite(score) or score <= threshold:
        return None

    masked = np.where(off, score_map, -np.inf)
    iy, ix = np.unravel_index(np.argmax(masked), masked.shape)
    cw, ch = reg["canvas_size"]
    return {"score": float(score), "threshold": float(threshold),
            "xy": (int((ix + 0.5) / grid_w * cw),
                   int((iy + 0.5) / grid_h * ch))}


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
        if res.review_xy is not None:
            # Circle what triggered the review. Without this the operator is
            # told to look again but not where, which is barely better than
            # not telling them.
            cv2.circle(heat, res.review_xy, 34, (0, 220, 255), 3)
            heat = viz.label(heat, f"review {res.review_score:.1f}",
                             (max(res.review_xy[0] - 40, 4),
                              max(res.review_xy[1] - 42, 16)),
                             (0, 220, 255), 0.55)
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
    # Apply the same per-SKU overrides registration used, or the model is
    # scored on a different patch grid than it was fitted on.
    reg["config"] = sku_cfg.apply_to(reg["config"])

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
