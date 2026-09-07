"""Recommend a per-SKU DINOv2 `input_size`.

The backbone sees the canonical canvas resized to `input_size`, so an ROI's
patch count is set by its share of the canvas -- not by the camera's
resolution, which is discarded at that resize. A small ROI can therefore hold
too few patches to score reliably even from a 48 MP photo.

That matters because `max` aggregation makes the ROI verdict ride on the
single worst patch. With ~16 patches a good part trips it on an unlucky draw;
with ~45 the statistic is stable. Raising `input_size` for that SKU buys real
patches instead of a luckier one.

Two questions, and they need different evidence:

  * How many valid patches does the ROI hold?  Golden images alone answer
    this, and it is the cheap screen.
  * Does the size actually separate good parts from bad?  Only labelled NG
    parts answer this, and it is the one that decides.

The second is not optional in the way it looks. On SKU 8983980631 the golden
scores sit at 1.47-1.50 at *every* size tested -- perfectly stable, and
completely uninformative. What moves is the worst passing part: at 672 it
reaches 17.59 against a best NG of 20.61, a 1.2x margin that is unusable. The
golden set never hints at it. So a recommendation made without NG parts is a
starting point, and this module labels it as such rather than pretending
otherwise.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as feat_mod
from . import subspace as sub_mod
from .config import DATA_DIR, TEST_DIR, PipelineConfig
from .pipeline import load_image, preprocess
from .register import register_sku
from .roi import ROI, SKUConfig, patch_grid_box

log = logging.getLogger(__name__)

# Sizes must be multiples of the patch size (14) or the backbone's grid does
# not divide evenly; these are the multiples that span the useful range.
COARSE = (672, 896, 1120)
REFINE_STEP = 112               # 8 patches -- one step either side of a winner
MIN_SIZE, MAX_SIZE = 448, 1568

# A `max`-aggregated ROI below this many valid patches rides on one or two
# patches. Measured, not guessed: 15-16 patches gave a 1.2x margin on
# 8983980631, 27-33 gave 5.1x, 44-52 gave 7.1x.
TARGET_PATCHES = 40
MIN_PATCHES = 25
# Below this, good and bad are not reliably separated at the part level.
GOOD_MARGIN = 3.0


@dataclass
class SizeResult:
    """What one candidate `input_size` achieved."""

    input_size: int
    grid: tuple[int, int]
    # Per ROI: the smallest valid-patch count seen across the golden set.
    roi_patches: dict[str, int] = field(default_factory=dict)
    golden_max: float = float("nan")
    worst_ok: float = float("nan")
    best_ng: float = float("nan")
    margin: float = float("nan")
    fit_seconds: float = 0.0
    note: str = ""

    @property
    def min_patches(self) -> int:
        return min(self.roi_patches.values()) if self.roi_patches else 0


@dataclass
class AutoSizeReport:
    """The sweep, plus what it does and does not establish."""

    sku: str
    results: list[SizeResult] = field(default_factory=list)
    recommended: int | None = None
    validated: bool = False          # True only when NG labels were used
    reason: str = ""
    n_ng: int = 0
    n_ok: int = 0

    def table(self) -> str:
        """A fixed-width summary; the same text serves CLI and UI."""
        if not self.results:
            return "no sizes evaluated"
        head = (f"{'input_size':>10} {'grid':>9} {'patches':>8} "
                f"{'golden':>7} {'worst OK':>9} {'best NG':>8} {'margin':>7}")
        rows = [head, "-" * len(head)]
        for r in sorted(self.results, key=lambda r: r.input_size):
            grid = f"{r.grid[0]}x{r.grid[1]}"
            margin = f"{r.margin:.1f}x" if np.isfinite(r.margin) else "-"
            worst = f"{r.worst_ok:.2f}" if np.isfinite(r.worst_ok) else "-"
            best = f"{r.best_ng:.2f}" if np.isfinite(r.best_ng) else "-"
            mark = "  <-- recommended" if r.input_size == self.recommended else ""
            rows.append(f"{r.input_size:>10} {grid:>9} {r.min_patches:>8} "
                        f"{r.golden_max:>7.2f} {worst:>9} {best:>8} "
                        f"{margin:>7}{mark}")
        return "\n".join(rows)


def read_labels(sku: str) -> dict[str, int]:
    """Ground truth from data/labels/<sku>.csv (`image,label`, 1 = NG)."""
    path = DATA_DIR / "labels" / f"{sku}.csv"
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or parts[1] in ("", "?"):
            continue
        try:
            out[parts[0]] = int(parts[1])
        except ValueError:
            continue                      # header row
    return out


def _appearance_rois(rois: list[ROI]) -> list[ROI]:
    """Only appearance ROIs are affected: geometry never sees a patch grid."""
    return [r for r in rois if r.type == "appearance"]


def _roi_scores(fmap, valid, model, cfg, pos_stats, roi, canvas):
    """Aggregate one ROI on one image, or NaN if it holds no valid patch."""
    gh, gw = fmap.shape[:2]
    gy0, gy1, gx0, gx1 = patch_grid_box(roi.box, canvas, gh, gw)
    maps = sub_mod.score_maps(fmap, model, cfg.subspace, pos_stats)
    sub = maps["score"][gy0:gy1, gx0:gx1]
    vsub = valid[gy0:gy1, gx0:gx1]
    if not vsub.any():
        return float("nan"), 0
    return (sub_mod.aggregate(sub, vsub, cfg.subspace.aggregation),
            int(vsub.sum()))


def evaluate_size(
    sku: str,
    size: int,
    golden: list[np.ndarray],
    golden_names: list[str],
    rois: list[ROI],
    base_cfg: PipelineConfig,
    test: list[tuple[str, np.ndarray]] | None = None,
    labels: dict[str, int] | None = None,
    say=None,
) -> SizeResult:
    """Fit at `size` and measure what it achieves.

    A full refit per candidate is unavoidable: the patch grid changes with the
    resolution, so the subspace and the position statistics are only valid at
    the size they were fitted at.
    """
    import copy

    cfg = copy.deepcopy(base_cfg)
    cfg.features.input_size = int(size)
    if say:
        say(f"fitting at input_size={size}")

    t0 = time.time()
    reg = register_sku(sku, golden, cfg, golden_names, rois)
    res = SizeResult(input_size=int(size), grid=tuple(reg["grid"]),
                     fit_seconds=time.time() - t0)

    canvas = reg["canvas_size"]
    checked = _appearance_rois(rois)
    if not checked:
        res.note = "no appearance ROI; input_size does not affect this SKU"
        return res

    # Golden side: patch counts and the normal score ceiling.
    golden_vals: list[float] = []
    counts: dict[str, list[int]] = {r.id: [] for r in checked}
    for pre in reg["results"]:
        fmap = feat_mod.extract_patch_features(pre.aligned_image, cfg.features)
        valid = feat_mod.patch_mask_from_pixel_mask(
            pre.aligned_valid_mask, fmap.shape[0], fmap.shape[1],
            cfg.features.patch_coverage)
        for roi in checked:
            s, n = _roi_scores(fmap, valid, reg["model"], cfg,
                               reg["position_stats"], roi, canvas)
            counts[roi.id].append(n)
            if np.isfinite(s):
                golden_vals.append(s)
    res.roi_patches = {k: (min(v) if v else 0) for k, v in counts.items()}
    res.golden_max = max(golden_vals) if golden_vals else float("nan")

    if not test or not labels:
        res.note = "screened on patch count only (no labelled NG parts)"
        return res

    # Test side: the worst part that must pass, and the best that must fail.
    ok_vals, ng_vals = [], []
    for name, img in test:
        y = labels.get(name)
        if y is None:
            continue
        pre = preprocess(img, cfg, ref_stats=reg["ref_stats"],
                         ref_mask=reg["ref_mask"], ref_image=reg["ref_image"],
                         source=name)
        if not pre.ok:
            continue                  # RETAKE says nothing about the part
        fmap = feat_mod.extract_patch_features(pre.aligned_image, cfg.features)
        valid = feat_mod.patch_mask_from_pixel_mask(
            pre.aligned_valid_mask, fmap.shape[0], fmap.shape[1],
            cfg.features.patch_coverage)
        per_roi = [_roi_scores(fmap, valid, reg["model"], cfg,
                               reg["position_stats"], roi, canvas)[0]
                   for roi in checked]
        per_roi = [v for v in per_roi if np.isfinite(v)]
        if not per_roi:
            continue
        # The part-level score is the worst ROI, exactly as inspect_image
        # decides a verdict.
        (ng_vals if y == 1 else ok_vals).append(max(per_roi))

    # A golden part is an OK part: it must pass too, so it belongs in the
    # constraint that sets the threshold's floor.
    ok_all = ok_vals + golden_vals
    res.worst_ok = max(ok_all) if ok_all else float("nan")
    res.best_ng = min(ng_vals) if ng_vals else float("nan")
    if np.isfinite(res.worst_ok) and np.isfinite(res.best_ng) and res.worst_ok > 0:
        res.margin = res.best_ng / res.worst_ok
    return res


def _score_candidate(r: SizeResult, validated: bool) -> tuple:
    """Rank candidates. Sorted descending, the first is the recommendation."""
    if validated and np.isfinite(r.margin):
        # Separation first, but only above the patch floor -- a large margin
        # measured on 16 patches is one lucky draw, not a stable result.
        return (r.min_patches >= MIN_PATCHES, round(r.margin, 2),
                -r.input_size)
    # Unvalidated: reach the patch target, then prefer the cheapest size.
    return (r.min_patches >= TARGET_PATCHES, r.min_patches >= MIN_PATCHES,
            -r.input_size)


def recommend(
    sku: str,
    golden_paths: list[Path] | None = None,
    rois: list[ROI] | None = None,
    base_cfg: PipelineConfig | None = None,
    candidates: tuple[int, ...] = COARSE,
    refine: bool = True,
    say=None,
) -> AutoSizeReport:
    """Sweep `input_size` and recommend one.

    Coarse pass over `candidates`, then -- if `refine` -- one step either side
    of the winner, which costs two more fits and catches a peak sitting
    between the coarse points.
    """
    say = say or (lambda m: log.info(m))
    sku_cfg = SKUConfig.load(sku)
    rois = rois if rois is not None else sku_cfg.rois
    base_cfg = base_cfg or PipelineConfig()

    report = AutoSizeReport(sku=sku)
    if not _appearance_rois(rois):
        report.reason = ("This SKU has no appearance ROI. `input_size` sets "
                         "the patch grid, which only appearance checks use -- "
                         "geometry measures the silhouette mask instead. "
                         "Nothing to tune.")
        return report

    from .config import GOLDEN_DIR
    if golden_paths is None:
        folder = GOLDEN_DIR / sku
        golden_paths = [p for p in sorted(folder.glob("*"))
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                        and not p.name.startswith("IS_")]
    if len(golden_paths) < 2:
        report.reason = (f"Need at least 2 golden images to fit a subspace; "
                         f"found {len(golden_paths)}.")
        return report

    golden = [load_image(p) for p in golden_paths]
    golden_names = [p.name for p in golden_paths]

    labels = read_labels(sku)
    test: list[tuple[str, np.ndarray]] = []
    if labels:
        folder = TEST_DIR / sku
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} \
                    and p.name in labels:
                test.append((p.name, load_image(p)))
    report.n_ng = sum(1 for n, _ in test if labels.get(n) == 1)
    report.n_ok = sum(1 for n, _ in test if labels.get(n) == 0)
    report.validated = report.n_ng > 0

    if not report.validated:
        say("no labelled NG parts -- screening on patch count only")

    seen: dict[int, SizeResult] = {}

    def run(size: int) -> SizeResult:
        size = int(round(size / 14) * 14)
        size = max(MIN_SIZE, min(MAX_SIZE, size))
        if size not in seen:
            seen[size] = evaluate_size(
                sku, size, golden, golden_names, rois, base_cfg,
                test if report.validated else None,
                labels if report.validated else None, say=say)
        return seen[size]

    for size in candidates:
        run(size)

    best = max(seen.values(), key=lambda r: _score_candidate(r, report.validated))
    if refine:
        for neighbour in (best.input_size - REFINE_STEP,
                          best.input_size + REFINE_STEP):
            if MIN_SIZE <= neighbour <= MAX_SIZE:
                run(neighbour)
        best = max(seen.values(),
                   key=lambda r: _score_candidate(r, report.validated))

    report.results = list(seen.values())
    report.recommended = best.input_size
    report.reason = _explain(best, report)
    return report


def _explain(best: SizeResult, report: AutoSizeReport) -> str:
    """Say what the recommendation rests on, and what it does not."""
    lines = []
    if report.validated:
        lines.append(
            f"VALIDATED against {report.n_ng} NG and {report.n_ok} OK part(s).")
        if np.isfinite(best.margin):
            lines.append(
                f"At input_size={best.input_size} the worst passing part scores "
                f"{best.worst_ok:.2f} and the best failing part {best.best_ng:.2f} "
                f"-- a {best.margin:.1f}x margin. Put the threshold in that gap.")
        if np.isfinite(best.margin) and best.margin < GOOD_MARGIN:
            lines.append(
                f"WARNING: a margin under {GOOD_MARGIN:.0f}x is thin. Good and "
                f"bad parts are close at every size tried; more golden images "
                f"spanning the real lighting variation usually helps more than "
                f"more resolution.")
    else:
        lines.append(
            "SCREENED on patch count only -- no labelled NG parts were "
            "available, so this is a starting point, not a verified choice.")
        lines.append(
            "The golden set cannot reveal a separation failure: on SKU "
            "8983980631 golden scores sat at 1.49 at every size while the "
            "usable margin collapsed from 7.1x to 1.2x. Add one NG photo per "
            "sheet check and re-run to confirm.")
    lines.append(
        f"Smallest ROI holds {best.min_patches} valid patches "
        f"(target {TARGET_PATCHES}, floor {MIN_PATCHES}).")
    if best.min_patches < MIN_PATCHES:
        lines.append(
            f"WARNING: still under {MIN_PATCHES} patches at the largest size "
            f"tried. With `max` aggregation the verdict rides on one or two "
            f"patches. Consider a larger ROI box, `top3mean` aggregation, or "
            f"a lower patch_coverage.")
    return "\n".join(lines)


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="Recommend a per-SKU DINOv2 input_size")
    ap.add_argument("--sku", required=True)
    ap.add_argument("--sizes", type=int, nargs="*", default=None,
                    help=f"candidate sizes (default {list(COARSE)})")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the refinement pass around the winner")
    ap.add_argument("--apply", action="store_true",
                    help="write the recommendation into the SKU's sku.json")
    args = ap.parse_args()

    report = recommend(args.sku,
                       candidates=tuple(args.sizes) if args.sizes else COARSE,
                       refine=not args.no_refine,
                       say=lambda m: print(f"  {m}"))
    if report.recommended is None:
        print(f"\n{report.reason}")
        return 1

    print(f"\n{report.table()}\n")
    print(report.reason)

    if args.apply:
        sku_cfg = SKUConfig.load(args.sku)
        sku_cfg.input_size = report.recommended
        sku_cfg.save()
        print(f"\nwrote input_size={report.recommended} to "
              f"{sku_cfg.sku}/sku.json")
        print("Re-register so the model is fitted at that resolution:")
        print(f"  python -m inspector.register --sku {args.sku}")
    else:
        print(f"\nTo apply: python -m inspector.autosize --sku {args.sku} --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
