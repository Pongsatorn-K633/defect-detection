"""Measure the two settings that need known-bad parts to establish.

`input_size` and the verdict threshold cannot be judged from good parts alone,
and they fail in the same way when guessed -- silently, on parts that look fine
until a real reject walks past. Both are measured here from one input: a folder
of known-bad photos in `ng_images/<sku>/`.

Why good parts are not enough:

  * `input_size` -- golden scores are flat across resolutions. On SKU
    8983980631 they sat at 1.47-1.50 at every size tried, while the usable
    margin between good and bad collapsed from 7.1x at 1120 to 1.2x at 672.
    Nothing in the golden set hints at that.
  * threshold -- the golden set is the cleanest parts you own. On SKU
    8983864641 golden peaked at 2.04 while the worst genuinely-good part
    reached 10.95, five times higher. A threshold read off the golden set
    rejects good parts.

The floor here comes from the golden images, since those are the only labelled
good parts available at registration time. That makes it optimistic -- real OK
parts score above golden ones -- so `suggest_threshold` leans the suggestion
toward the NG side rather than splitting the gap evenly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as feat_mod
from . import subspace as sub_mod
from .autosize import COARSE, MIN_PATCHES, TARGET_PATCHES, evaluate_size
from .config import NG_DIR, OK_DIR, PipelineConfig
from .pipeline import IMAGE_SUFFIXES, build_reference, load_image, preprocess
from .register import register_sku
from .roi import ROI, SKUConfig, patch_grid_box

log = logging.getLogger(__name__)


def _labelled_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES]


def ng_paths(sku: str) -> list[Path]:
    """Known-bad photos for a SKU."""
    return _labelled_paths(NG_DIR / sku)


def ok_paths(sku: str) -> list[Path]:
    """Known-good photos for a SKU -- ordinary passing parts, not golden ones.

    Optional. Without them the threshold has no measured floor and the
    suggestion can only hang off the NG side; with them it can sit in a real
    gap between two measured populations.
    """
    return _labelled_paths(OK_DIR / sku)


@dataclass
class TuningResult:
    """What the NG set establishes: a resolution and a threshold window."""

    sku: str
    input_size: int | None = None
    threshold: float | None = None
    golden_max: float = float("nan")
    # Worst score among parts that must pass. Includes the golden set, but the
    # number only means something once real OK parts are in it: golden parts
    # are fitted on and score near zero regardless.
    worst_ok: float = float("nan")
    best_ng: float = float("nan")
    n_ng: int = 0
    n_ok: int = 0
    n_golden: int = 0
    sweep: list = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.input_size is not None


def part_score(image, reg, cfg, rois, canvas) -> float:
    """Worst appearance-ROI score on one part -- how inspect_image decides."""
    pre = preprocess(image, cfg, ref_stats=reg["ref_stats"],
                     ref_mask=reg["ref_mask"], ref_image=reg["ref_image"])
    if not pre.ok:
        return float("nan")          # unusable photo says nothing about a part
    fmap = feat_mod.extract_patch_features(pre.aligned_image, cfg.features)
    valid = feat_mod.patch_mask_from_pixel_mask(
        pre.aligned_valid_mask, fmap.shape[0], fmap.shape[1],
        cfg.features.patch_coverage)
    maps = sub_mod.score_maps(fmap, reg["model"], cfg.subspace,
                              reg["position_stats"])
    best = float("nan")
    for roi in rois:
        if roi.type != "appearance":
            continue
        gy0, gy1, gx0, gx1 = patch_grid_box(roi.box, canvas, *fmap.shape[:2])
        sub = maps["score"][gy0:gy1, gx0:gx1]
        vsub = valid[gy0:gy1, gx0:gx1]
        if vsub.any():
            s = sub_mod.aggregate(sub, vsub, cfg.subspace.aggregation)
            best = s if not np.isfinite(best) else max(best, s)
    return best


SAFETY_MARGIN = 0.85


def suggest_in_gap(worst_ok: float, best_ng: float) -> float | None:
    """Sit inside a gap measured on both sides.

    The geometric mean of the worst passing part and the best failing one,
    which is the rule `autosize.recommend` already uses. It sits between the
    two populations without being dragged by either, and unlike the NG-anchored
    fallback it rests on a floor that was actually measured rather than assumed.

    Returns None when the populations overlap -- that is a real finding about
    the parts, not a number to paper over.
    """
    if not (np.isfinite(worst_ok) and np.isfinite(best_ng)):
        return None
    if best_ng <= worst_ok:
        return None                  # no window: good and bad overlap
    value = float(np.sqrt(max(worst_ok, 1e-6) * best_ng))
    return round(value, 1)


def suggest_threshold(golden_max: float, best_ng: float) -> float | None:
    """Sit just under the best bad part, ignoring the golden floor.

    The NG-only fallback, used when no known-good parts were supplied. Prefer
    `suggest_in_gap` whenever a real `worst_ok` exists.

    Anchoring anywhere near `golden_max` is wrong, and wrongly in the
    dangerous direction. Golden photos are the cleanest parts there are, and
    everyday good parts score several times higher: on SKU 8983864641 golden
    peaked at 2.04 while the worst genuinely-good part reached 14.35 -- 7x
    higher. A value derived from the floor landed at 12 and rejected two good
    parts.

    `best_ng` is the only number an NG-only run measures against real parts,
    so the suggestion hangs off it: a little below the weakest defect, which
    is as high as it can go while still catching everything known-bad. Too
    high merely misses a marginal defect; too low rejects good parts all day,
    which is the failure that stops a line.

    The floor is still enforced -- a threshold under `golden_max` would fail
    the reference parts themselves -- but it never sets the value.
    """
    if not (np.isfinite(golden_max) and np.isfinite(best_ng)):
        return None
    if best_ng <= golden_max:
        return None                  # no window: good and bad overlap
    value = best_ng * SAFETY_MARGIN
    if value <= golden_max:          # window too narrow to sit inside
        value = (golden_max + best_ng) / 2.0
    return round(float(value), 1)


def tune(
    sku: str,
    golden_paths: list[Path],
    rois: list[ROI],
    base_cfg: PipelineConfig | None = None,
    candidates: tuple[int, ...] = COARSE,
    say=None,
) -> TuningResult:
    """Pick `input_size` from the NG separation, then measure the threshold."""
    say = say or (lambda m: log.info(m))
    base_cfg = base_cfg or PipelineConfig()
    res = TuningResult(sku=sku, n_golden=len(golden_paths))

    ngs = ng_paths(sku)
    oks = ok_paths(sku)
    res.n_ng = len(ngs)
    res.n_ok = len(oks)
    if not ngs:
        res.reason = (
            f"No known-bad photos in `ng_images/{sku}/`. Both values need "
            f"them: good parts score almost identically at every resolution, "
            f"and the golden set cannot show where good and bad separate. "
            f"Add at least one photo per check on the inspection sheet.")
        return res
    if len(golden_paths) < 2:
        res.reason = ("Need at least 2 golden images — a subspace cannot be "
                      "fitted from one part.")
        return res
    if not [r for r in rois if r.type == "appearance"]:
        res.reason = ("No appearance check defined. `input_size` sets the "
                      "patch grid, which only appearance checks use, and the "
                      "threshold scores them — define one in step 2 first.")
        return res

    golden = [load_image(p) for p in golden_paths]
    names = [p.name for p in golden_paths]
    ng_images = [(p.name, load_image(p)) for p in ngs]
    # Known-good parts are optional, and are scored exactly like the bad ones:
    # as test parts against the fitted model, never as parts to fit on.
    ok_images = [(p.name, load_image(p)) for p in oks]
    labelled = ng_images + ok_images
    labels = {n: 1 for n, _ in ng_images}
    labels.update({n: 0 for n, _ in ok_images})

    # Resolution first: the threshold is only meaningful at the resolution the
    # model is actually fitted at.
    # Segment and align every photo once. None of it depends on the feature
    # resolution, and repeating it per size was most of the time: 3 sizes
    # re-segmented each golden photo 6 times and each labelled photo 3 times.
    say(f"finding and aligning the part in {len(golden) + len(labelled)} "
        f"photos (once, for every resolution)")
    t0 = time.time()
    prepared = build_reference(golden, base_cfg, names)
    say(f"golden photos aligned in {time.time() - t0:.1f} s")
    ref_stats, ref_image, ref_mask, _ = prepared
    test_pre = {}
    for n, img in labelled:
        t1 = time.time()
        test_pre[n] = preprocess(img, base_cfg, ref_stats=ref_stats,
                                 ref_mask=ref_mask, ref_image=ref_image,
                                 source=n)
        say(f"aligned {n} in {time.time() - t1:.1f} s")

    say(f"sweeping {len(candidates)} resolutions against {len(ngs)} bad "
        f"and {len(oks)} good part(s)")
    sized = []
    for size in candidates:
        sized.append(evaluate_size(sku, size, golden, names, rois, base_cfg,
                                   labelled, labels, say=say,
                                   prepared=prepared, test_pre=test_pre))
    res.sweep = sized

    def rank(r):
        m = r.margin if np.isfinite(r.margin) else -1.0
        return (r.min_patches >= MIN_PATCHES, round(m, 2), -r.input_size)

    best = max(sized, key=rank)
    res.input_size = best.input_size
    res.golden_max = best.golden_max
    res.worst_ok = best.worst_ok
    res.best_ng = best.best_ng

    # With real OK parts the floor is measured, so the threshold can sit in the
    # gap. Without them `worst_ok` is just the golden set scoring near zero, and
    # anchoring on it would put the threshold far below where good parts land.
    validated = res.n_ok > 0
    if validated:
        res.threshold = suggest_in_gap(best.worst_ok, best.best_ng)
    else:
        res.threshold = suggest_threshold(best.golden_max, best.best_ng)

    if res.threshold is None:
        floor = best.worst_ok if validated else best.golden_max
        which = "good parts" if validated else "golden parts"
        res.reason = (
            f"At input_size={best.input_size} the weakest bad part scores "
            f"{best.best_ng:.2f} and the worst of the {which} reaches "
            f"{floor:.2f} — they overlap, so no threshold separates them.\n\n"
            + ("Find the good part scoring highest and the bad part scoring "
               "lowest and compare them: usually either a good part is "
               "genuinely marked up, or a defect is too small for the check "
               "box around it."
               if validated else
               "More golden images spanning the real lighting variation "
               "usually helps more than more resolution."))
        return res

    if validated:
        margin = (best.best_ng / best.worst_ok
                  if best.worst_ok > 0 else float("inf"))
        res.reason = (
            f"Measured against **{res.n_ng}** known-bad, **{res.n_ok}** "
            f"known-good and **{res.n_golden}** golden part(s).\n\n"
            f"At `input_size={best.input_size}`:\n\n"
            f"| | score |\n|---|---|\n"
            f"| worst good part (must pass) | **{best.worst_ok:.2f}** |\n"
            f"| weakest bad part (must fail) | **{best.best_ng:.2f}** |\n"
            f"| separation | **{margin:.1f}x** |\n\n"
            f"**{res.threshold:g}** sits inside that gap — above every good "
            f"part measured, below every bad one. Both sides were measured on "
            f"real parts, so this is a verified window rather than an "
            f"estimate.\n\n"
            f"The narrower the gap, the more a new part can drift before it "
            f"lands on the wrong side of the line. More known-good parts "
            f"spanning real lighting and handling variation is what tightens "
            f"this.\n\n"
            f"The smallest check holds {best.min_patches} valid patches "
            f"(target {TARGET_PATCHES}).")
    else:
        res.reason = (
            f"Measured against **{res.n_ng}** known-bad and **{res.n_golden}** "
            f"golden part(s).\n\n"
            f"At `input_size={best.input_size}` the weakest bad part scores "
            f"**{best.best_ng:.2f}**, so **{res.threshold:g}** sits just under "
            f"it — high enough to pass everyday good parts, low enough to "
            f"still catch every known-bad one.\n\n"
            f"⚠️ **This is an upper bound, not a verified value.** No "
            f"known-good parts were supplied, so nothing here measured where "
            f"good parts actually land. Golden photos scored up to "
            f"{best.golden_max:.2f}, but the subspace is *fitted on* them — "
            f"they score near zero by construction and everyday good parts "
            f"score several times higher.\n\n"
            f"**Upload known-good parts above and re-measure** to turn this "
            f"into a measured window.\n\n"
            f"The smallest check holds {best.min_patches} valid patches "
            f"(target {TARGET_PATCHES}).")
    if best.min_patches < MIN_PATCHES:
        res.reason += (
            f"\n\n**Warning:** under {MIN_PATCHES} patches, the verdict rides "
            f"on one or two of them and a good part can trip it. Consider a "
            f"larger check box or a higher resolution.")
    return res


def table(res: TuningResult) -> str:
    """The sweep as markdown, so the recommendation shows its working."""
    if not res.sweep:
        return ""
    head = ("| input resolution | patches | worst golden | worst good | "
            "best bad | margin |\n|---|---|---|---|---|---|\n")
    rows = []
    for r in sorted(res.sweep, key=lambda r: r.input_size):
        margin = f"{r.margin:.1f}x" if np.isfinite(r.margin) else "—"
        best_ng = f"{r.best_ng:.2f}" if np.isfinite(r.best_ng) else "—"
        # Only meaningful once real OK parts are in it; otherwise worst_ok is
        # the golden set again and repeating it would imply a second reading.
        worst_ok = (f"{r.worst_ok:.2f}"
                    if res.n_ok > 0 and np.isfinite(r.worst_ok) else "—")
        mark = " ← chosen" if r.input_size == res.input_size else ""
        rows.append(f"| {r.input_size}{mark} | {r.min_patches} | "
                    f"{r.golden_max:.2f} | {worst_ok} | {best_ng} | {margin} |")
    return head + "\n".join(rows)
