"""Prove inspection-sheet boxes on labelled photos, and search for better ones.

A box can look right and still be wrong. On 898213-8750 the first box ran down
the flange edge, whose shine changes photo to photo, so a good part scored 69
while the weakest bad one scored 20 -- and no box of any kind could catch a
missing tab, because appearance reads the bare metal left behind as normal.
Neither mistake is visible by eye. Both are obvious once every golden, good
and bad photo is scored inside the box, which is what this module does:

  * `check_boxes` -- judge every labelled photo with the checks as drawn, using
    the same code as inspection, and say which photo each check gets wrong.
  * `search` -- try many boxes around a drawn one, as a surface check and as
    a "must be metal" / "must be empty" presence check, and return the best of
    each kind with the gap that proves it.

Photos are scored once (the slow part: segment, align, DINOv2) and every box
after that is arithmetic on the cached score maps and masks. Test photos are
never read: the boxes are chosen on the labelled sets, so a blind test stays
blind.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import subspace as sub_mod
from .autosize import MIN_PATCHES
from .config import GOLDEN_DIR, NG_DIR, OK_DIR
from .inspect import inspect_image, score_checks
from .pipeline import IMAGE_SUFFIXES, load_image
from .roi import GEOM_SUFFIX, ROI, SKUConfig, group_checks, patch_grid_box

log = logging.getLogger(__name__)

GOOD, BAD = 0, 1

# A surface box counts a bad part as caught only if it scores this far above
# the worst good part. At 1.15x on SKU 8983864641 one extra photo was enough
# to flip a verdict; 1.25x leaves room for the parts not photographed yet.
SURFACE_GAP = 1.25
# A surface box whose weakest caught bad part is this far above the worst good
# one is preferred to a box that catches one more defect on a thinner gap.
# Hiding one photo at a time on 898213-8750, the 1.4x box that caught all six
# bad photos failed a good part it had not seen and missed a defect.
SAFE_RATIO = 1.5
# A presence box needs good and bad parts this far apart in metal coverage
# (0-1). Good parts vary with how the tab sits -- 73-100% on 898213-8750 --
# so anything tighter would be fitting one batch of photos.
PRESENCE_GAP = 0.30
# Presence windows at least this far apart are preferred, largest first;
# below it the gap decides. Half the box is a difference no lighting or pose
# change on a good part has come close to.
CLEAR_GAP = 0.50


@dataclass
class Photo:
    """One labelled photo, scored once and cached."""

    name: str
    folder: str                  # "golden" | "good" | "bad"
    label: int                   # GOOD or BAD
    retake: str = ""             # why it could not be judged, if it could not
    mask: np.ndarray | None = None
    gray: np.ndarray | None = None       # aligned photo, for hole checks
    integral: np.ndarray | None = None
    score_map: np.ndarray | None = None
    valid: np.ndarray | None = None


@dataclass
class Suggestion:
    kind: str                    # "surface" | "must be metal" | "must be empty"
    box: tuple
    limit: float                 # surface threshold, or metal fraction 0-1
    caught: list = field(default_factory=list)
    missed: list = field(default_factory=list)
    worst_good: float = float("nan")
    weakest_bad: float = float("nan")
    worst_good_name: str = ""
    # Hide one photo, search again on the rest, judge the hidden one: how the
    # suggestion does on photos it was not chosen on. (right, judged, wrong)
    holdout: tuple = (0, 0, ())
    # Further tolerance keys the check carries (a hole's shape limits).
    extra: dict = field(default_factory=dict)

    @property
    def trust(self) -> str:
        right, judged, wrong = self.holdout
        if not judged:
            return "not tested on unseen photos"
        return (f"right on {right} of {judged} photos it had not seen"
                + (f" (wrong: {', '.join(wrong)})" if wrong else ""))

    @property
    def label(self) -> str:
        x, y, w, h = self.box
        n = len(self.caught) + len(self.missed)
        if self.kind == "surface":
            proof = (f"worst good {self.worst_good:.1f}, weakest caught bad "
                     f"{self.weakest_bad:.1f} ({self.weakest_bad / self.worst_good:.1f}x)")
            rule = f"limit {self.limit:.1f}"
        elif self.kind == "hole present" and not self.caught:
            return (f"geometry (hole must be there) at ({x}, {y}, {w}, {h}), "
                    f"hole at least {self.limit:.0f} px: every good photo "
                    f"passes (smallest hole {self.worst_good:.0f} px) -- NOT "
                    f"PROVEN: no bad photo shows this hole missing; "
                    f"{self.trust}")
        elif self.kind == "hole present":
            proof = (f"smallest hole on a good part {self.worst_good:.0f} px, "
                     f"largest on a caught bad part {self.weakest_bad:.0f} px")
            rule = f"hole at least {self.limit:.0f} px"
        else:
            proof = (f"good parts {100 * self.worst_good:.0f}% metal, caught bad "
                     f"{100 * self.weakest_bad:.0f}%")
            op = ">=" if self.kind == "must be metal" else "<="
            rule = f"metal {op} {100 * self.limit:.0f}%"
        shown = {"surface": "appearance",
                 "hole present": "geometry (hole must be there)",
                 "must be metal": "geometry (must be metal)",
                 "must be empty": "geometry (must be empty)"}[self.kind]
        return (f"{shown} at ({x}, {y}, {w}, {h}), {rule}: catches "
                f"{len(self.caught)} of {n} bad, every good passes -- {proof}; "
                f"{self.trust}")

    def rois(self, name: str, note: str = "") -> list[ROI]:
        box = tuple(int(v) for v in self.box)
        if self.kind == "surface":
            return [ROI(name, "appearance", box, threshold=round(self.limit, 2),
                        note=note)]
        if self.kind == "hole present":
            return [ROI(f"{name}{GEOM_SUFFIX}", "geometry", box,
                        tolerance={"min_hole_px": round(self.limit),
                                   "hole_px": round(self.worst_good),
                                   **self.extra},
                        note=note)]
        key ="min_area_frac" if self.kind == "must be metal" else "max_area_frac"
        return [ROI(f"{name}{GEOM_SUFFIX}", "geometry", box,
                    tolerance={key: round(self.limit, 2)}, note=note)]


def labelled_paths(sku: str) -> list[tuple[Path, str, int]]:
    """Golden and good photos (must pass) and bad photos (must fail)."""
    out = []
    for root, folder, label in ((GOLDEN_DIR, "golden", GOOD),
                                (OK_DIR, "good", GOOD), (NG_DIR, "bad", BAD)):
        d = root / sku
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES \
                    and not p.name.startswith("IS_"):
                out.append((p, folder, label))
    return out


def score_photos(sku: str, reg: dict, paths, progress=None) -> list[Photo]:
    """Segment, align and score every photo once against the fitted model."""
    bare = SKUConfig(sku=sku, canvas_size=reg["canvas_size"], rois=[])
    photos = []
    for i, (path, folder, label) in enumerate(paths):
        if progress:
            progress(i / max(len(paths), 1), desc=f"scoring {path.name}")
        ph = Photo(path.name, folder, label)
        try:
            res = inspect_image(load_image(path), reg, bare, source=path.name)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop it
            ph.retake = f"could not read: {exc}"
            photos.append(ph)
            continue
        if res.verdict == "RETAKE" or res.score_map is None:
            ph.retake = res.reason or "photo unusable"
        else:
            ph.mask = res.pre.aligned_part_mask
            ph.gray = cv2.cvtColor(res.pre.aligned_image, cv2.COLOR_BGR2GRAY)
            ph.integral = cv2.integral((ph.mask > 0).astype(np.uint8))
            ph.score_map = res.score_map
            ph.valid = res.valid_grid
        photos.append(ph)
    return photos


def check_boxes(photos: list[Photo], rois: list[ROI], reg: dict,
                threshold: float) -> dict:
    """Judge every usable photo with these checks, exactly as inspection would.

    Returns rows (one per photo) and the photos each check gets wrong: a good
    part it fails, and the bad parts that no check catches.
    """
    cfg = reg["config"]
    sku_cfg = SKUConfig(sku=reg.get("sku", ""), canvas_size=reg["canvas_size"],
                        rois=rois, threshold=float(threshold))
    checks = group_checks(rois)
    rows, wrong, uncaught, retakes = [], [], [], []
    false_fails = {name: [] for name, _, _ in checks}
    # Which half of a check failed the good part, so the advice can be "set
    # the type to geometry" when only the appearance half is at fault.
    half_fails = {name: {"appearance": 0, "geometry": 0} for name, _, _ in checks}
    for ph in photos:
        if ph.retake:
            retakes.append(f"{ph.name} ({ph.retake})")
            continue
        results = score_checks(SimpleNamespace(aligned_part_mask=ph.mask,
                                               aligned_gray=ph.gray),
                               ph.score_map, ph.valid, reg, sku_cfg, cfg)
        by_id = {r.id: r for r in results}
        cells, failed = [], []
        for name, app, geo in checks:
            parts = [by_id.get(r.id) for r in (app, geo) if r is not None]
            ok = all(r.passed for r in parts if r is not None)
            shown = " + ".join(
                "n/a" if not np.isfinite(r.score) else
                f"{r.score:.2f}/{r.threshold:.2f}" for r in parts if r is not None)
            cells.append(f"{'pass' if ok else 'FAIL'} {shown}")
            if not ok:
                failed.append(name)
                if ph.label == GOOD:
                    false_fails[name].append(ph.name)
                    for roi in (app, geo):
                        r = by_id.get(roi.id) if roi is not None else None
                        if r is not None and not r.passed:
                            half_fails[name][roi.type] += 1
        verdict = "FAIL" if failed else "PASS"
        want = "FAIL" if ph.label == BAD else "PASS"
        right = verdict == want
        if not right:
            wrong.append(ph.name)
            if ph.label == BAD:
                uncaught.append(ph.name)
        rows.append([("✅" if right else "❌"), ph.name, ph.folder, want,
                     verdict, *cells])
    headers = ["", "photo", "set", "should be", "got",
               *[name for name, _, _ in checks]]
    return {"headers": headers, "rows": rows, "wrong": wrong,
            "uncaught": uncaught, "false_fails": false_fails,
            "half_fails": half_fails,
            "both_halves": {name: bool(app and geo) for name, app, geo in checks},
            "retakes": retakes}


def _around(box, canvas, scales, shifts):
    """Boxes near `box`: each side scaled, the centre shifted, clipped."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    out = set()
    for sw in scales:
        for sh in scales:
            nw, nh = max(16, int(round(w * sw))), max(16, int(round(h * sh)))
            for fx in shifts:
                for fy in shifts:
                    nx = int(round(cx + fx * w - nw / 2.0))
                    ny = int(round(cy + fy * h - nh / 2.0))
                    out.add(ROI("c", box=(nx, ny, nw, nh)).clip(canvas).box)
    return sorted(out)


def _surface(photos, box, reg, cfg) -> Suggestion | None:
    canvas = reg["canvas_size"]
    goods = [p for p in photos if not p.retake and p.label == GOOD]
    bads = [p for p in photos if not p.retake and p.label == BAD]
    gh, gw = goods[0].score_map.shape[:2]
    best, best_key = None, None
    for cand in _around(box, canvas, (0.6, 0.8, 1.0, 1.25),
                        (-0.2, -0.1, 0.0, 0.1, 0.2)):
        gy0, gy1, gx0, gx1 = patch_grid_box(cand, canvas, gh, gw)

        def score(p):
            v = p.valid[gy0:gy1, gx0:gx1]
            if not v.any():
                return float("nan"), 0
            return (sub_mod.aggregate(p.score_map[gy0:gy1, gx0:gx1], v,
                                      cfg.subspace.aggregation), int(v.sum()))

        g = [score(p) for p in goods]
        # Too few patches and the verdict rides on one or two of them.
        if min(n for _, n in g) < MIN_PATCHES:
            continue
        gs = [s for s, _ in g]
        worst = max(gs)
        if not np.isfinite(worst) or worst <= 0:
            continue
        bs = [score(p)[0] for p in bads]
        caught = [(p.name, s) for p, s in zip(bads, bs)
                  if np.isfinite(s) and s > worst * SURFACE_GAP]
        if not caught:
            continue
        weakest = min(s for _, s in caught)
        ratio = weakest / worst
        key = (ratio >= SAFE_RATIO, len(caught), ratio, cand[2] * cand[3])
        if best_key is None or key > best_key:
            best_key = key
            names = {n for n, _ in caught}
            best = Suggestion(
                "surface", cand, float(np.sqrt(worst * weakest)),
                caught=sorted(names),
                missed=sorted(p.name for p in bads if p.name not in names),
                worst_good=worst, weakest_bad=weakest,
                worst_good_name=goods[int(np.argmax(gs))].name)
    return best


def _windows(box, canvas):
    """Small windows tiling the drawn box (plus a margin), for presence."""
    cw, ch = canvas
    x, y, w, h = box
    mx, my = int(0.15 * w), int(0.15 * h)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(cw, x + w + mx), min(ch, y + h + my)
    out = []
    # At least 32 px a side: alignment moves a part by a few pixels shot to
    # shot, which a 16 px window would read as the feature appearing or not.
    # Up to the whole drawn box, so the search can cover all of a feature's
    # edge rather than one corner of it.
    fracs = (1 / 8, 1 / 5, 1 / 3, 1 / 2, 2 / 3, 1.0)
    for fw in fracs:
        for fh in fracs:
            ww, hh = max(32, int(w * fw)), max(32, int(h * fh))
            sx, sy = max(6, ww // 2), max(6, hh // 2)
            for yy in range(y0, max(y0 + 1, y1 - hh + 1), sy):
                for xx in range(x0, max(x0 + 1, x1 - ww + 1), sx):
                    out.append((xx, yy, min(ww, cw - xx), min(hh, ch - yy)))
    return np.array(sorted(set(out)), dtype=np.int64).reshape(-1, 4)


def _coverage(integral, win):
    x0, y0 = win[:, 0], win[:, 1]
    x1, y1 = x0 + win[:, 2], y0 + win[:, 3]
    s = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0]
         + integral[y0, x0]).astype(np.float64)
    return s / np.maximum(win[:, 2] * win[:, 3], 1)


def _presence(photos, box, reg, kind) -> Suggestion | None:
    canvas = reg["canvas_size"]
    goods = [p for p in photos if not p.retake and p.label == GOOD]
    bads = [p for p in photos if not p.retake and p.label == BAD]
    win = _windows(box, canvas)
    if not len(win):
        return None
    ref = _coverage(cv2.integral((reg["ref_mask"] > 0).astype(np.uint8)), win)
    g = np.stack([_coverage(p.integral, win) for p in goods])
    b = np.stack([_coverage(p.integral, win) for p in bads])
    if kind == "must be metal":
        edge = g.min(0)                                  # least metal on a good
        caught = b <= (edge - PRESENCE_GAP)[None, :]
        ok = ref >= 0.9                                  # solid on the reference
    else:
        edge = g.max(0)                                  # most metal on a good
        caught = b >= (edge + PRESENCE_GAP)[None, :]
        ok = ref <= 0.1                                  # open on the reference
    n = caught.sum(0) * ok
    if n.max() == 0:
        return None
    # Most bad parts caught, then the LARGEST window that still separates
    # them by PRESENCE_GAP, then the widest gap. Ranking on the gap first
    # picked a 32 x 50 px corner of 898213-8750's tab: it proves one patch of
    # metal is there, not the tab, and a tab cut short still covers it.
    if kind == "must be metal":
        far = np.where(caught, b, -np.inf).max(0)        # closest caught bad
        gap = edge - far
    else:
        far = np.where(caught, b, np.inf).min(0)
        gap = far - edge
    # A clear gap comes before size, though: from a looser drawn box the
    # largest window that scraped past PRESENCE_GAP had good parts at 76%
    # and a missing tab at 42%, where a smaller one had 58% against 0%.
    area = win[:, 2] * win[:, 3]
    gap = np.where(n > 0, gap, -1)
    order = np.lexsort((gap, area, gap >= CLEAR_GAP, n))
    i = int(order[-1])
    names = {bads[j].name for j in np.nonzero(caught[:, i])[0]}
    worst_i = int(np.argmin(g[:, i]) if kind == "must be metal"
                  else np.argmax(g[:, i]))
    return Suggestion(
        kind, tuple(int(v) for v in win[i]),
        float((edge[i] + far[i]) / 2.0), caught=sorted(names),
        missed=sorted(p.name for p in bads if p.name not in names),
        worst_good=float(edge[i]), weakest_bad=float(far[i]),
        worst_good_name=goods[worst_i].name)


KINDS = ("surface", "hole present", "must be metal", "must be empty")


def _hole(photos, box, reg) -> Suggestion | None:
    """A "hole must be there" box: taller and wider versions of the drawn one.

    Growing the box lets a hole that shifts with the camera stay inside it; a
    box grown onto a neighbouring hole stops catching its own missing hole
    and loses on the count, so the photos keep that in check.
    """
    from .geometry import measure_hole

    canvas = reg["canvas_size"]
    goods = [p for p in photos if p.label == GOOD]
    bads = [p for p in photos if p.label == BAD]
    best, best_key = None, None
    unproven = _hole_unproven(goods, box, reg)
    for cand in _around(box, canvas, (1.0, 1.25, 1.5, 2.0), (0.0,)):
        g = [measure_hole(p.gray, cand) for p in goods]
        edge = min(g)
        if edge < 60:                   # a good part with no hole in the box
            continue
        b = [(p.name, measure_hole(p.gray, cand)) for p in bads]
        caught = [(n, a) for n, a in b if a <= edge * (1.0 - CLEAR_GAP)]
        if not caught:
            continue
        far = max(a for _, a in caught)
        # Larger first on a tie: room for a hole the camera has moved.
        key = (len(caught), round((edge - far) / edge, 2),
               cand[2] * cand[3])
        if best_key is None or key > best_key:
            best_key = key
            names = {n for n, _ in caught}
            best = Suggestion(
                "hole present", cand, (edge + far) / 2.0, caught=sorted(names),
                missed=sorted(p.name for p in bads if p.name not in names),
                worst_good=edge, weakest_bad=far,
                worst_good_name=goods[int(np.argmin(g))].name)
    if best is None and unproven is not None:
        unproven.missed = sorted(p.name for p in bads)
        best = unproven
    if best is not None:
        best.extra = _hole_shape(goods, best.box)
    return best


def _hole_shape(goods, box) -> dict:
    """Shape limits for the hole in `box`, from the good photos."""
    from .geometry import hole_shape_limits, measure_hole_shape

    return hole_shape_limits([measure_hole_shape(p.gray, box)[1]
                              for p in goods])


def _hole_unproven(goods, box, reg) -> Suggestion | None:
    """A hole box that passes every good photo, when no bad photo shows it.

    Without a bad photo nothing can prove a box catches a missing hole, but a
    false alarm on a good photo can still be fixed: grow the box -- mostly in
    height, where a camera off-centre moves the hole -- until the hole sits
    wholly inside it on every good photo. It must hold exactly one hole on
    the reference, so it never reaches a neighbour that would stand in for a
    missing one. On 898399-0830 the drawn boxes cut the top off the holes in
    the one photo taken furthest off-centre.
    """
    from .geometry import count_holes, measure_hole

    canvas = reg["canvas_size"]
    ref = cv2.cvtColor(reg["ref_image"], cv2.COLOR_BGR2GRAY)
    ref_px = measure_hole(ref, box)
    if ref_px < 150 or not goods:
        return None
    best, best_area = None, None
    x, y, w, h = box
    for sw in (1.0, 1.15, 1.3):
        for sh in (1.0, 1.25, 1.5, 1.75, 2.0):
            nw, nh = int(round(w * sw)), int(round(h * sh))
            cand = ROI("c", box=(int(round(x + w / 2 - nw / 2)),
                                 int(round(y + h / 2 - nh / 2)), nw, nh)
                       ).clip(canvas).box
            if count_holes(ref, cand, 0.3 * ref_px) != 1:
                continue
            g = [measure_hole(p.gray, cand) for p in goods]
            med = float(np.median(g))
            # Every good photo shows at least 3/4 of the usual hole: whole,
            # not clipped by the box edge.
            if med < 150 or min(g) < 0.75 * med:
                continue
            area = cand[2] * cand[3]
            if best_area is None or area < best_area:
                best_area = area
                best = Suggestion("hole present", cand, 0.5 * med,
                                  worst_good=min(g),
                                  worst_good_name=goods[int(np.argmin(g))].name)
    return best


def _best(kind, photos, box, reg) -> Suggestion | None:
    if kind == "surface":
        return _surface(photos, box, reg, reg["config"])
    if kind == "hole present":
        return _hole(photos, box, reg)
    return _presence(photos, box, reg, kind)


def _judge(s: Suggestion, ph: Photo, reg: dict) -> bool:
    """Would this suggestion fail this photo?"""
    rep = check_boxes([ph], s.rois("x"), reg, float("inf"))
    return bool(rep["rows"]) and rep["rows"][0][4] == "FAIL"


def _holdout(kind, photos, box, reg) -> tuple:
    """Hide each photo in turn, search on the rest, judge the hidden one.

    A good photo must pass. A bad photo is judged only by a surface box: a
    presence box is meant to catch its own defect (a missing tab), and a bad
    photo with a different defect passing it is not a mistake.
    """
    right, judged, wrong = 0, 0, []
    for i, hidden in enumerate(photos):
        s = _best(kind, photos[:i] + photos[i + 1:], box, reg)
        if s is None:
            continue
        fails = _judge(s, hidden, reg)
        if hidden.label == BAD and kind != "surface":
            if not fails:
                continue
        judged += 1
        ok = fails if hidden.label == BAD else not fails
        right += ok
        if not ok:
            wrong.append(hidden.name)
    return right, judged, tuple(wrong)


def search(photos: list[Photo], box, reg: dict) -> tuple[list[Suggestion], str]:
    """The best surface, must-be-metal and must-be-empty box near `box`.

    Every suggestion passes every usable good photo by construction and says
    which bad photos it catches. An empty list with a reason means no box of
    any kind here separates them -- worth knowing before a test, not after.
    """
    usable = [p for p in photos if not p.retake]
    goods = [p for p in usable if p.label == GOOD]
    bads = [p for p in usable if p.label == BAD]
    if not goods or not bads:
        return [], ("Needs at least one usable good photo and one usable bad "
                    "photo for this part.")
    found = [s for s in (_best(kind, usable, box, reg) for kind in KINDS) if s]
    for s in found:
        s.holdout = _holdout(s.kind, usable, box, reg)
    found.sort(key=lambda s: len(s.caught), reverse=True)
    if not found:
        return [], ("None of your bad photos shows a defect inside this box, "
                    "so no box here can be proven. If this sheet item has no "
                    "bad photo yet, keep the box as drawn (for a hole, set "
                    "its type to **geometry** or **both**) and make sure "
                    "**Check my boxes** passes every good photo. Otherwise "
                    "check the box is around the feature the bad photo "
                    "shows.")
    return found, ""
