"""Mask-contour geometry checks.

Several checks on the paper inspection sheets are shape questions, not
appearance questions -- "the top-left corner is chamfered", "the tab is
present". Measuring the aligned silhouette answers those directly, and does it
more reliably than asking an anomaly model: a missing tab is exactly the
"logical anomaly" case that SubspaceAD is documented to be weak on, because
the absent metal reads as ordinary background texture.
"""
from __future__ import annotations

import cv2
import numpy as np


def measure_roi_geometry(mask: np.ndarray, roi_box) -> dict:
    """Area, contour length, corner angles and fill ratio inside an ROI."""
    x, y, w, h = [int(v) for v in roi_box]
    x = max(0, x)
    y = max(0, y)
    crop = mask[y:y + h, x:x + w]
    if crop.size == 0:
        return {"area": 0.0, "area_frac": 0.0, "perimeter": 0.0,
                "fill_ratio": 0.0, "corner_angle": float("nan"),
                "n_components": 0, "solidity": 0.0}

    binary = (crop > 0).astype(np.uint8)
    area = float(binary.sum())
    box_area = float(binary.size)

    cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in cont))

    solidity = 0.0
    corner_angle = float("nan")
    if cont:
        biggest = max(cont, key=cv2.contourArea)
        hull = cv2.convexHull(biggest)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            solidity = float(cv2.contourArea(biggest)) / hull_area
        corner_angle = _dominant_corner_angle(biggest)

    return {
        "area": area,
        # Normalised by the ROI box so the measure is scale-free and can be
        # compared against a nominal captured at registration.
        "area_frac": area / box_area,
        "perimeter": perimeter,
        "fill_ratio": area / box_area,
        "corner_angle": corner_angle,
        "n_components": len(cont),
        "solidity": solidity,
    }


def _dominant_corner_angle(contour: np.ndarray) -> float:
    """Sharpest interior angle on the simplified contour, in degrees.

    A chamfered corner replaces one ~90-degree vertex with two shallower ones,
    so the sharpest angle in the ROI separates chamfered from square.
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True).reshape(-1, 2)
    if len(approx) < 3:
        return float("nan")

    best = 180.0
    n = len(approx)
    for i in range(n):
        a = approx[(i - 1) % n].astype(np.float64)
        b = approx[i].astype(np.float64)
        c = approx[(i + 1) % n].astype(np.float64)
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        best = min(best, float(np.degrees(np.arccos(cosang))))
    return best


def fit_nominal(measurements: list[dict]) -> dict:
    """Nominal geometry for an ROI: median and spread over the golden set."""
    keys = ["area", "area_frac", "perimeter", "fill_ratio", "corner_angle",
            "solidity"]
    nominal = {}
    for k in keys:
        vals = np.array([m[k] for m in measurements], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            nominal[k] = float("nan")
            nominal[f"{k}_std"] = float("nan")
            continue
        nominal[k] = float(np.median(vals))
        # Robust spread; a handful of golden parts makes the plain std fragile.
        nominal[f"{k}_std"] = float(np.median(np.abs(vals - nominal[k]))) * 1.4826
    nominal["n"] = len(measurements)
    return nominal


def check_geometry(measured: dict, nominal: dict, tolerance: dict) -> bool:
    """True if the measurement is within tolerance of nominal."""
    return not geometry_violations(measured, nominal, tolerance)


def geometry_violations(measured: dict, nominal: dict,
                        tolerance: dict) -> list[str]:
    """Which tolerances a measurement breaks, and by how much."""
    out = []

    area_pct = tolerance.get("area_pct")
    if area_pct is not None and np.isfinite(nominal.get("area_frac", np.nan)):
        nom = nominal["area_frac"]
        if nom > 0:
            delta = 100.0 * abs(measured["area_frac"] - nom) / nom
            if delta > area_pct:
                out.append(f"area off nominal by {delta:.1f}% (limit {area_pct}%)")

    fill_pct = tolerance.get("fill_ratio_pct")
    if fill_pct is not None and np.isfinite(nominal.get("fill_ratio", np.nan)):
        nom = nominal["fill_ratio"]
        if nom > 0:
            delta = 100.0 * abs(measured["fill_ratio"] - nom) / nom
            if delta > fill_pct:
                out.append(f"fill ratio off by {delta:.1f}% (limit {fill_pct}%)")

    ang = tolerance.get("corner_angle_deg")
    if ang is not None and np.isfinite(measured.get("corner_angle", np.nan)) \
            and np.isfinite(nominal.get("corner_angle", np.nan)):
        delta = abs(measured["corner_angle"] - nominal["corner_angle"])
        if delta > ang:
            out.append(f"corner angle off by {delta:.1f} deg (limit {ang})")

    return out


PRESENCE_KEYS = ("min_area_frac", "max_area_frac", "min_hole_px")
HOLE_KEY = "min_hole_px"
# Below this spread of grey inside the box there is nothing dark to call a
# hole: a missing hole leaves the box all one metal.
HOLE_MIN_CONTRAST = 40.0


def is_hole(tolerance: dict | None) -> bool:
    """A "the hole must be there" check."""
    return bool(tolerance) and HOLE_KEY in tolerance


def measure_hole(gray: np.ndarray, roi_box) -> float:
    """Area in px of the largest dark region fully enclosed inside the box.

    A punched through-hole shows the table through it: dark, and ringed by
    metal on every side. Segmentation cannot be trusted with it -- on
    898399-0830 BiRefNet filled both small holes in every photo and the big
    ones in some -- and a fixed spot cannot either: the hole row there is a
    raised strip, and a camera slightly off-centre moved it 40 px against the
    outline in three of sixteen photos. So the hole is found by brightness,
    anywhere inside the box. Dark regions that touch the box edge are the
    gaps and background around the part, not a hole, and are ignored.
    """
    return measure_hole_shape(gray, roi_box)[0]


def measure_hole_shape(gray: np.ndarray, roi_box) -> tuple[float, float]:
    """(area px, elongation) of the largest enclosed dark hole in the box.

    Elongation is the ratio of the hole's long axis to its short one: 1.0 is
    round. On 898399-0830 the oval hole measured 1.24-1.30 on every good
    photo and the round ones 1.01-1.11, while a round hole punched where the
    oval belongs measured 1.04 -- the same area, so only the shape tells.
    NaN when there is no hole.
    """
    x, y, w, h = [int(v) for v in roi_box]
    crop = gray[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return 0.0, float("nan")
    lo, hi = np.percentile(crop, 5), np.percentile(crop, 90)
    if hi - lo < HOLE_MIN_CONTRAST:
        return 0.0, float("nan")
    dark = (crop < (lo + hi) / 2.0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    ch, cw = crop.shape[:2]
    best = None
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        if x0 == 0 or y0 == 0 or x0 + bw >= cw or y0 + bh >= ch:
            continue                                  # touches the edge
        if best is None or area > stats[best, 4]:
            best = i
    if best is None:
        return 0.0, float("nan")
    area = float(stats[best, 4])
    if area < 20:
        return area, float("nan")
    ys, xs = np.nonzero(lab == best)
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(float))))
    return area, float(np.sqrt(ev[1] / max(ev[0], 1e-9)))


# A round hole reads 1.01-1.11 once aligned and photographed; an oval one
# 1.24 and up. Checks on shape sit between the two.
ROUND_MAX_ELONG = 1.18


def hole_shape_limits(elongations) -> dict:
    """Shape limits for a hole check from the good photos' elongations.

    An oval hole must stay oval (a round one punched in its place fails) and
    a round hole must stay round. One-sided on purpose: the photo taken
    furthest off-centre stretched the oval to 1.50, and that is still an oval.
    """
    e = np.asarray([v for v in elongations if np.isfinite(v)], float)
    if not len(e):
        return {}
    med = float(np.median(e))
    if med >= 1.15:
        return {"hole_elong": round(med, 2),
                "min_elong": round((med + 1.05) / 2.0, 2)}
    return {"hole_elong": round(med, 2), "max_elong": ROUND_MAX_ELONG}


def count_holes(gray: np.ndarray, roi_box, min_px: float) -> int:
    """How many enclosed dark holes of at least `min_px` the box holds."""
    x, y, w, h = [int(v) for v in roi_box]
    crop = gray[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return 0
    lo, hi = np.percentile(crop, 5), np.percentile(crop, 90)
    if hi - lo < HOLE_MIN_CONTRAST:
        return 0
    dark = (crop < (lo + hi) / 2.0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    ch, cw = crop.shape[:2]
    return sum(1 for i in range(1, n)
               if stats[i, 4] >= min_px and stats[i, 0] > 0 and stats[i, 1] > 0
               and stats[i, 0] + stats[i, 2] < cw
               and stats[i, 1] + stats[i, 3] < ch)


def hole_result(area: float, tolerance: dict,
                elong: float = float("nan")) -> tuple[float, list[str]]:
    """Score and violations for a hole check; 1.0 is exactly at the limit.

    Size first -- a missing hole has no shape -- then shape, where the check
    carries shape limits (`hole_shape_limits`).
    """
    limit = float(tolerance[HOLE_KEY])
    nominal = float(tolerance.get("hole_px", 2.0 * limit))
    score = max(0.0, nominal - area) / max(nominal - limit, 1.0)
    if area < limit:
        return score, [f"hole missing or too small: {area:.0f} px open, needs "
                       f"at least {limit:.0f} px"]
    if not np.isfinite(elong):
        return score, []
    if "min_elong" in tolerance:
        lo = float(tolerance["min_elong"])
        nom = float(tolerance.get("hole_elong", lo + 0.1))
        score = max(score, (nom - elong) / max(nom - lo, 1e-3))
        if elong < lo:
            return score, [f"hole is round, should be oval (shape {elong:.2f}, "
                           f"needs at least {lo:.2f})"]
    if "max_elong" in tolerance:
        hi = float(tolerance["max_elong"])
        score = max(score, (elong - 1.0) / max(hi - 1.0, 1e-3))
        if elong > hi:
            return score, [f"hole is oval, should be round (shape {elong:.2f},"
                           f" at most {hi:.2f})"]
    return score, []


def is_presence(tolerance: dict | None) -> bool:
    """A presence check: "this spot must be metal" or "must be empty".

    Some sheet items are a feature that is there or not -- a tab welded over a
    window, a hole that must stay open. The outline measures in
    `geometry_violations` compare against a nominal with a relative tolerance,
    which cannot express "empty" at all (a nominal of 0 has no percentage) and
    reads a box that is solid metal on every good part as a check that can
    never fire. A presence check is an absolute limit on how much of the box
    is part, so it needs no nominal and works on solid or open spots alike.
    On 898213-8750 a missing tab left the spot 0% metal against 73-100% on
    every good part.
    """
    return bool(tolerance) and any(k in tolerance for k in PRESENCE_KEYS)


def presence_result(measured: dict, tolerance: dict) -> tuple[float, list[str]]:
    """Score and violations for a presence check; 1.0 is exactly at the limit.

    The score runs from 0 on a spot exactly as it should be to above 1.0 when
    it breaks the limit, so it reads on the same axis as the other shape
    checks.
    """
    frac = float(measured.get("area_frac", 0.0))
    if HOLE_KEY in tolerance:
        return hole_result(float(measured.get("hole_px", 0.0)), tolerance,
                           float(measured.get("hole_elong", float("nan"))))
    if "min_area_frac" in tolerance:
        limit = float(np.clip(tolerance["min_area_frac"], 0.05, 0.95))
        score = (1.0 - frac) / (1.0 - limit)
        bad = frac < limit
        why = (f"only {100 * frac:.0f}% metal, must be at least "
               f"{100 * limit:.0f}% -- feature missing")
    else:
        limit = float(np.clip(tolerance["max_area_frac"], 0.05, 0.95))
        score = frac / limit
        bad = frac > limit
        why = (f"{100 * frac:.0f}% metal, must be at most "
               f"{100 * limit:.0f}% -- something is there that should not be")
    return float(score), ([why] if bad else [])


def geometry_score(measured: dict, nominal: dict, tolerance: dict) -> float:
    """Worst tolerance usage as a ratio: 1.0 is exactly at the limit.

    Scaled onto the appearance score's axis so a single part-level threshold
    covers both check types.
    """
    ratios = []

    area_pct = tolerance.get("area_pct")
    if area_pct and np.isfinite(nominal.get("area_frac", np.nan)) \
            and nominal["area_frac"] > 0:
        delta = 100.0 * abs(measured["area_frac"] - nominal["area_frac"]) \
            / nominal["area_frac"]
        ratios.append(delta / area_pct)

    fill_pct = tolerance.get("fill_ratio_pct")
    if fill_pct and np.isfinite(nominal.get("fill_ratio", np.nan)) \
            and nominal["fill_ratio"] > 0:
        delta = 100.0 * abs(measured["fill_ratio"] - nominal["fill_ratio"]) \
            / nominal["fill_ratio"]
        ratios.append(delta / fill_pct)

    ang = tolerance.get("corner_angle_deg")
    if ang and np.isfinite(measured.get("corner_angle", np.nan)) \
            and np.isfinite(nominal.get("corner_angle", np.nan)):
        ratios.append(abs(measured["corner_angle"] - nominal["corner_angle"]) / ang)

    return float(max(ratios)) if ratios else 0.0
