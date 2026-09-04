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
