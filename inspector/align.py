"""Canonical pose normalisation.

This is the load-bearing module. If alignment does not hold, every downstream
number is meaningless -- a 3 px pose error moves a patch into a neighbouring
patch cell and the residual explodes on a perfectly good part.

The transform chain is:
    similarity from mask moments (translation + rotation + scale)
      -> 180-degree axis disambiguation against the reference mask
      -> optional ECC homography refinement to absorb mild out-of-plane tilt
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import AlignConfig

log = logging.getLogger(__name__)


def mask_moments(mask: np.ndarray) -> dict:
    """Centroid, principal-axis angle and area from image moments."""
    binary = (mask > 0).astype(np.uint8)
    m = cv2.moments(binary, binaryImage=True)
    if m["m00"] == 0:
        raise ValueError("empty mask - nothing to align")

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]

    # Principal axis of the pixel-coordinate covariance.
    angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)

    cov = np.array([[mu20, mu11], [mu11, mu02]])
    evals = np.linalg.eigvalsh(cov)
    elongation = float(np.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9)))

    return {
        "centroid": (float(cx), float(cy)),
        "angle": float(angle),          # radians
        "area": float(m["m00"]),
        "elongation": elongation,
    }


def reference_stats(mask: np.ndarray, cfg: AlignConfig) -> dict:
    """Canonical-frame parameters, computed once per SKU from golden image 1.

    Every later part of the SKU is mapped into this same frame, so these
    numbers must be saved with the model and never recomputed at inspection.
    """
    mom = mask_moments(mask)
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)

    c = np.array(mom["centroid"], np.float32)
    a = mom["angle"]
    rot = np.array([[np.cos(-a), -np.sin(-a)],
                    [np.sin(-a), np.cos(-a)]], np.float32)
    local = (pts - c) @ rot.T
    ext_x = float(local[:, 0].max() - local[:, 0].min())
    ext_y = float(local[:, 1].max() - local[:, 1].min())
    rob_x, rob_y = _robust_extent(mask, mom["centroid"], a)

    long_side = cfg.canvas_long_side or 1024
    if ext_x >= ext_y:
        canvas_w = long_side
        canvas_h = int(round(long_side * ext_y / max(ext_x, 1e-6)))
    else:
        canvas_h = long_side
        canvas_w = int(round(long_side * ext_x / max(ext_y, 1e-6)))
    # Pad the canvas so the part occupies `part_fill_frac` of it, leaving a
    # margin for pose jitter and for ROIs that sit on the silhouette edge.
    canvas_w = int(round(canvas_w / cfg.part_fill_frac))
    canvas_h = int(round(canvas_h / cfg.part_fill_frac))
    canvas_w += canvas_w % 2
    canvas_h += canvas_h % 2

    target_extent = cfg.part_fill_frac * max(canvas_w, canvas_h)
    scale = target_extent / max(ext_x, ext_y, 1e-6)

    return {
        "area": mom["area"],
        "angle": mom["angle"],
        "canvas_size": [canvas_w, canvas_h],
        "scale": float(scale),
        "extent": [ext_x, ext_y],
        "robust_extent": [rob_x, rob_y],
        "elongation": mom["elongation"],
    }


def _similarity(centroid, angle, scale, canvas_size) -> np.ndarray:
    """2x3 similarity mapping the part to canvas centre at canonical pose."""
    cw, ch = canvas_size
    cx, cy = centroid
    ca, sa = np.cos(-angle), np.sin(-angle)
    s = scale
    # Rotate about the centroid, scale, then translate to canvas centre.
    a11, a12 = s * ca, -s * sa
    a21, a22 = s * sa, s * ca
    tx = cw / 2.0 - (a11 * cx + a12 * cy)
    ty = ch / 2.0 - (a21 * cx + a22 * cy)
    return np.array([[a11, a12, tx], [a21, a22, ty]], np.float32)


def canonical_transform(mask: np.ndarray, ref_stats: dict) -> np.ndarray:
    """2x3 similarity transform mapping this part to the canonical pose.

    centroid -> translation, principal axis -> rotation,
    sqrt(ref_area / this_area) -> scale.
    """
    mom = mask_moments(mask)
    canvas = ref_stats["canvas_size"]
    scale = ref_stats["scale"] * float(np.sqrt(ref_stats["area"] / max(mom["area"], 1e-6)))
    return _similarity(mom["centroid"], mom["angle"], scale, canvas)


def _warp_mask(mask, M, canvas_size) -> np.ndarray:
    cw, ch = canvas_size
    if M.shape[0] == 2:
        return cv2.warpAffine(mask, M, (cw, ch), flags=cv2.INTER_NEAREST)
    return cv2.warpPerspective(mask, M, (cw, ch), flags=cv2.INTER_NEAREST)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a > 0
    b_b = b > 0
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_b, b_b).sum()) / float(union)


def resolve_axis_ambiguity(mask, angle, ref_mask, ref_stats) -> float:
    """The principal axis is 180-degree ambiguous; pick the better orientation.

    Both candidates are warped and scored by IoU against the reference mask.
    Without this the part flips upside down roughly half the time and every
    ROI lands on the wrong feature.
    """
    mom = mask_moments(mask)
    canvas = ref_stats["canvas_size"]

    best_angle, best_iou = angle, -1.0
    for cand in (angle, angle + np.pi):
        scale = _scale_for(mask, ref_stats, mom["centroid"], cand)
        M = _similarity(mom["centroid"], cand, scale, canvas)
        iou = _iou(_warp_mask(mask, M, canvas), ref_mask)
        if iou > best_iou:
            best_iou, best_angle = iou, cand
    return float(best_angle), float(best_iou)


def _robust_extent(mask, centroid, angle, pct=98.0):
    """Extent of the mask along its own axes, measured at a percentile.

    Area is a poor scale cue here: thin protruding flanges are segmented
    inconsistently shot to shot, so mask area swings by 3x on the same part
    and an area-derived scale over-corrects badly. A high-percentile extent
    along the principal axes is far steadier -- it tracks the solid body of
    the part and ignores the few hundred pixels of flange that come and go.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    pts = np.column_stack([xs, ys]).astype(np.float32) - np.asarray(centroid, np.float32)
    ca, sa = np.cos(-angle), np.sin(-angle)
    rot = np.array([[ca, -sa], [sa, ca]], np.float32)
    local = pts @ rot.T
    lo, hi = (100 - pct) / 2.0, 100 - (100 - pct) / 2.0
    ex = float(np.percentile(local[:, 0], hi) - np.percentile(local[:, 0], lo))
    ey = float(np.percentile(local[:, 1], hi) - np.percentile(local[:, 1], lo))
    return max(ex, 1e-6), max(ey, 1e-6)


def _scale_for(mask, ref_stats, centroid, angle):
    ref_ext = ref_stats.get("robust_extent")
    if not ref_ext:
        mom_area = float((mask > 0).sum())
        return ref_stats["scale"] * float(
            np.sqrt(ref_stats["area"] / max(mom_area, 1e-6))
        )
    ex, ey = _robust_extent(mask, centroid, angle)
    # Geometric mean of the two axis ratios: a single inconsistent flange
    # perturbs one axis, and the geometric mean halves its influence rather
    # than letting it set the scale outright.
    ratio = float(np.sqrt((ref_ext[0] / ex) * (ref_ext[1] / ey)))
    return ref_stats["scale"] * ratio


def similarity_to_canonical(mask, ref_stats, ref_mask=None):
    """Full similarity step: moments, then flip disambiguation if we have a
    reference mask to compare against."""
    mom = mask_moments(mask)
    canvas = ref_stats["canvas_size"]
    angle = mom["angle"]
    iou = None
    if ref_mask is not None:
        angle, iou = resolve_axis_ambiguity(mask, angle, ref_mask, ref_stats)
    scale = _scale_for(mask, ref_stats, mom["centroid"], angle)
    M = _similarity(mom["centroid"], angle, scale, canvas)
    return M, {"angle": angle, "scale": scale, "flip_iou": iou,
               "centroid": mom["centroid"], "area": mom["area"]}


def _signed_dt(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the mask boundary, squashed into [-1, 1].

    Inside is positive, outside negative. The tanh squash bounds the range so
    distant background does not dominate the correlation, while keeping a
    smooth gradient near the outline for the optimiser to descend.
    """
    m = (mask > 0).astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    sdf = inside - outside
    return np.tanh(sdf / 20.0).astype(np.float32)


def refine_homography(warped_img, ref_img, warped_mask, ref_mask, cfg: AlignConfig):
    """Polish the similarity into a homography, absorbing mild out-of-plane tilt.

    Runs cv2.findTransformECC (MOTION_HOMOGRAPHY) on masked grayscale. If ECC
    fails to converge -- or converges to something implausible -- the identity
    is returned and the caller keeps the similarity result. A diverged ECC is
    far more damaging than no refinement at all, so the result is validated by
    corner displacement before being accepted.
    """
    eye = np.eye(3, dtype=np.float32)
    if not cfg.refine_homography:
        return eye, {"ecc_ok": False, "reason": "disabled"}

    both = cv2.bitwise_and((ref_mask > 0).astype(np.uint8),
                           (warped_mask > 0).astype(np.uint8))
    if both.sum() < 100:
        return eye, {"ecc_ok": False, "reason": "insufficient overlap"}

    if cfg.ecc_on == "mask":
        # Register the silhouettes, not the pixels. Driving ECC with image
        # intensity lets it trade a better outline match for a better texture
        # match -- IoU improves while the part quietly slides several pixels,
        # which is precisely the error that moves an ROI off its feature.
        # A signed distance transform gives the optimiser a smooth basin
        # around the true outline instead of a binary cliff. The whole canvas
        # participates here, so no ECC mask is supplied.
        g1, g2 = _signed_dt(ref_mask), _signed_dt(warped_mask)
        ecc_mask = None
    else:
        g1 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g2 = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # ECC is sensitive to global illumination; blurring keeps it from
        # locking onto a lighting difference instead of structure.
        g1 = cv2.GaussianBlur(g1, (0, 0), 2.0)
        g2 = cv2.GaussianBlur(g2, (0, 0), 2.0)
        ecc_mask = both

    warp = eye.copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                cfg.ecc_iterations, cfg.ecc_eps)
    try:
        _, warp = cv2.findTransformECC(
            g1, g2, warp, cv2.MOTION_HOMOGRAPHY, criteria, ecc_mask, 5
        )
    except cv2.error as exc:
        log.warning("ECC did not converge (%s); keeping similarity transform",
                    str(exc).splitlines()[-1][:120])
        return eye, {"ecc_ok": False, "reason": "no convergence"}

    h, w = ref_img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    try:
        moved = cv2.perspectiveTransform(corners, np.linalg.inv(warp))
    except np.linalg.LinAlgError:
        return eye, {"ecc_ok": False, "reason": "singular"}
    shift = float(np.abs(moved.reshape(-1, 2) - corners.reshape(-1, 2)).max())
    limit = cfg.ecc_max_corner_shift_frac * max(h, w)
    if not np.isfinite(shift) or shift > limit:
        log.warning("ECC result rejected: corner shift %.1f px exceeds %.1f px",
                    shift, limit)
        return eye, {"ecc_ok": False, "reason": "implausible", "corner_shift": shift}

    return warp.astype(np.float32), {"ecc_ok": True, "corner_shift": shift,
                                     "tilt_px": shift}


def compose(similarity_2x3: np.ndarray, homography_3x3: np.ndarray) -> np.ndarray:
    """Combine the similarity and the ECC refinement into one 3x3."""
    S = np.vstack([similarity_2x3, [0, 0, 1]]).astype(np.float32)
    return (np.linalg.inv(homography_3x3) @ S).astype(np.float32)


def warp_image(image, M, canvas_size, nearest=False):
    cw, ch = canvas_size
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if M.shape[0] == 2:
        return cv2.warpAffine(image, M, (cw, ch), flags=flags)
    return cv2.warpPerspective(image, M, (cw, ch), flags=flags)


def alignment_quality(warped_mask: np.ndarray, ref_mask: np.ndarray) -> dict:
    """IoU and centroid drift against the reference -- the Experiment 1 metrics."""
    iou = _iou(warped_mask, ref_mask)
    try:
        a = mask_moments(warped_mask)["centroid"]
        b = mask_moments(ref_mask)["centroid"]
        drift = float(np.hypot(a[0] - b[0], a[1] - b[1]))
    except ValueError:
        drift = float("nan")
    return {"iou": iou, "centroid_drift_px": drift}
