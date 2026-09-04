"""Foreground segmentation of the part against the workbench.

Two interchangeable backends:
  * `birefnet`  -- frozen zero-shot BiRefNet (or RMBG-2.0) via HuggingFace.
  * `classical` -- brightness/saturation segmentation exploiting the dark
                   workbench. No download, no GPU, and on these images it is
                   competitive with the deep model.

`auto` prefers BiRefNet and silently falls back to classical if the weights
are not available offline.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from .config import SegmentConfig

log = logging.getLogger(__name__)

# Module-level cache. Loading BiRefNet costs seconds; never do it per image.
_MODEL_CACHE: dict[str, object] = {}
_CACHE_LOCK = threading.Lock()


class SegmentationError(RuntimeError):
    pass


def _load_birefnet(cfg: SegmentConfig):
    key = f"birefnet::{cfg.birefnet_model}"
    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        import torch
        from transformers import AutoModelForImageSegmentation

        last_err: Exception | None = None
        for name in (cfg.birefnet_model, cfg.birefnet_fallback):
            try:
                model = AutoModelForImageSegmentation.from_pretrained(
                    name, trust_remote_code=True
                )
                model.eval()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                if device == "cuda":
                    model.half()
                _MODEL_CACHE[key] = (model, device)
                log.info("Loaded segmentation model %s on %s", name, device)
                return _MODEL_CACHE[key]
            except Exception as exc:  # noqa: BLE001 - report all load failures
                log.warning("Could not load %s: %s", name, exc)
                last_err = exc
        raise SegmentationError(f"No BiRefNet-class model available: {last_err}")


def birefnet_available(cfg: SegmentConfig) -> bool:
    try:
        _load_birefnet(cfg)
        return True
    except Exception:  # noqa: BLE001
        return False


def _birefnet_matte(image: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """Soft foreground probability in [0, 1] at the image's own resolution."""
    import torch
    from torchvision import transforms

    model, device = _load_birefnet(cfg)
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.birefnet_input, cfg.birefnet_input),
                          antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(rgb).unsqueeze(0).to(device)
    if device == "cuda":
        x = x.half()

    with torch.no_grad():
        out = model(x)
    # BiRefNet returns a list of multi-scale logits; the last is the finest.
    if isinstance(out, (list, tuple)):
        out = out[-1]
    if isinstance(out, (list, tuple)):
        out = out[-1]
    pred = out.sigmoid().float().squeeze().cpu().numpy()
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)


def _classical_matte(image: np.ndarray) -> np.ndarray:
    """Bright-part-on-dark-bench segmentation.

    The parts are bare or light-painted steel and the bench is near-black
    textured wood, so value separates them well -- but not by brightness
    alone. Deep-drawn channels and the shadowed recess around the stud are as
    dark as the bench, and a pure Otsu threshold punches a large hole through
    the middle of the part. Two extra cues fix that:

      * texture -- the bench is high-frequency grain, the part is smooth even
        where it is dark, so local gradient energy separates them;
      * convexity -- the interior of the outer silhouette belongs to the part
        regardless of how dark a given pixel is.

    The result is a mask whose *area is stable* across shots, which matters
    more than per-pixel prettiness: alignment scale comes straight from area.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = cv2.bilateralFilter(hsv[:, :, 2], 9, 60, 60)
    s = hsv[:, :, 1]

    otsu, bright = cv2.threshold(v, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Steel is bright and desaturated; coloured clutter that passes the value
    # test (green tape, blue crates) is suppressed here.
    bright = cv2.bitwise_and(bright, (s < 90).astype(np.uint8) * 255)
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    if bright.sum() == 0:
        return np.zeros(v.shape, np.float32)

    # Model the *background* rather than the part. The bench is a consistent
    # dark grain; the part contains both bright faces and deep shadowed
    # channels, so no single description of "part" holds. Estimating the bench
    # from the image border -- which is reliably bench -- and then keeping
    # everything sufficiently unlike it recovers the dark recesses that a
    # brightness threshold alone punches straight through.
    h, w = v.shape
    border = np.ones((h, w), bool)
    m = int(round(0.06 * min(h, w))) or 1
    border[m:-m, m:-m] = False
    border &= bright == 0
    if border.sum() < 100:
        bg_mu, bg_sigma = 0.0, 1.0
    else:
        vals = v[border].astype(np.float32)
        bg_mu = float(np.median(vals))
        # Robust spread: the bench grain is heavy-tailed, so MAD beats std.
        bg_sigma = float(np.median(np.abs(vals - bg_mu))) * 1.4826
    bg_sigma = max(bg_sigma, 3.0)

    unlike_bg = ((v.astype(np.float32) - bg_mu) > 2.5 * bg_sigma)
    unlike_bg = (unlike_bg.astype(np.uint8)) * 255

    # Bright core, grown into adjacent not-background pixels. The grow is
    # constrained to the flood region so it stops at the true silhouette
    # instead of leaking across the bench.
    seed = bright
    grow = cv2.bitwise_or(seed, unlike_bg)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grow, 8)
    if n <= 1:
        return bright.astype(np.float32) / 255.0
    # Choose the component holding most of the bright seed, not merely the
    # biggest: a large dim bench region can otherwise win.
    best, best_overlap = 1, -1
    for i in range(1, n):
        overlap = int(((labels == i) & (seed > 0)).sum())
        if overlap > best_overlap:
            best, best_overlap = i, overlap
    main = ((labels == best).astype(np.uint8)) * 255

    # Close across the dark channels with a kernel scaled to the part's size,
    # bridging the recesses without swallowing the surrounding bench.
    span = max(stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT])
    k = max(5, int(round(span * 0.05)) | 1)
    closed = cv2.morphologyEx(
        main, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    return closed.astype(np.float32) / 255.0


def _largest_component(mask: np.ndarray, min_area: float) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if areas.max() < min_area:
        return np.zeros_like(mask)
    return ((labels == idx).astype(np.uint8)) * 255


def _fill_small_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    """Fill segmentation speckle but preserve real stamped holes.

    Stamped holes are large (well above 0.5% of part area) and are genuine
    geometry that the alignment and geometry checks rely on. Only sub-threshold
    holes -- specular glints that punched through the matte -- are filled.
    """
    part_area = float((mask > 0).sum())
    if part_area == 0:
        return mask
    inv = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = mask.copy()
    max_area = max_frac * part_area
    h, w = mask.shape
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches_border = x == 0 or y == 0 or x + cw >= w or y + ch >= h
        if touches_border:
            continue  # this is the outside background, not a hole
        if area < max_area:
            out[labels == i] = 255
    return out


def segment_part(image: np.ndarray, cfg: SegmentConfig | None = None) -> np.ndarray:
    """Foreground segmentation -> binary mask (uint8, 0/255).

    Keeps the largest connected component only, fills holes smaller than
    `fill_hole_max_frac` of part area (segmentation noise, not stamped holes),
    and morphologically cleans the edges.
    """
    cfg = cfg or SegmentConfig()
    if image is None or image.size == 0:
        raise SegmentationError("empty image")

    method = cfg.method
    if method == "auto":
        method = "birefnet" if birefnet_available(cfg) else "classical"

    if method == "birefnet":
        matte = _birefnet_matte(image, cfg)
    elif method == "classical":
        matte = _classical_matte(image)
    else:
        raise SegmentationError(f"unknown segmentation method {method!r}")

    mask = (matte >= cfg.birefnet_threshold).astype(np.uint8) * 255

    k = cfg.morph_kernel
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    min_area = cfg.min_area_frac * image.shape[0] * image.shape[1]
    mask = _largest_component(mask, min_area)
    mask = _fill_small_holes(mask, cfg.fill_hole_max_frac)
    return mask


def touches_border(mask: np.ndarray, margin: int = 2,
                   min_run_frac: float = 0.04) -> bool:
    """True if the part is meaningfully clipped by the frame edge.

    A clipped photo cannot be inspected: the missing region is
    indistinguishable from a missing feature, and the alignment moments are
    biased by whatever was cut off. So it is a hard reject rather than an
    anomaly score.

    But "any pixel on the border" is too strict. A part can graze the edge by
    a few pixels -- or have a legitimately protruding feature reach it --
    while still being fully visible, and rejecting those sends the operator to
    retake a perfectly usable photo. The test is therefore how *much* of a
    border is covered: a genuine cut-off leaves a long run of part pixels
    along that edge, a graze leaves a handful.
    """
    if mask.max() == 0:
        return False
    m = max(1, margin)
    h, w = mask.shape
    # (band, axis to collapse, length of that edge)
    edges = (
        (mask[:m, :], 0, w),      # top
        (mask[-m:, :], 0, w),     # bottom
        (mask[:, :m], 1, h),      # left
        (mask[:, -m:], 1, h),     # right
    )
    for band, axis, length in edges:
        covered = int((band > 0).any(axis=axis).sum())
        if covered >= max(3, int(min_run_frac * length)):
            return True
    return False


def mask_stats(mask: np.ndarray) -> dict:
    area = float((mask > 0).sum())
    if area == 0:
        return {"area": 0.0, "centroid": (0.0, 0.0), "bbox": (0, 0, 0, 0)}
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    ys, xs = np.nonzero(mask)
    return {
        "area": area,
        "centroid": (float(cx), float(cy)),
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    }
