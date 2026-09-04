"""Blue marker-ink exclusion.

Operators mark inspected parts with a blue felt pen. The ink position and
shape vary part to part, so unless it is excluded it becomes the dominant
false positive: every ink stroke is, correctly, "not like the golden set".

The final valid mask is `part_mask AND NOT ink_mask`.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import InkConfig


def blue_ink_mask(image: np.ndarray, cfg: InkConfig | None = None) -> np.ndarray:
    """Blue marker ink -> binary mask of pixels to EXCLUDE.

    Two cues, OR-ed:

      * an HSV window, for saturated ink on darker metal;
      * blue-channel dominance (B - R), which is what actually catches the pen
        on these images. A light-blue felt pen on bright bare steel measures
        hue ~99 but saturation only ~45, so an absolute saturation floor high
        enough to reject the metal's own cool cast misses the ink entirely.
        The B-R difference stays strongly positive for ink and hovers around
        -7 for the steel, so it separates cleanly where saturation does not.

    Dilated slightly so anti-aliased stroke edges -- partially ink-coloured
    and just as anomalous -- are covered too.
    """
    cfg = cfg or InkConfig()
    if not cfg.enabled:
        return np.zeros(image.shape[:2], np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg.hue_lo, cfg.sat_min, cfg.val_min], np.uint8)
    upper = np.array([cfg.hue_hi, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    b = image[:, :, 0].astype(np.int16)
    r = image[:, :, 2].astype(np.int16)
    h = hsv[:, :, 0]
    dominance = (
        ((b - r) >= cfg.blue_dominance)
        & (h >= cfg.hue_lo) & (h <= cfg.hue_hi)
        & (hsv[:, :, 2] >= cfg.val_min)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, dominance)

    # Single-pixel JPEG chroma noise is not ink.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    if cfg.dilate_px > 0:
        d = 2 * cfg.dilate_px + 1
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        )
    return mask


def valid_mask(part_mask: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """part AND NOT ink."""
    return cv2.bitwise_and(part_mask, cv2.bitwise_not(ink))


def ink_fraction(part_mask: np.ndarray, ink: np.ndarray) -> float:
    """Share of the part covered by ink -- useful as a sanity readout.

    A very high value usually means the HSV bounds are catching the part's
    own cool-toned specular highlights rather than the pen.
    """
    area = float((part_mask > 0).sum())
    if area == 0:
        return 0.0
    both = cv2.bitwise_and(part_mask, ink)
    return float((both > 0).sum()) / area
