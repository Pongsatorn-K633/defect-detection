"""Visualisation helpers: overlays, heatmaps and stage contact sheets."""
from __future__ import annotations

import sys

import cv2
import numpy as np


def console_safe(text: str) -> str:
    """Make text printable on whatever encoding the console happens to use.

    Operator messages are written in Thai, and a Windows console running cp1252
    raises UnicodeEncodeError rather than printing them -- which would crash a
    CLI over a status line. Unrepresentable characters degrade to '?'; the
    stored text is untouched.
    """
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode(enc, errors="replace").decode(enc, errors="replace")


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.45):
    out = image.copy()
    layer = np.zeros_like(image)
    layer[mask > 0] = color
    cv2.addWeighted(layer, alpha, out, 1 - alpha, 0, out)
    cont, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cont, -1, color, 2)
    return out


def heatmap_overlay(image, score_map, vmin=None, vmax=None, alpha=0.5):
    """Colour-map a score map onto the image.

    vmin/vmax are supplied by the caller from FIXED thresholds, never derived
    per-image: per-image min-max normalisation is exactly what design rule #3
    forbids, because it makes a clean part look as alarming as a defective one.
    """
    h, w = image.shape[:2]
    s = cv2.resize(score_map.astype(np.float32), (w, h),
                   interpolation=cv2.INTER_LINEAR)
    lo = 0.0 if vmin is None else float(vmin)
    hi = float(vmax) if vmax is not None else float(max(s.max(), lo + 1e-6))
    norm = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(cm, alpha, image, 1 - alpha, 0)


def label(image, text, org=(8, 24), color=(255, 255, 255), scale=0.6):
    out = image.copy()
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)
    return out


def to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def grid(tiles, cols=4, tile_w=320, pad=6, bg=(24, 24, 24)):
    """Contact sheet of (image, caption) pairs."""
    if not tiles:
        return np.zeros((64, 64, 3), np.uint8)
    prepared = []
    for item in tiles:
        img, cap = item if isinstance(item, tuple) else (item, "")
        if img is None:
            continue
        if img.ndim == 2:
            img = to_bgr(img)
        h, w = img.shape[:2]
        th = int(round(tile_w * h / max(w, 1)))
        img = cv2.resize(img, (tile_w, th), interpolation=cv2.INTER_AREA)
        if cap:
            img = label(img, cap)
        prepared.append(img)

    rows = []
    max_h = max(p.shape[0] for p in prepared)
    for i in range(0, len(prepared), cols):
        chunk = prepared[i:i + cols]
        padded = []
        for p in chunk:
            canvas = np.full((max_h, tile_w, 3), bg, np.uint8)
            canvas[:p.shape[0]] = p
            padded.append(canvas)
        while len(padded) < cols:
            padded.append(np.full((max_h, tile_w, 3), bg, np.uint8))
        rows.append(np.hstack([np.pad(p, ((pad, pad), (pad, pad), (0, 0)),
                                      constant_values=0) for p in padded]))
    return np.vstack(rows)


def draw_rois(image, rois, scores=None, threshold=None):
    """Draw ROI boxes, green for pass and red for fail."""
    out = image.copy()
    for roi in rois:
        x, y, w, h = roi["box"]
        score = (scores or {}).get(roi["id"])
        color = (0, 200, 0)
        if score is not None and threshold is not None:
            color = (0, 0, 255) if score > threshold else (0, 200, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cap = roi["id"] if score is None else f"{roi['id']} {score:.1f}"
        out = label(out, cap, (x + 3, max(y - 6, 14)), color, 0.5)
    return out


def stage_sheet(res, title=""):
    """The raw -> mask -> ink -> aligned strip for one image."""
    tiles = [(res.image, f"{title} raw")]
    if res.part_mask is not None:
        tiles.append((overlay_mask(res.image, res.part_mask), "part mask"))
    if res.ink_mask is not None and res.ink_mask.max() > 0:
        tiles.append((overlay_mask(res.image, res.ink_mask, (255, 0, 0)), "ink mask"))
    if res.aligned_image is not None:
        tiles.append((res.aligned_image, "aligned"))
    if res.aligned_valid_mask is not None:
        tiles.append((overlay_mask(res.aligned_image, res.aligned_valid_mask),
                      "valid mask"))
    return grid(tiles, cols=len(tiles))
