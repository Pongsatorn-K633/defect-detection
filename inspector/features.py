"""Frozen DINOv2 patch features.

Nothing here is trained or fine-tuned. The backbone is loaded once, put in
eval mode, and used purely as a fixed feature extractor.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import (DINOV2_DEPTH, DINOV2_DIM, DINOV2_NAMES, DINOV2_REPO,
                     PATCH_SIZE, FeatureConfig)

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}
_CACHE_LOCK = threading.Lock()

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def get_device(cfg: FeatureConfig) -> str:
    if cfg.device:
        return cfg.device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(cfg: FeatureConfig):
    """Load and cache a frozen DINOv2 backbone."""
    device = get_device(cfg)
    key = f"{cfg.backbone}::{device}"
    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        import torch

        name = DINOV2_NAMES[cfg.backbone]
        # Prefer the already-cloned hub repo. `torch.hub.load` given a GitHub
        # spec contacts github.com to resolve the repo *before* it touches the
        # cached weights, so a machine holding everything on disk still waits
        # on the network at every process start -- and when GitHub is
        # unreachable it waits for the timeout before falling back. Pointing at
        # the clone skips that; the fallback covers a first run, where there is
        # no clone yet and fetching is the whole point.
        repo_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        try:
            if not repo_dir.is_dir():
                raise FileNotFoundError(repo_dir)
            model = torch.hub.load(str(repo_dir), name, source="local",
                                   verbose=False)
        except Exception as exc:  # noqa: BLE001 - fall back to fetching it
            log.info("Local hub repo unusable (%s); fetching %s",
                     exc, DINOV2_REPO)
            model = torch.hub.load(DINOV2_REPO, name, verbose=False)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[key] = (model, device)
        log.info("Loaded %s on %s", name, device)
        return _MODEL_CACHE[key]


def resolve_layers(cfg: FeatureConfig) -> list[int]:
    """Clamp the configured layer band to the backbone's actual depth.

    The spec's default band (22..28) is sized for ViT-L/ViT-G. On ViT-S/B,
    which are only 12 blocks deep, it would index past the end -- so the band
    is remapped to the equivalent relative depth instead of erroring out.
    """
    depth = DINOV2_DEPTH[cfg.backbone]
    layers = sorted({int(l) for l in cfg.layers})
    if max(layers) < depth:
        return layers
    ref_depth = 28  # the depth the default band was written against
    scaled = {min(depth - 1, max(0, int(round(l * depth / ref_depth))))
              for l in layers}
    out = sorted(scaled)
    log.info("Layer band %s remapped to %s for %s (depth %d)",
             layers, out, cfg.backbone, depth)
    return out


def preprocess_for_backbone(image: np.ndarray, cfg: FeatureConfig):
    """Resize so both sides are multiples of the patch size, then normalise."""
    size = cfg.input_size
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh = max(PATCH_SIZE, int(round(h * scale / PATCH_SIZE)) * PATCH_SIZE)
    nw = max(PATCH_SIZE, int(round(w * scale / PATCH_SIZE)) * PATCH_SIZE)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return norm.transpose(2, 0, 1), (nh // PATCH_SIZE, nw // PATCH_SIZE)


def extract_patch_features(image: np.ndarray,
                           cfg: FeatureConfig | None = None) -> np.ndarray:
    """Frozen DINOv2 patch features, mean-pooled across a band of deep layers.

    Returns an (h, w, D) map, where h and w are the patch-grid dimensions.
    """
    cfg = cfg or FeatureConfig()
    import torch

    model, device = load_backbone(cfg)
    arr, (gh, gw) = preprocess_for_backbone(image, cfg)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)

    layers = resolve_layers(cfg)
    with torch.no_grad():
        outs = model.get_intermediate_layers(x, n=layers, reshape=False,
                                             return_class_token=False)
        # Mean-pool the token maps across the selected layers (paper Sec 3.2).
        stacked = torch.stack([o for o in outs], dim=0).mean(dim=0)

    feats = stacked.squeeze(0).float().cpu().numpy()   # (gh*gw, D)
    if feats.shape[0] != gh * gw:
        raise RuntimeError(
            f"token count {feats.shape[0]} != grid {gh}x{gw}={gh * gw}"
        )
    fmap = feats.reshape(gh, gw, -1)
    if cfg.l2_normalize:
        fmap /= np.linalg.norm(fmap, axis=-1, keepdims=True) + 1e-8
    return fmap


def patch_mask_from_pixel_mask(mask: np.ndarray, h: int, w: int,
                               coverage: float = 0.9) -> np.ndarray:
    """Downsample a pixel mask onto the patch grid.

    A patch is valid only if at least `coverage` of its pixels lie inside the
    mask. Partial-coverage boundary patches are EXCLUDED: they mix part and
    background, vary with every pose jitter, and are a major false positive
    source -- greying out the background is not enough, the patches have to be
    dropped from both the fit and the scoring.
    """
    frac = cv2.resize(
        (mask > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA
    )
    return frac >= coverage


def feature_dim(cfg: FeatureConfig) -> int:
    return DINOV2_DIM[cfg.backbone]
