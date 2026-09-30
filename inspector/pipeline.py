"""The shared preprocessing path.

Design rule #1: registration and inspection must run an *identical* path --
same segmentation, same alignment, same crop, same resize, same masking. That
is enforced structurally here: this is the only implementation, and both
`register.py` and `inspect.py` call it. There is deliberately no second copy.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import align as align_mod
from . import cancel, inkmask, segment
from .config import PipelineConfig

log = logging.getLogger(__name__)

# iPhones shoot HEIC by default, so parts photographed on a phone arrive in a
# format OpenCV's prebuilt wheels carry no codec for. pillow-heif teaches Pillow
# to open them; `load_image` then falls back to Pillow whenever cv2 declines.
# Optional on purpose: an environment without it keeps working for JPEG/PNG.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:  # noqa: BLE001 - HEIC support is a nice-to-have
    HEIF_OK = False

# Every extension the pipeline can actually decode. Folder scans filter on this
# rather than each keeping its own copy, so adding a format stays a one-line
# change here instead of five edits that drift apart.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
if HEIF_OK:
    IMAGE_SUFFIXES |= {".heic", ".heif"}


@dataclass
class PreprocessResult:
    """Everything the later stages need, plus every intermediate for display."""

    source: str = ""
    image: np.ndarray | None = None          # original BGR
    part_mask: np.ndarray | None = None      # original frame
    ink_mask: np.ndarray | None = None       # original frame
    aligned_image: np.ndarray | None = None  # canonical canvas
    aligned_part_mask: np.ndarray | None = None
    aligned_ink_mask: np.ndarray | None = None
    aligned_valid_mask: np.ndarray | None = None
    transform: np.ndarray | None = None
    ok: bool = True
    reject_reason: str = ""
    # Operator-facing wording in Thai. The station is staffed by Thai-speaking
    # operators, and a retake instruction is only useful if the person holding
    # the camera can act on it.
    reject_reason_th: str = ""
    info: dict = field(default_factory=dict)

    @property
    def valid_mask(self) -> np.ndarray:
        return self.aligned_valid_mask


def load_image(path: str | Path) -> np.ndarray:
    """Read an image, honouring EXIF orientation.

    Phone photos routinely carry a rotation flag; ignoring it feeds the
    pipeline a sideways part and alignment then has to undo a 90-degree error
    it was never designed for.

    `cv2.IMREAD_COLOR` already applies the EXIF orientation -- all eight
    values, mirrors included. Rotating again on top of it is what this
    function used to do, and it turned tagged phone photos sideways or upside
    down. Do not add a second orientation step here.
    """
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)  # unicode-safe on Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        # HEIC and anything else this OpenCV build has no codec for.
        return _load_via_pillow(path)
    return img


def _load_via_pillow(path: str) -> np.ndarray:
    """Decode with Pillow and hand back BGR, the way cv2.imread would.

    `exif_transpose` applies the orientation tag, matching what cv2 does on
    the main path, so both paths hand back an upright image.
    """
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as pil:
            pil = ImageOps.exif_transpose(pil)
            rgb = np.asarray(pil.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 - report as an unreadable image
        suffix = Path(path).suffix.lower()
        if suffix in {".heic", ".heif"} and not HEIF_OK:
            raise ValueError(
                f"could not read image: {path}\n"
                f"HEIC support needs pillow-heif: pip install pillow-heif"
            ) from exc
        raise ValueError(f"could not read image: {path}") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess(
    image: np.ndarray,
    cfg: PipelineConfig,
    ref_stats: dict | None = None,
    ref_mask: np.ndarray | None = None,
    ref_image: np.ndarray | None = None,
    source: str = "",
) -> PreprocessResult:
    """Segment -> ink-mask -> align. The one true path.

    `ref_stats` is None only for the very first golden image, which *defines*
    the canonical frame. Every other call -- golden or test -- reuses the
    saved reference and therefore lands in exactly the same frame.
    """
    # Every photo a fit or a threshold measurement handles passes here first,
    # so this is where a cancel takes effect -- within about one photo.
    cancel.check()
    res = PreprocessResult(source=source, image=image)
    # Seconds per stage, so a slow inspection says where it went instead of
    # only how long it took in total.
    timing = res.info.setdefault("timing", {})
    t = time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timing[name] = timing.get(name, 0.0) + now - t
        t = now

    part = segment.segment_part(image, cfg.segment)
    lap("segment")
    res.part_mask = part
    if part.max() == 0:
        res.ok = False
        res.reject_reason = "no part found in frame"
        res.reject_reason_th = "ไม่พบชิ้นงานในภาพ"
        return res

    res.info.update(segment.mask_stats(part))
    lap("mask stats")
    if segment.touches_border(part, cfg.segment.border_margin_px):
        # Not scored as an anomaly: a clipped part is an invalid photo, and
        # calling it a defect would be the wrong message to the operator.
        res.ok = False
        res.reject_reason = "part is cut off by the frame edge - retake the photo"
        res.reject_reason_th = "จัดชิ้นส่วนให้อยู่ในพื้นที่"

    ink = inkmask.blue_ink_mask(image, cfg.ink)
    res.ink_mask = ink
    res.info["ink_fraction"] = inkmask.ink_fraction(part, ink)
    lap("ink mask")

    if not cfg.align.enabled:
        # Alignment-off comparison mode: pad the original into a fixed canvas
        # so downstream shapes still match, but do not pose-normalise.
        canvas = (ref_stats or {}).get("canvas_size")
        if canvas is None:
            long_side = cfg.align.canvas_long_side or 1024
            h, w = image.shape[:2]
            s = long_side / max(h, w)
            canvas = [int(round(w * s)), int(round(h * s))]
        cw, ch = canvas
        res.aligned_image = cv2.resize(image, (cw, ch), interpolation=cv2.INTER_LINEAR)
        res.aligned_part_mask = cv2.resize(part, (cw, ch), interpolation=cv2.INTER_NEAREST)
        res.aligned_ink_mask = cv2.resize(ink, (cw, ch), interpolation=cv2.INTER_NEAREST)
        res.aligned_valid_mask = inkmask.valid_mask(res.aligned_part_mask,
                                                    res.aligned_ink_mask)
        res.transform = None
        res.info["aligned"] = False
        return res

    seed_angle = None
    if ref_stats is None:
        ref_stats = align_mod.reference_stats(part, cfg.align)
        res.info["defined_reference"] = True
        seed_angle = ref_stats["angle"]
    canvas = ref_stats["canvas_size"]

    M, sim_info = align_mod.similarity_to_canonical(part, ref_stats, ref_mask,
                                                    angle=seed_angle)
    res.info.update(sim_info)
    lap("rotation search")

    aligned_img = align_mod.warp_image(image, M, canvas)
    aligned_part = align_mod.warp_image(part, M, canvas, nearest=True)
    lap("warp")

    # Homography refinement needs something to refine against.
    if ref_image is not None and ref_mask is not None and cfg.align.refine_homography:
        H, ecc_info = align_mod.refine_homography(
            aligned_img, ref_image, aligned_part, ref_mask, cfg.align
        )
        res.info.update(ecc_info)
        lap("fine align (ECC)")
        if ecc_info.get("ecc_ok"):
            M_full = align_mod.compose(M, H)
            aligned_img = align_mod.warp_image(image, M_full, canvas)
            aligned_part = align_mod.warp_image(part, M_full, canvas, nearest=True)
            M = M_full

    aligned_ink = align_mod.warp_image(ink, M, canvas, nearest=True)
    lap("warp")

    res.aligned_image = aligned_img
    res.aligned_part_mask = aligned_part
    res.aligned_ink_mask = aligned_ink
    res.aligned_valid_mask = inkmask.valid_mask(aligned_part, aligned_ink)
    res.transform = M
    res.info["aligned"] = True
    res.info["ref_stats"] = ref_stats

    if ref_mask is not None:
        res.info.update(align_mod.alignment_quality(aligned_part, ref_mask))
        lap("pose check")
        iou = res.info.get("iou", 1.0)
        if iou < cfg.align.min_iou:
            # A different pose, not a defect: scoring it would compare the
            # wrong view patch by patch and fail a good part.
            res.info["pose_rejected"] = True
            if res.ok:
                res.ok = False
                res.reject_reason = (
                    f"part is not lying as on the inspection sheet (outline "
                    f"overlap {iou:.2f}, needs {cfg.align.min_iou:.2f}) - lay "
                    f"it flat as shown and retake")
                res.reject_reason_th = "วางชิ้นงานตามรูป"
    return res


def preprocess_path(path, cfg, **kw) -> PreprocessResult:
    return preprocess(load_image(path), cfg, source=str(path), **kw)


def build_reference(images: list[np.ndarray], cfg: PipelineConfig,
                    sources: list[str] | None = None, refine_passes: int = 1):
    """Establish the canonical frame, then align every golden image into it.

    The first image only *seeds* the frame. Anchoring permanently to one
    arbitrary photo makes the whole SKU inherit that shot's segmentation
    quirks and pose error. So after the first pass the reference is rebuilt as
    the consensus of the aligned set -- median mask, mean image -- and
    everything is re-aligned against that. The consensus is a better target
    than any individual photo, and it is what gets saved with the model.

    Returns (ref_stats, ref_image, ref_mask, [PreprocessResult, ...]).
    """
    sources = sources or [f"image_{i}" for i in range(len(images))]
    if not images:
        raise ValueError("no reference images supplied")

    seed = preprocess(images[0], cfg, source=sources[0])
    if seed.part_mask is None or seed.part_mask.max() == 0:
        raise ValueError(f"could not segment the first reference image ({sources[0]})")

    ref_stats = seed.info.get("ref_stats") or align_mod.reference_stats(
        seed.part_mask, cfg.align
    )
    ref_image = seed.aligned_image
    ref_mask = seed.aligned_part_mask

    results: list[PreprocessResult] = []
    for _ in range(max(refine_passes, 0) + 1):
        results = [
            preprocess(img, cfg, ref_stats=ref_stats, ref_mask=ref_mask,
                       ref_image=ref_image, source=src)
            for img, src in zip(images, sources)
        ]
        # A photo in a different pose would be averaged in as a ghost -- the
        # double outline 898213-8740's reference showed -- so it is left out.
        aligned = [r for r in results if r.aligned_part_mask is not None
                   and not r.info.get("pose_rejected")]
        if len(aligned) < 2:
            break
        # Median silhouette: a pixel belongs to the consensus part if at least
        # half the aligned golden images agree it does.
        stack = np.stack([(r.aligned_part_mask > 0) for r in aligned])
        ref_mask = (stack.mean(0) >= 0.5).astype(np.uint8) * 255
        ref_image = np.mean(
            [r.aligned_image.astype(np.float32) for r in aligned], axis=0
        ).astype(np.uint8)

    return ref_stats, ref_image, ref_mask, results
