"""Central configuration for the inspection pipeline.

Every knob that changes results lives here so experiments and the demo UI can
override them explicitly rather than by editing code.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Literal
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = REPO_ROOT / "outputs"

# Source image folders as laid out in this repo.
GOLDEN_DIR = REPO_ROOT / "normal_images"
TEST_DIR = REPO_ROOT / "test_images"
# Known-bad parts, used only to tune input_size and the verdict threshold.
# Separate from test_images/ on purpose: these are registration inputs with a
# known label, not parts being judged, and mixing the two made it unclear
# which folder a photo belonged in.
NG_DIR = REPO_ROOT / "ng_images"

Backbone = Literal["vits14", "vitb14", "vitl14", "vitg14"]
Segmenter = Literal["birefnet", "classical", "auto"]
Aggregation = Literal["max", "top3mean", "tvar1pct"]


@dataclass
class SegmentConfig:
    """Foreground segmentation."""

    method: Segmenter = "auto"
    birefnet_model: str = "ZhengPeng7/BiRefNet"
    birefnet_fallback: str = "briaai/RMBG-2.0"
    birefnet_input: int = 1024
    # Probability threshold on the BiRefNet soft matte.
    birefnet_threshold: float = 0.5
    # Holes smaller than this fraction of part area are segmentation noise and
    # get filled. Stamped holes in the part are larger and are preserved --
    # they are real geometry the alignment and geometry checks depend on.
    fill_hole_max_frac: float = 0.005
    morph_kernel: int = 5
    # A part touching the image border is not fully in frame; inspection of
    # such a photo is invalid regardless of anomaly score.
    border_margin_px: int = 2
    min_area_frac: float = 0.01


@dataclass
class InkConfig:
    """Blue marker ink exclusion."""

    enabled: bool = True
    # OpenCV HSV ranges: H 0-179, S 0-255, V 0-255.
    # Blue marker on bare/painted metal sits around H=100-130.
    hue_lo: int = 90
    hue_hi: int = 135
    sat_min: int = 60
    val_min: int = 40
    # Blue-channel dominance (B - R) in BGR. This is the cue that actually
    # fires on light-blue pen over bright steel, where saturation is only
    # ~45 and an absolute sat_min high enough to reject the metal misses it.
    blue_dominance: int = 22
    dilate_px: int = 3


@dataclass
class AlignConfig:
    """Canonical pose normalisation."""

    enabled: bool = True
    # Long side of the canonical canvas. Derived per SKU from the golden set
    # when `canvas_long_side` is None.
    canvas_long_side: int | None = 1024
    # Fraction of the canvas the part should occupy along its long axis.
    part_fill_frac: float = 0.82
    refine_homography: bool = True
    # What ECC registers: "mask" (signed distance transform of the silhouette)
    # or "intensity" (masked grayscale). Mask mode keeps the outline and the
    # centroid honest; intensity mode chases texture and lets the part drift.
    ecc_on: Literal["mask", "intensity"] = "mask"
    ecc_iterations: int = 200
    ecc_eps: float = 1e-5
    # Reject an ECC result whose corner displacement exceeds this fraction of
    # the canvas -- a diverged ECC produces wild homographies that silently
    # destroy alignment.
    ecc_max_corner_shift_frac: float = 0.08


@dataclass
class FeatureConfig:
    """Frozen DINOv2 patch features."""

    backbone: Backbone = "vitl14"
    input_size: int = 672
    # Paper Sec 3.2 mean-pools a band of deep layers. Indices are clamped to
    # the actual depth of whichever backbone is selected.
    layers: tuple[int, ...] = (22, 23, 24, 25, 26, 27, 28)
    # A patch counts as valid only if this fraction of its pixels lie inside
    # the valid mask. Partial-coverage boundary patches are a major false
    # positive source and are excluded.
    patch_coverage: float = 0.9
    l2_normalize: bool = False
    device: str | None = None


@dataclass
class SubspaceConfig:
    """PCA normal-subspace model."""

    tau: float = 0.97
    max_components: int = 512
    # Rotation augmentation is OFF by default: alignment handles pose, and
    # random rotations only widen the subspace and hide defects.
    n_aug: int = 0
    aug_max_deg: float = 5.0
    position_normalize: bool = True
    # Floor on the per-position sigma so a position with near-zero variance in
    # the golden set cannot produce an exploding z-score.
    position_sigma_floor: float = 1e-3
    aggregation: Aggregation = "max"
    use_mahalanobis: bool = True
    # Leave-one-out screen at registration: a golden part scoring above this
    # many robust sigmas against a subspace built from the others is rejected
    # as contaminated.
    loo_reject_sigma: float = 4.0
    loo_enabled: bool = True


@dataclass
class GeometryConfig:
    """Mask-contour geometry checks."""

    # Default tolerances, overridable per ROI in the SKU config.
    area_pct: float = 8.0
    fill_ratio_pct: float = 10.0
    corner_angle_deg: float = 12.0


@dataclass
class PipelineConfig:
    """The whole pipeline. One object, passed everywhere."""

    segment: SegmentConfig = field(default_factory=SegmentConfig)
    ink: InkConfig = field(default_factory=InkConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    # Part-level verdict threshold on the aggregated ROI z-score.
    threshold: float = 6.0

    # Scope: this station answers the INSPECTION SHEET, nothing else.
    #
    # A separate inspection station upstream catches general defects -- dents,
    # scratches, damage. So an anomaly outside the sheet's ROIs is not this
    # system's business, and flagging it would only add noise to a queue
    # someone else already handles. Verdicts are therefore PASS, FAIL and
    # RETAKE only.
    #
    # (An earlier REVIEW tier surfaced off-sheet anomalies to a human. It was
    # removed deliberately, not because it failed -- it correctly caught a
    # left-side defect scoring 180 -- but because that defect is the upstream
    # station's job. Narrow scope beats a second opinion nobody asked for.)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            segment=SegmentConfig(**d.get("segment", {})),
            ink=InkConfig(**d.get("ink", {})),
            align=AlignConfig(**d.get("align", {})),
            features=FeatureConfig(**{
                **d.get("features", {}),
                "layers": tuple(d.get("features", {}).get("layers", FeatureConfig.layers)),
            }),
            subspace=SubspaceConfig(**d.get("subspace", {})),
            geometry=GeometryConfig(**d.get("geometry", {})),
            threshold=d.get("threshold", 6.0),
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, **kw) -> "PipelineConfig":
        return replace(self, **kw)


# Signature of the settings that invalidate a fitted subspace. The demo uses
# this to decide whether a refit is required, so that changing only the test
# image never triggers the slow path.
def fit_signature(cfg: PipelineConfig, image_keys: list[str]) -> str:
    relevant = {
        "images": sorted(image_keys),
        "segment": asdict(cfg.segment),
        "ink": asdict(cfg.ink),
        "align": asdict(cfg.align),
        "features": asdict(cfg.features),
        "subspace": asdict(cfg.subspace),
    }
    return json.dumps(relevant, sort_keys=True, default=str)


DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_NAMES = {
    "vits14": "dinov2_vits14",
    "vitb14": "dinov2_vitb14",
    "vitl14": "dinov2_vitl14",
    "vitg14": "dinov2_vitg14",
}
DINOV2_DEPTH = {"vits14": 12, "vitb14": 12, "vitl14": 24, "vitg14": 40}
DINOV2_DIM = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}
PATCH_SIZE = 14
