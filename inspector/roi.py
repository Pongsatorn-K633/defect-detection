"""SKU configuration: ROI definitions and on-disk model layout."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import MODEL_DIR

WHOLE_PART = "whole_part"

# One box drawn by the operator is one *check*, stored as up to two ROIs over
# the same box: `<name>` scores the surface (appearance) and `<name>_geom` the
# outline (geometry). They catch opposite failures -- a damaged feature versus
# a missing or bent one -- but the person drawing the box should not have to
# know that, so the editor and the result view show one box per check.
#
# Both use the box exactly as drawn. Padding the geometry box was measured on
# SKU 8983864641 and only hurt: at +20% a missing wing scored 0.52 (passes)
# instead of 1.19, and on the top-left corner a good part scored 4.61.
GEOM_SUFFIX = "_geom"
DEFAULT_TOLERANCE = {"area_pct": 10.0, "fill_ratio_pct": 12.0,
                     "corner_angle_deg": 15.0}


@dataclass
class ROI:
    id: str
    type: str = "appearance"          # "appearance" | "geometry"
    box: tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h on the canvas
    threshold: float | None = None    # None -> use the SKU default
    tolerance: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type, "box": list(self.box)}
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.tolerance:
            d["tolerance"] = self.tolerance
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ROI":
        return cls(
            id=d["id"],
            type=d.get("type", "appearance"),
            box=tuple(int(v) for v in d["box"]),
            threshold=d.get("threshold"),
            tolerance=d.get("tolerance", {}),
            note=d.get("note", ""),
        )

    def clip(self, canvas_size) -> "ROI":
        cw, ch = canvas_size
        x, y, w, h = self.box
        x, y = max(0, min(int(x), cw - 1)), max(0, min(int(y), ch - 1))
        w, h = max(1, min(int(w), cw - x)), max(1, min(int(h), ch - y))
        return ROI(self.id, self.type, (x, y, w, h), self.threshold,
                   self.tolerance, self.note)


@dataclass
class SKUConfig:
    sku: str
    canvas_size: tuple[int, int] = (1024, 768)
    ref_area_px: float = 0.0
    rois: list[ROI] = field(default_factory=list)
    threshold: float = 6.0
    # What the threshold was measured against, and on which fit.
    #
    # A threshold is not a preference: it is the gap between the worst passing
    # part and the best failing one, and both of those scores exist only
    # relative to one fitted model. Re-fit at a different resolution and the
    # same parts score differently -- on SKU 8983864641 the usable window moved
    # from 10.95-13.14 at input_size 672 to 18.93-21.72 at 896. A threshold
    # carried across that change silently fails good parts.
    #
    # Recording the grid it was measured on makes the staleness detectable
    # instead of silent; `threshold_stale_for` does the comparison.
    threshold_grid: list | None = None      # [h, w] patch grid at measurement
    threshold_worst_ok: float | None = None
    threshold_best_ng: float | None = None
    # Per-SKU DINOv2 input resolution, overriding the global default. Some
    # checks are simply under-resolved at 672: if an ROI holds only ~16
    # patches, the verdict rides on the single worst one and a good part can
    # trip it. Raising the resolution for that SKU buys real patches instead
    # of a luckier draw. None means "use the pipeline default".
    input_size: int | None = None

    def to_dict(self) -> dict:
        d = {
            "sku": self.sku,
            "canvas_size": list(self.canvas_size),
            "ref_area_px": self.ref_area_px,
            "threshold": self.threshold,
            "rois": [r.to_dict() for r in self.rois],
        }
        if self.input_size is not None:
            d["input_size"] = self.input_size
        if self.threshold_grid is not None:
            d["threshold_grid"] = list(self.threshold_grid)
        if self.threshold_worst_ok is not None:
            d["threshold_worst_ok"] = self.threshold_worst_ok
        if self.threshold_best_ng is not None:
            d["threshold_best_ng"] = self.threshold_best_ng
        return d

    def threshold_stale_for(self, grid) -> bool:
        """True if this threshold was measured on a different patch grid.

        Unmeasured thresholds are not reported stale -- there is nothing to be
        stale against, and `threshold_measured` covers that case separately.
        """
        if self.threshold_grid is None:
            return False
        return tuple(int(v) for v in self.threshold_grid) != tuple(
            int(v) for v in grid)

    @property
    def threshold_measured(self) -> bool:
        """Was this threshold measured against known-bad parts, or guessed?"""
        return self.threshold_grid is not None

    @classmethod
    def from_dict(cls, d: dict) -> "SKUConfig":
        return cls(
            sku=d["sku"],
            canvas_size=tuple(d.get("canvas_size", (1024, 768))),
            ref_area_px=d.get("ref_area_px", 0.0),
            rois=[ROI.from_dict(r) for r in d.get("rois", [])],
            threshold=d.get("threshold", 6.0),
            input_size=d.get("input_size"),
            threshold_grid=d.get("threshold_grid"),
            threshold_worst_ok=d.get("threshold_worst_ok"),
            threshold_best_ng=d.get("threshold_best_ng"),
        )

    def apply_to(self, cfg):
        """Return `cfg` with this SKU's overrides applied.

        Registration and inspection must both call this, or the model is
        fitted at one resolution and tested at another -- which silently
        changes the patch grid and invalidates the position statistics.
        """
        if self.input_size is None:
            return cfg
        import copy

        cfg = copy.deepcopy(cfg)
        cfg.features.input_size = int(self.input_size)
        return cfg

    def save(self, path: Path | None = None) -> Path:
        path = path or sku_config_path(self.sku)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False keeps the Thai from the inspection sheets readable
        # in the file; escaped to \uXXXX it is valid JSON but useless to the
        # operator who has to check the box against the paper sheet.
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, sku: str, path: Path | None = None) -> "SKUConfig":
        path = path or sku_config_path(sku)
        if not path.exists():
            raise FileNotFoundError(
                f"no ROI config for SKU {sku} at {path}; run "
                f"`python -m inspector.annotate --sku {sku}` to create one"
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def whole_part_roi(self) -> ROI:
        cw, ch = self.canvas_size
        return ROI(WHOLE_PART, "appearance", (0, 0, cw, ch))

    def effective_rois(self) -> list[ROI]:
        """Configured ROIs, or the whole part if none are defined yet."""
        return [r.clip(self.canvas_size) for r in self.rois] or \
            [self.whole_part_roi()]


def check_name(roi: ROI) -> str:
    """The check an ROI belongs to: its id, minus the geometry suffix."""
    if roi.type == "geometry" and roi.id.endswith(GEOM_SUFFIX):
        return roi.id[:-len(GEOM_SUFFIX)]
    return roi.id


def group_checks(rois) -> list[tuple[str, ROI | None, ROI | None]]:
    """`(name, appearance, geometry)` per check, in the order first seen.

    Either half may be missing: a surface-only check has no geometry ROI, and
    a hand-written sku.json can hold a geometry ROI on its own.
    """
    order, parts = [], {}
    for roi in rois:
        name = check_name(roi)
        if name not in parts:
            order.append(name)
            parts[name] = [None, None]
        parts[name][0 if roi.type == "appearance" else 1] = roi
    return [(n, *parts[n]) for n in order]


def check_box(app: ROI | None, geo: ROI | None):
    """The box a check is drawn and edited as: the appearance box if any."""
    return (app or geo).box


def sku_dir(sku: str) -> Path:
    return MODEL_DIR / sku


def sku_config_path(sku: str) -> Path:
    return sku_dir(sku) / "sku.json"


def model_path(sku: str) -> Path:
    return sku_dir(sku) / "model.npz"


def crop_to_roi(image: np.ndarray, box) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    return image[max(0, y):y + h, max(0, x):x + w]


def patch_grid_box(box, canvas_size, grid_h, grid_w):
    """Map an ROI box from canvas pixels onto the patch grid.

    Used when a single whole-canvas feature map is scored and then sliced per
    ROI -- far cheaper than re-running the backbone on every crop, and it
    keeps every ROI on one consistent feature grid.
    """
    cw, ch = canvas_size
    x, y, w, h = box
    gx0 = int(np.floor(x / cw * grid_w))
    gy0 = int(np.floor(y / ch * grid_h))
    gx1 = int(np.ceil((x + w) / cw * grid_w))
    gy1 = int(np.ceil((y + h) / ch * grid_h))
    gx0, gy0 = max(0, gx0), max(0, gy0)
    gx1, gy1 = min(grid_w, max(gx1, gx0 + 1)), min(grid_h, max(gy1, gy0 + 1))
    return gy0, gy1, gx0, gx1
