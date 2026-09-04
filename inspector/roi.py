"""SKU configuration: ROI definitions and on-disk model layout."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import MODEL_DIR

WHOLE_PART = "whole_part"


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

    def to_dict(self) -> dict:
        return {
            "sku": self.sku,
            "canvas_size": list(self.canvas_size),
            "ref_area_px": self.ref_area_px,
            "threshold": self.threshold,
            "rois": [r.to_dict() for r in self.rois],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SKUConfig":
        return cls(
            sku=d["sku"],
            canvas_size=tuple(d.get("canvas_size", (1024, 768))),
            ref_area_px=d.get("ref_area_px", 0.0),
            rois=[ROI.from_dict(r) for r in d.get("rois", [])],
            threshold=d.get("threshold", 6.0),
        )

    def save(self, path: Path | None = None) -> Path:
        path = path or sku_config_path(self.sku)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
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
