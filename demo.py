"""Experimentation UI for the training-free defect inspector.

    python demo.py

Deliberately minimal -- this is a tool for running experiments and showing the
approach to a client, not the production operator interface. Every control
that changes a result is exposed, and every intermediate stage is displayed,
because the point is to see *where* it goes wrong rather than only what the
final answer was.
"""
from __future__ import annotations

import logging
import re
import time
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from inspector import autosize
from inspector import metrics as met
from inspector import metricsviz as mviz
from inspector import subspace as sub_mod
from inspector import tuning
from inspector import viz
from inspector.config import (DATA_DIR, GOLDEN_DIR, MODEL_DIR, NG_DIR,
                              TEST_DIR,
                              PipelineConfig, fit_signature)
from inspector.inspect import inspect_image
from inspector.pipeline import load_image
from inspector.register import (load_registration, register_sku,
                                save_registration)
from inspector.roi import ROI, SKUConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
# httpx logs every request at INFO and buries Gradio's "Running on"
# banner, which is the one line the user actually needs.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("demo")

# Gradio 6 takes `css` on launch(), not the Blocks constructor -- passing it
# to Blocks is dropped silently. A Row centres its children and Markdown has
# no fixed height, so without this the refresh button stretches and floats.
UI_CSS = """
#sku_row { align-items: flex-end !important; gap: 8px !important; }
#sku_refresh { min-width: 40px !important; width: 40px !important;
               height: 40px !important; flex: 0 0 40px !important;
               padding: 0 !important; font-size: 18px !important; }
/* Inspection images: show the whole frame, capped so it stays on screen.
   A fixed height alone makes Gradio fill the box and crop the edges off --
   which on a heatmap hides exactly the defect the operator is looking for.
   `contain` fits the full image inside instead, letterboxing rather than
   cutting, and the vh cap keeps the verdict above it visible without
   scrolling. */
#heat_img img, #stages_img img {
    object-fit: contain !important;
    width: 100% !important;
    height: auto !important;
}
#heat_img img { max-height: 55vh !important; }
/* The stages strip is a wide row of small panels, so it needs less height
   than the single heatmap to read at the same scale. */
#stages_img img { max-height: 30vh !important; }
#heat_img, #stages_img { overflow: hidden !important; }
/* Test-image thumbnails: whole part visible, and the box no taller than the
   thumbnails inside it. Capping the cells while letting the grid keep its own
   height left one image floating in a mostly empty panel, so the grid is
   sized by its contents and only starts scrolling once there are enough rows
   to need it. */
#test_gallery .thumbnail-item img,
#test_gallery .grid-wrap img {
    object-fit: contain !important;
    max-height: 26vh !important;
}
#test_gallery .grid-wrap {
    max-height: none !important;
    height: auto !important;
    overflow-y: auto !important;
}
#test_gallery .grid-container {
    align-content: start !important;
    gap: 8px !important;
}
/* A single saved image should not be stretched across the full row width --
   it just becomes a wide, short cell. Capping the cell keeps its proportions. */
#test_gallery .thumbnail-item {
    max-width: 320px !important;
    aspect-ratio: auto !important;
}
"""

WHOLE_PART_MODE = "whole part"
ROI_MODE = "ROI boxes"


def _to_bgr(img) -> np.ndarray:
    """Gradio hands us RGB; the pipeline works in BGR throughout."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_rgb(img) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_config(backbone, input_size, tau, aggregation, position_norm,
                 ink_on, align_on, segmenter, coverage, threshold) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.features.backbone = backbone
    cfg.features.input_size = int(input_size)
    cfg.features.patch_coverage = float(coverage)
    cfg.subspace.tau = float(tau)
    cfg.subspace.aggregation = {"max": "max", "top-3 mean": "top3mean",
                                "top-1% TVaR": "tvar1pct"}[aggregation]
    cfg.subspace.position_normalize = bool(position_norm)
    cfg.ink.enabled = bool(ink_on)
    cfg.align.enabled = bool(align_on)
    cfg.segment.method = segmenter
    cfg.threshold = float(threshold)
    return cfg


def parse_rois(text: str | None, canvas) -> list[ROI]:
    """ROI table: one per line, `id, type, x, y, w, h`.

    An untouched Gradio Textbox hands back None rather than "", and every
    caller here can be reached with the box never having been typed in, so
    the empty case is normalised once rather than at four call sites.
    """
    rois = []
    for i, line in enumerate((text or "").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            raise ValueError(f"line {i}: expected `id, type, x, y, w, h`")
        rid, kind = parts[0], parts[1].lower()
        if kind not in ("appearance", "geometry"):
            raise ValueError(f"line {i}: type must be appearance or geometry")
        try:
            box = tuple(int(round(float(v))) for v in parts[2:6])
        except ValueError:
            raise ValueError(f"line {i}: x, y, w, h must be numbers")
        tol = {"area_pct": 10.0, "fill_ratio_pct": 12.0,
               "corner_angle_deg": 15.0} if kind == "geometry" else {}
        rois.append(ROI(rid, kind, box, tolerance=tol).clip(canvas))
    return rois


# ---------------------------------------------------------------- panel 1 + 2

def do_fit(files, backbone, input_size, tau, aggregation, position_norm,
           ink_on, align_on, segmenter, coverage, threshold, roi_mode,
           roi_text, state, sku=None, progress=gr.Progress()):
    """Fit the subspace on the uploaded reference images, and save it.

    The fit used to live only in session state under the name "demo", so it
    vanished on restart and nothing on disk ever changed -- which made the
    UI's Fit button quietly weaker than the CLI's.
    """
    if not files:
        return (state, "Upload at least one reference image first.", None,
                None, "", gr.update(), [])

    paths = [f if isinstance(f, str) else f.name for f in files]
    cfg = build_config(backbone, input_size, tau, aggregation, position_norm,
                       ink_on, align_on, segmenter, coverage, threshold)

    sig = fit_signature(cfg, paths)
    # Caching is the whole point of splitting fit from inspect: changing the
    # test image must never trigger a refit, because refitting is the slow
    # part and its cost is not representative of per-part production time.
    if state.get("signature") == sig and state.get("reg") is not None:
        reg = state["reg"]
        return (state,
                f"Reusing the cached fit ({len(reg['sources'])} images, "
                f"r={reg['model']['r']}). Nothing that affects the subspace "
                f"changed.",
                _roi_canvas(reg, state.get("draft_rois", [])),
                state.get("stage_sheet"),
                state.get("stats", ""), gr.update(),
                _roi_rows(state.get("draft_rois", [])))

    msgs = []

    def say(m):
        msgs.append(m)
        progress(0.5, desc=m)

    # Prefer the boxes drawn on the canvas; the text box is the advanced
    # fallback. Reading only the text box silently dropped clicked ROIs, and
    # a geometry ROI that never reaches register_sku gets no nominal measured
    # -- which then makes every geometry check auto-pass at inspection.
    if roi_mode == ROI_MODE:
        fit_rois = state.get("draft_rois") or parse_rois(roi_text,
                                                         (10_000, 10_000))
    else:
        fit_rois = []

    try:
        images = [load_image(p) for p in paths]
        t0 = time.time()
        reg = register_sku(sku or "demo", images, cfg,
                           [Path(p).name for p in paths], fit_rois,
                           progress=say)
        fit_s = time.time() - t0
        saved_to = None
        if sku:
            # save_registration keeps the ROIs, threshold and input_size that
            # are already on disk; only canvas geometry comes from the fit.
            saved_to = save_registration(reg)
    except Exception as exc:  # noqa: BLE001 - surface the real cause in the UI
        log.exception("fit failed")
        return (state, f"Fit failed: {exc}\n\n{traceback.format_exc()[-800:]}",
                None, None, "", gr.update(), [])

    canvas = reg["canvas_size"]
    sheet = viz.grid(
        [(r.aligned_image, Path(r.source).stem) for r in reg["results"]],
        cols=min(4, len(reg["results"]))
    )

    warn = []
    if reg["near_duplicates"]:
        pairs = ", ".join(f"{Path(reg['sources'][i]).stem}~"
                          f"{Path(reg['sources'][j]).stem}"
                          for i, j, _ in reg["near_duplicates"][:5])
        warn.append(
            f"WARNING: {len(reg['near_duplicates'])} near-duplicate pair(s) "
            f"({pairs}). These add fit time without widening the subspace -- "
            f"coverage of lighting and highlight positions is what helps, not "
            f"image count."
        )
    if reg["loo_screen"].get("rejected"):
        warn.append(
            f"WARNING: leave-one-out screen rejected image(s) "
            f"{reg['loo_screen']['rejected']} as outliers. A defective golden "
            f"part would put its defect permanently inside the normal subspace."
        )
    if len(reg["results"]) < 8:
        warn.append(
            f"Only {len(reg['results'])} reference image(s). The expected "
            f"useful range is 8-16 spanning different lighting and highlight "
            f"positions; below that the subspace is too narrow and normal "
            f"lighting variation reads as anomalous."
        )

    gs = reg["golden_scores"].get("whole_part", [])
    stats = "\n".join([
        f"Fit time            {fit_s:.1f} s",
        f"Reference images    {len(reg['results'])}",
        f"Canonical canvas    {canvas[0]} x {canvas[1]}",
        f"Patch grid          {reg['grid'][0]} x {reg['grid'][1]}",
        f"Valid patches (fit) {reg['n_valid_patches']}",
        f"Per image           min {min(reg['per_image_patches'])}  "
        f"max {max(reg['per_image_patches'])}",
        f"Retained components r = {reg['model']['r']}",
        f"Golden whole-part   min {min(gs):.2f}  max {max(gs):.2f}"
        if gs else "",
        "",
        "A threshold has to sit above the golden max, or good parts fail.",
    ])

    state = dict(state)
    state.update({"reg": reg, "cfg": cfg, "signature": sig,
                  "canvas": canvas, "stats": stats,
                  "stage_sheet": _to_rgb(sheet)})
    # Keep whatever the ROI editor already had; a refit should not silently
    # discard boxes the operator has drawn.
    state.setdefault("draft_rois", [])
    state["roi_pending"] = None

    # Warnings emitted during the fit (e.g. a geometry ROI on flat interior)
    # arrive early and would otherwise be cut by the msgs[-6:] tail.
    if saved_to:
        msgs.append(f"saved model to {saved_to}")
    warn = [m for m in msgs if m.startswith("WARNING")] + warn
    status = "\n".join(msgs[-6:] + [""] + warn) if warn else "\n".join(msgs[-6:])
    default_roi = roi_text or ""
    if roi_mode == ROI_MODE and not default_roi.strip():
        cw, ch = canvas
        default_roi = (f"# id, type, x, y, w, h   (canvas {cw} x {ch})\n"
                       f"example, appearance, {cw // 4}, {ch // 4}, "
                       f"{cw // 4}, {ch // 4}\n")

    draft = state.get("draft_rois", [])
    return (state, status, _roi_canvas(reg, draft), _to_rgb(sheet), stats,
            gr.update(value=default_roi), _roi_rows(draft))


def scope_note(roi_mode) -> str:
    """Say what the selected scope means, and flag the baseline as a baseline."""
    if roi_mode == WHOLE_PART_MODE:
        return (
            "**Whole part** — the entire in-mask area is scored as one "
            "region and the boxes below are ignored.\n\n"
            "This is a comparison baseline, not how the station runs. It "
            "flags anything anywhere on the part, including defects no "
            "inspection-sheet check covers — which a separate station "
            "upstream already handles. A good part with an off-sheet defect "
            "will fail here."
        )
    return (
        "**ROI boxes** — only the checks below are scored, each against its "
        "own limit. This is what production does: the inspection sheet is "
        "the only thing this station may reject on."
    )


def do_preview_rois(roi_mode, roi_text, state):
    """Draw the ROI boxes on the aligned reference so they can be checked."""
    reg = state.get("reg")
    if reg is None:
        return None, "Fit the subspace first."
    if roi_mode == WHOLE_PART_MODE:
        return (_to_rgb(reg["ref_image"]),
                "Whole-part mode: the entire in-mask area is scored as one "
                "region. Useful as the comparison baseline against ROI "
                "scoping.")
    try:
        rois = parse_rois(roi_text, reg["canvas_size"])
    except ValueError as exc:
        return None, f"Could not parse the ROI table: {exc}"
    if not rois:
        return _to_rgb(reg["ref_image"]), "No ROIs defined yet."
    img = viz.draw_rois(reg["ref_image"], [r.to_dict() for r in rois])
    return _to_rgb(img), f"{len(rois)} ROI(s) drawn on the aligned reference."


def saved_reference(sku: str):
    """The aligned reference a previous registration left on disk.

    Registration writes `reference_aligned.png` next to the model, so a part
    that has been fitted before does not need re-fitting just to draw ROIs on
    it -- which otherwise costs minutes every time the server restarts.
    """
    if not sku:
        return None
    path = MODEL_DIR / sku / "reference_aligned.png"
    if not path.exists():
        return None
    try:
        return load_image(path)
    except ValueError:
        return None


def box_fill(box, reg=None, sku=None) -> float:
    """Fraction of an ROI box that is actually part, on the reference.

    A box mostly covering background dilutes every measure taken inside it.
    The tab check on SKU 8983864641 ran 167 px past the part's edge -- 40%
    fill -- and geometry could not separate a missing tab from a good one
    (0.82x). Tightened to 56% fill it separated cleanly (1.16x), and the
    appearance margin improved from 1.15x to 1.51x at the same time.
    """
    mask = reg["ref_mask"] if reg else None
    if mask is None and sku:
        path = MODEL_DIR / sku / "reference_mask.png"
        if path.exists():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return float("nan")
    x, y, w, h = (int(v) for v in box)
    crop = mask[max(0, y):y + h, max(0, x):x + w]
    return float((crop > 0).mean()) if crop.size else float("nan")


def geometry_useless_here(box, reg=None, sku=None) -> bool:
    """Does the silhouette have an edge inside this box?

    A geometry check measures the part's outline. On solid interior the box is
    simply full on every good part, so the check can never catch a defect --
    an embossed mark or a scratch does not move the outline -- while still
    firing whenever segmentation wobbles. Threshold from the sample SKUs: a box
    straddling a real outline measures 0.40 (tab) or 0.62 (chamfered corner);
    one on flat interior measures 0.97 (the embossed RH mark).

    Answered here from the reference mask, so it can be said while the box is
    being drawn rather than only after a fit.
    """
    mask = reg["ref_mask"] if reg else None
    if mask is None and sku:
        path = MODEL_DIR / sku / "reference_mask.png"
        if path.exists():
            try:
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            except Exception:  # noqa: BLE001 - advisory only
                mask = None
    if mask is None:
        return False
    from inspector import geometry as geom_mod

    frac = geom_mod.measure_roi_geometry(mask, box).get("area_frac", float("nan"))
    return bool(np.isfinite(frac) and (frac > 0.95 or frac < 0.05))


def _roi_canvas(reg, rois, pending=None, sku=None):
    """Aligned reference with the saved boxes drawn, plus any pending corner.

    Falls back to the reference saved on disk when this session has not
    fitted yet, so a previously registered part is editable immediately.
    """
    base = reg["ref_image"] if reg else saved_reference(sku)
    if base is None:
        return None
    img = viz.draw_rois(base, [r.to_dict() for r in rois])
    if pending is not None:
        x, y = pending
        cv2.drawMarker(img, (int(x), int(y)), (255, 255, 255),
                       cv2.MARKER_CROSS, 34, 3)
        img = viz.label(img, "click the opposite corner",
                        (max(int(x) - 90, 4), max(int(y) - 16, 16)),
                        (255, 255, 255), 0.55)
    return _to_rgb(img)


def _roi_rows(rois):
    return [[r.id, r.type, *r.box, r.note] for r in rois]


def do_roi_click(state, evt: gr.SelectData):
    """Two clicks define a box: first corner, then the opposite one.

    Click-to-place rather than drag: `gr.Image` reports a single coordinate on
    select, with no press/release pair to bracket a drag, and its editor mode
    returns painted pixels rather than coordinates -- a drag would mean
    recovering a rectangle from a bitmap. Two clicks give exact numbers with
    nothing to infer, and the live preview after the first click makes the
    second one easy to place.
    """
    reg = state.get("reg")
    sku = state.get("sku")
    # A previously registered part has its reference on disk, so ROIs can be
    # drawn without re-fitting just because the server restarted.
    if reg is None and saved_reference(sku) is None:
        return (state, None, [],
                "Fit the subspace first (step 1) — the canvas ROI boxes are "
                "drawn on does not exist until then.")

    x, y = int(evt.index[0]), int(evt.index[1])
    state = dict(state)
    rois = list(state.get("draft_rois", []))
    first = state.get("roi_pending")

    if first is None:
        state["roi_pending"] = (x, y)
        return (state, _roi_canvas(reg, rois, (x, y), sku=state.get("sku")),
                _roi_rows(rois),
                f"**Corner 1 set at ({x}, {y})** — now click the opposite "
                f"corner to close the box. *Clicked in the wrong place? Press "
                f"**cancel corner** to start this box again.*")

    x0, y0 = first
    state["roi_pending"] = None
    bx, by = min(x0, x), min(y0, y)
    bw, bh = abs(x - x0), abs(y - y0)
    if bw < 8 or bh < 8:
        return (state, _roi_canvas(reg, rois, sku=state.get("sku")), _roi_rows(rois),
                "That box is too small — start again with two corners further "
                "apart.")

    # A placeholder name: the table is the editor, so drawing only places the
    # box. Both types are created because they catch opposite failures over the
    # same region -- appearance sees a damaged feature, geometry sees a missing
    # or bent one, and neither sees the other's case. Delete the geometry row
    # when the check is surface-only (an embossed mark does not move the
    # silhouette, so geometry there measures nothing).
    existing = {r.id for r in rois}
    n = len(rois) + 1
    while f"check_{n}" in existing or f"check_{n}_geom" in existing:
        n += 1
    name = f"check_{n}"

    rois.append(ROI(name, "appearance", (bx, by, bw, bh)))
    rois.append(ROI(f"{name}_geom", "geometry", (bx, by, bw, bh),
                    tolerance={"area_pct": 10.0, "fill_ratio_pct": 12.0,
                               "corner_angle_deg": 15.0}))
    state["draft_rois"] = rois
    msg = (f"Added **{name}** (appearance) and **{name}_geom** (geometry) at "
           f"({bx}, {by}, {bw}, {bh}) — they catch opposite failures over the "
           f"same region. Rename them in the table below.")
    if geometry_useless_here((bx, by, bw, bh), reg, state.get("sku")):
        msg += (f"\n\n⚠️ **The outline has no edge inside this box** — it sits "
                f"on flat part interior. A geometry check there can never "
                f"catch a defect, but it will still fire when segmentation "
                f"wobbles and fail a good part. **Delete `{name}_geom`** "
                f"unless you drew this box for a shape feature.")
    else:
        msg += (f"\n\nIf this check is a surface mark that does not change the "
                f"part's outline — an embossed stamp, a printed character — "
                f"delete the `_geom` row.")
    fill = box_fill((bx, by, bw, bh), reg, state.get("sku"))
    if np.isfinite(fill) and fill < 0.45:
        msg += (f"\n\n⚠️ **Only {fill * 100:.0f}% of this box is part** — the "
                f"rest is background, which dilutes every measure taken "
                f"inside it. Drag it tighter around the feature.")
    return (state, _roi_canvas(reg, rois, sku=state.get("sku")),
            _roi_rows(rois), msg)


def do_roi_table_edit(rows, state):
    """Rebuild the ROI list from the edited table.

    The table is the single editor: id, type, geometry and description are all
    changed in place, and the canvas redraws from the result so a box moves as
    soon as its numbers do. Bad rows are skipped rather than raising -- a
    half-typed coordinate must not wipe the row being typed into.
    """
    state = dict(state)
    reg = state.get("reg")
    sku = state.get("sku")
    canvas = (reg["canvas_size"] if reg
              else (SKUConfig.load(sku).canvas_size
                    if sku and sku_config_exists(sku) else (10_000, 10_000)))

    # Gradio hands back a DataFrame or a list of lists depending on version.
    try:
        records = rows.values.tolist()
    except AttributeError:
        records = list(rows or [])

    rois, problems, seen = [], [], set()
    for i, row in enumerate(records, 1):
        cells = list(row) + [""] * (7 - len(row))
        rid = str(cells[0] or "").strip()
        if not rid:
            continue                      # a blank row is just an empty line
        kind = str(cells[1] or "appearance").strip().lower()
        if kind.startswith("geom"):
            kind = "geometry"
        elif kind.startswith("app"):
            kind = "appearance"
        else:
            problems.append(f"row {i}: type must be appearance or geometry")
            continue
        try:
            box = tuple(int(round(float(v))) for v in cells[2:6])
        except (TypeError, ValueError):
            problems.append(f"row {i} ({rid}): x, y, w, h must be numbers")
            continue
        if box[2] < 1 or box[3] < 1:
            problems.append(f"row {i} ({rid}): width and height must be > 0")
            continue
        if rid in seen:
            problems.append(f"row {i}: '{rid}' is a duplicate name")
            continue
        seen.add(rid)
        tol = {"area_pct": 10.0, "fill_ratio_pct": 12.0,
               "corner_angle_deg": 15.0} if kind == "geometry" else {}
        rois.append(ROI(rid, kind, box, tolerance=tol,
                        note=str(cells[6] or "").strip()).clip(canvas))

    state["draft_rois"] = rois
    state["roi_pending"] = None
    msg = f"{len(rois)} check(s) defined."
    if problems:
        msg += " Skipped — " + "; ".join(problems[:4])
    # Same advisory as when a box is drawn: a geometry row on flat interior
    # measures nothing and can only misfire.
    flat = [r.id for r in rois if r.type == "geometry"
            and geometry_useless_here(r.box, reg, sku)]
    if flat:
        msg += (" ⚠️ " + ", ".join(f"`{i}`" for i in flat) +
                (" sits" if len(flat) == 1 else " sit") +
                " on flat part interior — the outline has no edge there, so "
                "the check can only misfire on segmentation noise. Delete "
                "unless drawn for a shape feature.")
    msg += " *Not saved yet — press **Save sku.json** to make it stick.*"
    return state, _roi_canvas(reg, rois, sku=sku), msg


def sku_config_exists(sku: str) -> bool:
    from inspector.roi import sku_config_path
    return sku_config_path(sku).exists()


def do_roi_cancel(state):
    """Drop a half-placed box without touching the ones already defined."""
    state = dict(state)
    reg = state.get("reg")
    rois = list(state.get("draft_rois", []))
    if state.get("roi_pending") is None:
        return (state, _roi_canvas(reg, rois, sku=state.get("sku")),
                _roi_rows(rois), "No half-placed corner to cancel.")
    state["roi_pending"] = None
    return (state, _roi_canvas(reg, rois, sku=state.get("sku")),
            _roi_rows(rois),
            "Cancelled — click a fresh first corner when you are ready.")


def do_roi_undo(state):
    """Drop the most recently added box."""
    state = dict(state)
    reg = state.get("reg")
    sku = state.get("sku")
    rois = list(state.get("draft_rois", []))
    state["roi_pending"] = None
    if not rois:
        return (state, _roi_canvas(reg, rois, sku=sku), [], "Nothing to undo.")
    gone = rois.pop()
    state["draft_rois"] = rois
    return (state, _roi_canvas(reg, rois, sku=sku), _roi_rows(rois),
            f"Removed **{gone.id}**. {len(rois)} left. *Not saved yet — press "
            f"**Save sku.json** to make it stick.*")


def do_roi_delete(name, state):
    """Remove one named check, rather than only the last one added.

    Undo is no help when the wrong box is the second of four: without this
    the only route is `clear all` and redrawing everything.
    """
    state = dict(state)
    reg = state.get("reg")
    sku = state.get("sku")
    rois = list(state.get("draft_rois", []))
    want = (name or "").strip()
    if not want:
        return (state, _roi_canvas(reg, rois, sku=sku), _roi_rows(rois),
                "Type the name of the check to remove, exactly as it appears "
                "in the table.")
    keep = [r for r in rois if r.id != want]
    if len(keep) == len(rois):
        have = ", ".join(f"`{r.id}`" for r in rois) or "none defined"
        return (state, _roi_canvas(reg, rois, sku=sku), _roi_rows(rois),
                f"No check called **{want}**. Defined: {have}.")
    state["draft_rois"] = keep
    state["roi_pending"] = None
    return (state, _roi_canvas(reg, keep, sku=sku), _roi_rows(keep),
            f"Removed **{want}**. {len(keep)} left. *Not saved yet — press "
            f"**Save sku.json** to make it stick.*")


def do_roi_clear(state):
    reg = state.get("reg")
    state = dict(state)
    state["draft_rois"] = []
    state["roi_pending"] = None
    canvas = _roi_canvas(reg, [], sku=state.get("sku"))
    return (state, canvas, [],
            "Cleared every check. *Not saved yet — press **Save sku.json** to "
            "make it stick, or **reload saved checks** to get them back.*")


def do_roi_load(sku, state):
    """Load an existing sku.json so it can be edited rather than retyped.

    Also records the selected part in state, which is what lets the canvas
    fall back to the reference image on disk.
    """
    reg = state.get("reg")
    state = dict(state)
    state["sku"] = sku
    has_saved = saved_reference(sku) is not None
    hint = ("" if (reg or has_saved) else
            " The canvas is blank until this part has been fitted at least "
            "once (step 1).")
    try:
        cfg = SKUConfig.load(sku)
    except FileNotFoundError:
        state["draft_rois"] = []
        state["roi_pending"] = None
        return (state, _roi_canvas(reg, [], sku=sku), [],
                f"No saved checks for {sku} yet — draw boxes and save.{hint}")
    state["draft_rois"] = list(cfg.rois)
    state["roi_pending"] = None
    source = "this session's fit" if reg else "the saved reference"
    return (state, _roi_canvas(reg, cfg.rois, sku=sku),
            _roi_rows(cfg.rois),
            f"Loaded {len(cfg.rois)} ROI(s) from {sku}, threshold "
            f"{cfg.threshold}. Drawn on {source}. Edit and save to "
            f"overwrite.{hint}")


def _labelled_list(root, sku):
    """Photos already saved for this part under a labelled folder."""
    folder = root / sku if sku else None
    if not folder or not folder.exists():
        return []
    return [str(x) for x in sorted(folder.iterdir())
            if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]


def _ng_list(sku):
    """Known-bad photos already saved for this part."""
    from inspector.config import NG_DIR
    return _labelled_list(NG_DIR, sku)


def _ok_list(sku):
    """Known-good photos already saved for this part."""
    from inspector.config import OK_DIR
    return _labelled_list(OK_DIR, sku)


def _labelled_label(root, sku, kind):
    """Gallery title that says what is on disk, so saved photos read as ready.

    An empty-looking gallery and a full one titled the same way are
    indistinguishable at a glance, and re-uploading photos that are already
    saved is the obvious thing to do about it.
    """
    n = len(_labelled_list(root, sku))
    if not n:
        return gr.update(label=f"{kind} set for this part — none saved yet")
    return gr.update(label=f"{kind} set for this part — {n} saved, ready to use")


def _ng_gallery_label(sku):
    from inspector.config import NG_DIR
    return _labelled_label(NG_DIR, sku, "NG")


def _ok_gallery_label(sku):
    from inspector.config import OK_DIR
    return _labelled_label(OK_DIR, sku, "OK")


# What each set is for, and how many photos it takes to do that job. The
# golden numbers are the binding ones: the subspace is fitted on them, so too
# few models almost no legitimate variation and ordinary good parts start
# scoring as anomalies. OK and NG are measured against that model, so they are
# about sampling the range rather than about the fit.
GOLDEN_MIN, GOLDEN_GOOD = 4, 8
OK_MIN, OK_GOOD = 2, 5
NG_GOOD = 3


def _golden_count(sku):
    """Golden photos, excluding the inspection sheet."""
    from inspector.config import GOLDEN_DIR
    return len([p for p in _labelled_list(GOLDEN_DIR, sku)
                if not Path(p).name.startswith("IS_")])


def _tune_ready(sku):
    """Say whether Measure can run now, and what more photos would buy.

    Each set limits the result in its own way, so a single "add more images"
    would be useless: golden constrains the fit, OK sets the floor the
    threshold sits above, NG sets the ceiling it sits below.
    """
    if not sku:
        return ""
    from inspector.config import NG_DIR, OK_DIR
    n_ng = len(_labelled_list(NG_DIR, sku))
    n_ok = len(_labelled_list(OK_DIR, sku))
    n_gold = _golden_count(sku)

    tips = []
    if n_gold < GOLDEN_MIN:
        tips.append(
            f"**Golden — {n_gold}, want at least {GOLDEN_MIN}.** The subspace "
            f"is fitted on these, and {n_gold} models almost none of the "
            f"lighting and handling variation a real part shows, so ordinary "
            f"good parts score as anomalies. This one limits everything "
            f"downstream — add here first.")
    elif n_gold < GOLDEN_GOOD:
        tips.append(
            f"**Golden — {n_gold}, {GOLDEN_GOOD} is comfortable.** More golden "
            f"parts spanning real lighting and placement make every score "
            f"below steadier.")
    if not n_ok:
        tips.append(
            f"**Known-good — none.** Without them the threshold is an upper "
            f"bound, not a measured window: the golden set is *fitted on*, so "
            f"it scores near zero and cannot say where everyday good parts "
            f"land. {OK_MIN} is the minimum, {OK_GOOD} is comfortable.")
    elif n_ok < OK_MIN:
        tips.append(
            f"**Known-good — {n_ok}.** The floor rests on a single part, so "
            f"one unusual photo moves the threshold. {OK_GOOD} is comfortable.")
    elif n_ok < OK_GOOD:
        tips.append(
            f"**Known-good — {n_ok}, {OK_GOOD} is comfortable.** These set the "
            f"floor the threshold must clear, so they should cover the full "
            f"range of parts you would still ship.")
    if not n_ng:
        tips.append(
            f"**Known-bad — none.** Nothing can be measured without them: they "
            f"are what shows which resolution separates good from bad, and "
            f"where the failing side starts. At least one per check on the "
            f"inspection sheet.")
    elif n_ng < NG_GOOD:
        tips.append(
            f"**Known-bad — {n_ng}.** Aim for at least one per check on the "
            f"inspection sheet, so every defect the sheet names is represented "
            f"by a part that actually shows it.")

    if not n_ng:
        head = ("**Cannot measure yet — no known-bad photos for this part.**")
    else:
        have = [f"**{n_ng}** known-bad"]
        if n_ok:
            have.append(f"**{n_ok}** known-good")
        have.append(f"**{n_gold}** golden")
        head = (f"Ready to measure on {', '.join(have)} photo(s) already saved "
                f"for this part — just press the button, no upload needed.")
    if tips:
        head += "\n\nTo make the result stronger:\n\n" + "\n".join(
            f"- {t}" for t in tips)
    return head


def _import_labelled(files, sku, root, folder_name, what):
    """Copy uploaded photos into <root>/<sku>/.

    Separate from test_images/ on purpose: these are tuning inputs with a
    known label, not parts being judged. Keeping them apart removes the
    question of which folder a photo belongs in.
    """
    if not sku:
        return "Select a part first."
    paths = [Path(f if isinstance(f, str) else f.name) for f in (files or [])]
    if not paths:
        return "No images to save."

    import shutil

    dest = root / sku
    dest.mkdir(parents=True, exist_ok=True)
    added, same = [], []
    for src in paths:
        if not src.exists():
            continue
        target = dest / src.name
        if target.exists() and target.stat().st_size == src.stat().st_size \
                and target.read_bytes() == src.read_bytes():
            same.append(src.name)
            continue
        shutil.copy2(src, target)
        added.append(src.name)
    if added:
        return (f"Saved **{len(added)}** {what} photo(s) to "
                f"`{folder_name}/{sku}/`" +
                (f"; {len(same)} already there." if same else "."))
    return f"All {len(same)} photo(s) are already in `{folder_name}/{sku}/`."


def do_ng_import(files, sku):
    """Copy uploaded bad-part photos into ng_images/<sku>/."""
    from inspector.config import NG_DIR
    return _import_labelled(files, sku, NG_DIR, "ng_images", "bad-part")


def do_ok_import(files, sku):
    """Copy uploaded good-part photos into ok_images/<sku>/.

    Not the same as the golden set in step 1: those are what the subspace is
    fitted on, so they score near zero and cannot say where everyday good parts
    land. These are ordinary passing parts, scored like any test part, and they
    are what gives the threshold a measured floor.
    """
    from inspector.config import OK_DIR
    return _import_labelled(files, sku, OK_DIR, "ok_images", "good-part")


def _fmt_duration(seconds: float) -> str:
    """Elapsed time in the units a person would use for it.

    Measuring takes minutes, so raw seconds past ~90 read as noise: "4 min 12 s"
    lands where "252.4 s" has to be divided first.
    """
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs:02d} s"


def do_tune(files, sku, state, progress=gr.Progress()):
    """Measure input_size and the threshold from the NG set.

    Both need known-bad parts: good parts score almost identically at every
    resolution, and the golden set cannot show where good and bad separate.
    Nothing is written -- the values land in their controls for you to apply.
    """
    rois = state.get("draft_rois", [])
    if not rois and sku:
        try:
            rois = SKUConfig.load(sku).rois
        except FileNotFoundError:
            rois = []
    if not rois:
        return gr.update(), gr.update(), ("**Do step 2 first** — define at "
                                          "least one check to measure inside.")

    paths = [Path(f if isinstance(f, str) else f.name) for f in (files or [])]
    if not paths and sku:
        paths = [p for p in sorted((GOLDEN_DIR / sku).glob("*"))
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                 and not p.name.startswith("IS_")]
    if len(paths) < 2:
        return gr.update(), gr.update(), ("**Do step 1 first** — at least 2 "
                                          "golden images are needed to fit.")

    # Measuring refits at every candidate resolution, so it runs for minutes.
    # Reporting how long it took -- on the failure path too -- is the
    # difference between "that was slow" and "that hung".
    t0 = time.time()
    try:
        res = tuning.tune(sku, paths, rois,
                          say=lambda m: progress(0.5, desc=m))
    except Exception as exc:  # noqa: BLE001 - surface the real cause
        log.exception("tuning failed")
        return (gr.update(), gr.update(),
                f"Measuring failed after {_fmt_duration(time.time() - t0)}: "
                f"{exc}\n\n```\n{traceback.format_exc()[-600:]}\n```")
    elapsed = time.time() - t0

    if not res.ok:
        return (gr.update(), gr.update(),
                f"{res.reason}\n\n*Measured in {_fmt_duration(elapsed)}.*")

    # Remember what was measured so saving can record the window alongside the
    # number. Only a run with real OK parts measured a floor worth recording.
    if res.n_ok > 0 and res.threshold is not None:
        state["tuned"] = {"threshold": res.threshold,
                          "input_size": res.input_size,
                          "worst_ok": res.worst_ok,
                          "best_ng": res.best_ng}
    else:
        state.pop("tuned", None)

    md = (f"### input resolution **{res.input_size}** · score threshold "
          f"**{res.threshold:g}**\n\n"
          f"*Measured in {_fmt_duration(elapsed)} across "
          f"{len(res.sweep)} resolution(s).*\n\n"
          f"{tuning.table(res)}\n\n{res.reason}")
    thr = gr.update() if res.threshold is None else res.threshold
    return res.input_size, thr, md

def recommend_threshold(state, sku=None):
    """Suggest a verdict threshold, and say what the suggestion rests on.

    Two tiers, because they are not equally trustworthy:

      * golden only -- gives a *floor*. Good parts scored this high, so the
        threshold must clear it. It cannot tell you the ceiling, and the golden
        set is the cleanest parts you own: real OK parts score higher. On SKU
        8983980631 the golden max was 1.50 at every resolution while the worst
        genuinely-good part reached 17.59, so a golden-only number can sit far
        under where good parts actually land.
      * golden + labelled NG -- gives the real window. The geometric mean of
        the worst passing part and the best failing one sits inside the gap and
        is not dragged by either side.

    Returns (value_or_None, markdown).
    """
    reg = state.get("reg")
    sku = sku or state.get("sku")
    gs = reg.get("golden_scores") if reg else None
    if not gs and sku:
        try:
            gs = load_registration(sku).get("golden_scores")
        except (FileNotFoundError, KeyError):
            gs = None
    if not gs:
        return None, ("*No fitted model yet.* Fit in step 1 — the "
                      "recommendation comes from what good parts scored.")

    per_roi = {k: v for k, v in gs.items()
               if v and k not in ("whole_part", "outside_rois")}
    pool = per_roi or {k: v for k, v in gs.items() if v}
    golden_max = max(max(v) for v in pool.values())

    labels = autosize.read_labels(sku) if sku else {}
    ng_names = [n for n, y in labels.items() if y == 1]

    if not ng_names:
        # Floor only. 3x golden max leaves room for parts dirtier than the
        # golden set without being so high it stops catching anything.
        value = round(max(golden_max * 3.0, golden_max + 2.0), 1)
        rows = "\n".join(f"| `{k}` | {max(v):.2f} |" for k, v in pool.items())
        return value, (
            f"### Suggested threshold: **{value}**\n\n"
            f"**Floor only — not validated.** Based on what your good parts "
            f"scored:\n\n| check | worst golden score |\n|---|---|\n{rows}\n\n"
            f"The threshold has to clear **{golden_max:.2f}** or good parts "
            f"fail. Above that, this is a guess: the golden set is your "
            f"cleanest parts, and real OK parts score higher. Add a few NG "
            f"photos and re-run **Accuracy** to get the real window."
        )

    try:
        saved = SKUConfig.load(sku).threshold if sku else None
    except FileNotFoundError:
        saved = None
    saved_row = (f"| **in use for this part** | **{saved:g}** | what "
                 f"`sku.json` saves and production uses |\n"
                 if saved is not None else "")
    return saved, (
        f"### This part is set to **{saved:g}**\n\n" if saved is not None
        else "### Measure it in the Accuracy tab\n\n"
    ) + (
        f"| number | value | what it means |\n|---|---|---|\n"
        f"| golden floor | {golden_max:.2f} | the worst score a *golden* part "
        f"got — anything at or below this fails good parts |\n"
        f"{saved_row}\n"
        f"**These are not competing numbers.** The floor is a hard minimum, "
        f"not a suggestion: golden photos are the cleanest parts you have, "
        f"and everyday good parts score well above them — often several "
        f"times higher.\n\n"
        f"You have **{len(ng_names)}** labelled NG part(s), so the working "
        f"value can be measured rather than guessed: run **Accuracy → "
        f"Evaluate** and the threshold slider will show exactly where good "
        f"and bad separate."
    )


def fitted_settings_note(state, sku=None) -> str:
    """What the loaded model was fitted at -- read-only, by design.

    Inspection has no free choice here. The subspace and the position
    statistics are tied to the patch grid they were fitted on, so scoring at a
    different resolution would silently invalidate them; `inspect_image` even
    warns and drops position normalisation when the grids disagree. Showing
    these as editable controls would imply a choice that does not exist.
    """
    reg = state.get("reg")
    cfg = state.get("cfg")
    source = "this session's fit"
    # A model saved by an earlier session (or by the CLI) is just as usable.
    # Reporting "no model fitted yet" while model.npz sits on disk sent people
    # back to re-fit something they already had.
    if reg is None or cfg is None:
        sku = sku or state.get("sku")
        try:
            reg = load_registration(sku) if sku else None
            cfg = reg["config"] if reg else None
            source = f"the saved model for **{sku}**"
        except (FileNotFoundError, KeyError, TypeError):
            reg = cfg = None
    if reg is None or cfg is None:
        return ("*No model for this part yet.* Run **Registration → Fit "
                "subspace**; these settings then come from that fit.")
    gh, gw = reg["grid"]
    cw, ch = reg["canvas_size"]
    return (
        f"**Inherited from {source} — not editable here.**\n\n"
        f"| | |\n|---|---|\n"
        f"| input resolution | `{cfg.features.input_size}` |\n"
        f"| patch grid | `{gh} x {gw}` |\n"
        f"| canonical canvas | `{cw} x {ch}` |\n"
        f"| backbone | `{cfg.features.backbone}` |\n"
        f"| aggregation | `{cfg.subspace.aggregation}` |\n\n"
        f"The model was fitted at this resolution, so it has to be scored at "
        f"the same one — a different patch grid would invalidate the "
        f"per-position statistics. Change it in Registration and re-fit."
    )


def do_save_tuned(sku, input_size, threshold, state):
    """Write just the two measured settings into sku.json.

    Separate from the ROI save because it needs neither ROIs nor a session
    fit: measuring produces these two numbers and nothing else, and leaving
    them only in the controls meant a measured value silently never reached
    the file the pipeline reads.
    """
    if not sku:
        return "Select a part first."
    try:
        cfg = SKUConfig.load(sku)
    except FileNotFoundError:
        return (f"No `sku.json` for **{sku}** yet - define its checks in "
                f"step 2 and save those first.")

    old_size, old_thr = cfg.input_size, cfg.threshold
    cfg.input_size = int(input_size)
    cfg.threshold = float(threshold)
    for r in cfg.rois:
        # Per-ROI overrides that mirrored the SKU default would otherwise
        # keep the old number and quietly win at inspection.
        if r.threshold is not None and abs(r.threshold - old_thr) < 1e-9:
            r.threshold = float(threshold)

    # The recorded window belongs to the value that was measured. A run with
    # known-good parts measured both sides, so record it against the number it
    # produced; anything else leaves the file vouching for a value nobody
    # measured, so clear it.
    tuned = (state or {}).get("tuned") or {}
    measured_here = (
        tuned and abs(tuned["threshold"] - float(threshold)) < 1e-9
        and int(tuned["input_size"]) == int(input_size))
    if measured_here:
        cfg.threshold_worst_ok = float(tuned["worst_ok"])
        cfg.threshold_best_ng = float(tuned["best_ng"])
    elif abs((old_thr or 0) - float(threshold)) > 1e-9:
        cfg.threshold_grid = None
        cfg.threshold_worst_ok = None
        cfg.threshold_best_ng = None

    path = cfg.save()
    changed = []
    if old_size != cfg.input_size:
        changed.append(f"input resolution {old_size} to **{cfg.input_size}**")
    if abs((old_thr or 0) - cfg.threshold) > 1e-9:
        changed.append(f"threshold {old_thr:g} to **{cfg.threshold:g}**")
    if not changed:
        return f"`{path.name}` already had these values - nothing changed."

    msg = f"Saved to `{path}`: " + ", ".join(changed) + "."
    if old_size != cfg.input_size:
        msg += (f"\n\n**Re-fit in step 1** - the saved model is still at "
                f"{old_size}, and a model scored at a different resolution "
                f"than it was fitted at loses its position statistics.")
    return msg


def do_roi_save(sku, threshold, state):
    """Write sku.json -- the file the pipeline actually reads."""
    reg = state.get("reg")
    if reg is None:
        return state, "Fit the subspace first — the canvas size comes from it."
    rois = state.get("draft_rois", [])
    if not rois:
        return state, "No ROIs to save."

    try:
        existing = SKUConfig.load(sku)
    except FileNotFoundError:
        existing = None

    cfg = SKUConfig(sku=sku, canvas_size=reg["canvas_size"],
                    ref_area_px=reg["ref_stats"].get("area", 0.0),
                    rois=rois, threshold=float(threshold),
                    input_size=existing.input_size if existing else None)
    # Carry the threshold's provenance forward, but only while the number is
    # unchanged: an edited threshold is no longer the one that was measured,
    # and claiming otherwise would hide it from the staleness check.
    if existing is not None and existing.threshold_measured \
            and abs(existing.threshold - float(threshold)) < 1e-9:
        cfg.threshold_grid = existing.threshold_grid
        cfg.threshold_worst_ok = existing.threshold_worst_ok
        cfg.threshold_best_ng = existing.threshold_best_ng
    path = cfg.save()
    return (state,
            f"Saved **{len(rois)} ROI(s)** to `{path}`.\n\n"
            f"Geometry ROIs need their nominals measured, so re-run "
            f"registration before inspecting:\n\n"
            f"```\npython -m inspector.register --sku {sku} "
            f"--segmenter birefnet\n```")


# -------------------------------------------------------------------- panel 3

def do_inspect(test_file, roi_mode, roi_text, threshold, state, sku=None):
    """Score one test image against the fitted subspace. Never refits.

    Falls back to the model saved on disk, so a part registered in an earlier
    session (or from the CLI) can be inspected without re-fitting it first.
    """
    reg = state.get("reg")
    cfg = state.get("cfg")
    if reg is None or cfg is None:
        sku = sku or state.get("sku")
        try:
            reg = load_registration(sku) if sku else None
            cfg = reg["config"] if reg else None
        except (FileNotFoundError, KeyError, TypeError):
            reg = cfg = None
    if reg is None or cfg is None:
        return (None, None,
                "No model for this part yet — run **Registration → Fit "
                "subspace** first.", "")
    if test_file is None:
        return (None, None,
                "Choose a part to inspect: upload one, or click an image in "
                "**saved test images**.", "")

    canvas = reg["canvas_size"]
    # Prefer the checks in the table; the text box is the advanced fallback.
    # Reading only the text box silently inspected with no ROIs at all.
    try:
        if roi_mode == WHOLE_PART_MODE:
            rois = []
        else:
            rois = state.get("draft_rois") or parse_rois(roi_text, canvas)
    except ValueError as exc:
        return (None, None, f"Could not parse the ROI table: {exc}", "")

    sku_cfg = SKUConfig(sku="demo", canvas_size=canvas, rois=rois,
                        threshold=float(threshold))

    path = test_file if isinstance(test_file, str) else test_file.name
    img = load_image(path)
    t0 = time.time()
    res = inspect_image(img, reg, sku_cfg, cfg, source=Path(path).name)
    infer_s = time.time() - t0

    if res.verdict == "RETAKE":
        stages = viz.grid([(res.pre.image, "raw"),
                           (viz.overlay_mask(res.pre.image, res.pre.part_mask)
                            if res.pre.part_mask is not None else res.pre.image,
                            "part mask")], cols=2)
        return (None, _to_rgb(stages),
                f"### RETAKE — ถ่ายภาพใหม่\n\n"
                f"## {res.reason_th}\n\n"
                f"*{res.reason}*\n\n"
                f"This is a photography problem, not a defective part -- "
                f"retake the picture rather than scrapping the part. "
                f"The part has NOT been inspected.",
                f"Inference time  {infer_s * 1000:.0f} ms")

    thr = float(threshold)
    masked = np.where(res.valid_grid, res.score_map, 0.0)
    heat = viz.heatmap_overlay(res.pre.aligned_image, masked,
                               vmin=0.0, vmax=max(thr * 2.0, 1e-6))
    heat = viz.draw_rois(heat, [r.to_dict() for r in sku_cfg.effective_rois()],
                         {r.id: r.score for r in res.rois}, thr)

    tiles = [(res.pre.image, "raw photo")]
    if res.pre.part_mask is not None:
        tiles.append((viz.overlay_mask(res.pre.image, res.pre.part_mask),
                      f"{cfg.segment.method} mask"))
    if res.pre.ink_mask is not None:
        tiles.append((viz.overlay_mask(res.pre.image, res.pre.ink_mask,
                                       (255, 0, 0)), "ink mask"))
    tiles.append((res.pre.aligned_image, "aligned canvas"))
    tiles.append((viz.overlay_mask(res.pre.aligned_image,
                                   res.pre.aligned_valid_mask), "valid mask"))
    for roi in sku_cfg.effective_rois()[:3]:
        x, y, w, h = roi.box
        crop = res.pre.aligned_image[y:y + h, x:x + w]
        if crop.size:
            tiles.append((crop, f"ROI {roi.id}"))
    stages = viz.grid(tiles, cols=min(5, len(tiles)))

    lines = [f"### {res.verdict}", ""]
    lines.append("| ROI | type | score | threshold | verdict | detail |")
    lines.append("|---|---|---|---|---|---|")
    for r in res.rois:
        mark = "PASS" if r.passed else "**FAIL**"
        score = "n/a" if not np.isfinite(r.score) else f"{r.score:.2f}"
        lines.append(f"| {r.id} | {r.type} | {score} | "
                     f"{r.threshold:.2f} | {mark} | {r.detail} |")

    gs = reg["golden_scores"].get("whole_part", [])
    if gs:
        lines += ["", f"Golden reference scores ranged {min(gs):.2f} - "
                      f"{max(gs):.2f}. A test score inside that band is "
                      f"indistinguishable from a normal part."]

    align = res.pre.info
    timing = "\n".join([
        f"Inference time   {infer_s * 1000:.0f} ms",
        f"Fit time         {state.get('fit_seconds', 0):.1f} s (cached, not rerun)",
        f"Alignment IoU    {align.get('iou', float('nan')):.3f}",
        f"Centroid drift   {align.get('centroid_drift_px', float('nan')):.2f} px",
        f"ECC refinement   {align.get('ecc_ok', '-')}",
        f"Ink covered      {100 * align.get('ink_fraction', 0):.2f}% of part",
    ])
    return (_to_rgb(heat), _to_rgb(stages), "\n".join(lines), timing)


# ---------------------------------------------------------------- panel 5

def _first_score(text):
    """The score that decided the verdict, from either verdict layout."""
    m = re.search(r"\| score \| \*\*([\d.]+)\*\*", text)
    if m:
        return m.group(1)
    # Older layout: a per-ROI table. The worst score is what decided it.
    vals = [float(v) for v in re.findall(r"\| ([\d.]+) \| [\d.]+ \|", text)]
    return f"{max(vals):.2f}" if vals else "-"


def _failed_checks(text):
    m = re.search(r"^Failed: (.+)$", text, re.M)
    if m:
        return m.group(1)
    # Older layout: read the rows marked FAIL out of the per-ROI table.
    bad = re.findall(r"^\| ([\w_]+) \| \w+ \|[^|]+\|[^|]+\| \*\*FAIL\*\*",
                     text, re.M)
    return ", ".join(f"`{b}`" for b in bad)


def do_inspect_all(gallery, roi_mode, roi_text, threshold, state,
                   sku=None, progress=gr.Progress()):
    """Score every image in the gallery and list the verdicts.

    The single-part button answers "is this one good"; this answers "how did
    the batch do". No heatmap: that belongs to one part, and picking which
    one to show would be arbitrary.
    """
    paths = []
    for item in (gallery or []):
        path = item[0] if isinstance(item, (list, tuple)) else item
        if isinstance(path, dict):
            path = path.get("path", "")
        if path:
            paths.append(str(path))
    if not paths:
        return (None, None,
                "Nothing to inspect - upload parts or load this part's saved "
                "test images first.", "")

    rows = []
    counts = {"PASS": 0, "FAIL": 0, "RETAKE": 0}
    t0 = time.time()
    for i, path in enumerate(paths):
        progress(i / len(paths),
                 desc=f"{Path(path).name}  ({i + 1}/{len(paths)})")
        out = do_inspect(path, roi_mode, roi_text, threshold, state, sku)
        text = out[2] or ""
        # Match on the word, not the heading level: do_inspect's verdict
        # format has changed more than once and a level-specific parser
        # silently reported every part as unknown.
        first = text.split(chr(10))[0]
        if "RETAKE" in first:
            verdict, score, why = "RETAKE", "-", "photo unusable - retake it"
        elif "PASS" in first:
            verdict, score, why = "PASS", _first_score(text), ""
        elif "FAIL" in first:
            verdict, score, why = "FAIL", _first_score(text), _failed_checks(text)
        else:
            verdict, score, why = "-", "-", first[:60]
        counts[verdict] = counts.get(verdict, 0) + 1
        mark = {"PASS": "PASS", "FAIL": "**FAIL**",
                "RETAKE": "*RETAKE*"}.get(verdict, verdict)
        rows.append(f"| {Path(path).name} | {mark} | {score} | {why} |")

    elapsed = time.time() - t0
    head = (f"# {counts['PASS']} pass, {counts['FAIL']} fail"
            + (f", {counts['RETAKE']} retake" if counts["RETAKE"] else "")
            + chr(10) + chr(10)
            + "| part | result | score | why |" + chr(10)
            + "|---|---|---|---|" + chr(10))
    timing = (f"{len(paths)} part(s) in {elapsed:.1f} s "
              f"({elapsed / len(paths) * 1000:.0f} ms each)")
    return None, None, head + chr(10).join(rows), timing


def _read_labels(sku: str) -> dict[str, int]:
    """Ground truth from data/labels/<sku>.csv (`image,label`, 1 = NG)."""
    path = DATA_DIR / "labels" / f"{sku}.csv"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or parts[1] in ("", "?"):
            continue
        try:
            out[parts[0]] = int(parts[1])
        except ValueError:
            continue                      # header row
    return out


def do_evaluate(sku, roi_mode, roi_text, threshold, state,
                progress=gr.Progress()):
    """Score every labelled test image ONCE; metrics then move with the slider.

    Scoring is the slow part, so it happens here and the results are cached in
    session state. Dragging the threshold afterwards is pure arithmetic.
    """
    reg = state.get("reg")
    if reg is None:
        return (state, "Fit the subspace first (Panel 1).", None, None, None,
                None, None, "")

    labels = _read_labels(sku)
    if not labels:
        return (state,
                f"No ground truth at data/labels/{sku}.csv.\n\n"
                f"Format: `image,label` with 1 = NG and 0 = OK. Without it "
                f"there is nothing to measure the scores against.",
                None, None, None, None, None, "")

    folder = TEST_DIR / sku
    paths = [p for p in sorted(folder.glob("*"))
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
             and p.name in labels]
    if not paths:
        return (state, f"No labelled images found in {folder}.",
                None, None, None, None, None, "")

    cfg = state["cfg"]
    canvas = reg["canvas_size"]
    try:
        rois = [] if roi_mode == WHOLE_PART_MODE else parse_rois(roi_text, canvas)
    except ValueError as exc:
        return (state, f"Could not parse the ROI table: {exc}",
                None, None, None, None, None, "")
    sku_cfg = SKUConfig(sku=sku, canvas_size=canvas, rois=rois,
                        threshold=float(threshold))

    scores, ys, names, retakes = [], [], [], []
    for i, p in enumerate(paths):
        progress(i / len(paths), desc=f"scoring {p.name}")
        res = inspect_image(load_image(p), reg, sku_cfg, cfg, source=p.name)
        if res.verdict == "RETAKE":
            # Nothing was learned about this part; counting it either way
            # would be inventing an outcome.
            retakes.append(p.name)
            continue
        scores.append(res.max_score)
        ys.append(labels[p.name])
        names.append(p.name)

    if not scores:
        return (state, "Every image was RETAKE -- nothing could be scored.",
                None, None, None, None, None, "")

    state = dict(state)
    state["eval"] = {"scores": scores, "labels": ys, "names": names,
                     "retakes": retakes, "sku": sku}

    figs = _render_metrics(state["eval"], float(threshold))
    note = (f"Scored {len(scores)} labelled images "
            f"({sum(ys)} NG, {len(ys) - sum(ys)} OK)"
            + (f", {len(retakes)} excluded as RETAKE: "
               f"{', '.join(retakes)}" if retakes else "")
            + ".\n\nThe threshold slider now updates the metrics instantly -- "
              "no re-scoring.")
    return (state, note) + figs


def do_rescore(threshold, state):
    """Recompute metrics at a new threshold from the cached scores."""
    ev = state.get("eval")
    if not ev:
        return (None, None, None, None, None, "")
    return _render_metrics(ev, float(threshold))


def _render_metrics(ev, threshold):
    scores, ys, names = ev["scores"], ev["labels"], ev["names"]
    c = met.confusion_at(scores, ys, threshold, ev.get("retakes"))
    rows = met.sweep(scores, ys)
    sug = met.suggest_threshold(scores, ys, min_capture=1.0)
    fpr, tpr, auc = met.roc(scores, ys)

    table = [[r["Metric"], r["Engineering name"], r["Formula"],
              r["Value"], r["Counts"], r["Want"]] for r in c.as_rows()]

    lines = [
        f"**At threshold {threshold:.2f}** — "
        f"capture {met._pct(c.capture_rate)}, "
        f"false alarm {met._pct(c.false_alarm_rate)}, "
        f"review load {met._pct(c.review_load)}",
        "",
        "To make the system better: **capture rate UP, false alarm rate DOWN.** "
        "They pull against each other, so every threshold is a compromise.",
    ]
    if c.fn:
        missed = [n for n, s, y in zip(names, scores, ys)
                  if y == 1 and s <= threshold]
        lines += ["", f"**{c.fn} MISSED defect(s):** {', '.join(missed)} — "
                      f"a missed NG normally costs far more than a false alarm."]
    if sug:
        lines += ["", f"Lowest false alarm rate that still catches every NG: "
                      f"**threshold {sug['threshold']:.2f}** "
                      f"({met._pct(sug['false_alarm_rate'])} false alarms)."]
    else:
        lines += ["", "**No threshold catches every NG** — the score "
                      "distributions overlap, so no operating point separates "
                      "them. More or better golden images is the fix, not a "
                      "different threshold."]
    sep = met.separation(scores, ys)
    if np.isfinite(sep):
        lines += ["", f"Separation {sep:.2f} · AUROC "
                      + (f"{auc:.3f}" if np.isfinite(auc) else "n/a")
                      + ". A high AUROC with a small separation means the "
                        "ranking is right but the margin is thin."]

    return (mviz.confusion_matrix(c),
            table,
            mviz.sweep_chart(rows, current=threshold,
                             suggested=sug["threshold"] if sug else None),
            mviz.score_distribution(scores, ys, names, threshold),
            mviz.roc_chart(fpr, tpr, auc),
            "\n".join(lines))


def load_examples(sku, which):
    folder = (GOLDEN_DIR if which == "golden" else TEST_DIR) / sku
    if not folder.exists():
        return []
    return [str(p) for p in sorted(folder.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            and not p.name.upper().startswith("IS_")]


def inspection_sheet(sku: str):
    """The paper inspection sheet for a SKU, if one was photographed.

    Convention: `normal_images/<sku>/IS_*.jpg`. Everywhere else these files are
    deliberately EXCLUDED -- a sheet is a photo of a document, and letting one
    into the golden set would put a blue background and printed Thai text
    inside the model's idea of a normal part.

    Here it is the one place it belongs: drawing ROI boxes means translating
    the sheet's checks into coordinates, and doing that without the sheet in
    front of you is how boxes end up on the wrong feature.
    """
    folder = GOLDEN_DIR / sku
    if not folder.exists():
        return None
    for p in sorted(folder.iterdir()):
        if p.name.upper().startswith("IS_") and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp"}:
            img = load_image(p)
            return _to_rgb(img) if img is not None else None
    return None


def available_skus() -> list[str]:
    """Every SKU with a golden folder or a fitted model.

    A SKU can exist in either place: created here but not yet fitted, or
    registered from the CLI without its photos being copied in.
    """
    names = set()
    for root in (GOLDEN_DIR, MODEL_DIR):
        if root.exists():
            names.update(p.name for p in root.iterdir() if p.is_dir())
    return sorted(names)


def sku_summary(sku: str) -> str:
    """What exists on disk for this SKU -- photos, model, ROIs, labels.

    Laid out across one row rather than down a column: this sits above the
    tabs as a status strip, and a tall table there would push the actual work
    off the screen.
    """
    if not sku:
        return ("*No part selected.* Add a part number above to start, or "
                "pick an existing one.")
    golden = load_examples(sku, "golden")
    tests = load_examples(sku, "test")
    model = (MODEL_DIR / sku / "model.npz").exists()

    default_size = PipelineConfig().features.input_size
    try:
        cfg = SKUConfig.load(sku)
        n_roi = len(cfg.rois)
        # Show the number that will actually be used. "pipeline default" is
        # not something you can compare against a recommendation.
        size = (f"{cfg.input_size}" if cfg.input_size
                else f"{default_size} (default)")
        thr = f"{cfg.threshold:g}"
        # A threshold measured on a different patch grid is not just old, it
        # is wrong for this model -- say so where the number is displayed.
        if not cfg.threshold_measured:
            thr += " (not measured)"
        else:
            try:
                if cfg.threshold_stale_for(load_registration(sku)["grid"]):
                    thr += " ⚠️ stale"
            except (FileNotFoundError, KeyError):
                pass
    except FileNotFoundError:
        n_roi = 0
        size = f"{default_size} (default)"
        thr = "not set"
    labels = (DATA_DIR / "labels" / f"{sku}.csv").exists()

    def tick(ok):
        return "yes" if ok else "**no**"

    return (
        f"| part | golden | test | model | ROIs | input resolution "
        f"| score threshold | NG labels |\n"
        f"|---|---|---|---|---|---|---|---|\n"
        f"| **{sku}** | {len(golden)} | {len(tests)} | {tick(model)} "
        f"| {n_roi} | {size} | {thr} | {tick(labels)} |\n"
    )


def _sku_paths(sku: str) -> list[Path]:
    """Everything on disk belonging to one SKU."""
    return [GOLDEN_DIR / sku, TEST_DIR / sku, MODEL_DIR / sku,
            DATA_DIR / "labels" / f"{sku}.csv"]


def _refresh_skus(selected: str | None = None):
    """Re-read the SKU list and point every picker at the same choices."""
    skus = available_skus()
    value = selected if selected in skus else (skus[0] if skus else None)
    return skus, value


def do_sku_create(new_sku, state):
    """Create an empty SKU: a golden folder is all a new part needs to start."""
    name = (new_sku or "").strip()
    if not name:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                "Enter a part number first.", gr.update())
    # Keep it a safe single path segment: these names become directories.
    if any(c in name for c in '/\\:*?"<>|') or name in (".", ".."):
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                f"`{name}` is not a usable folder name.", gr.update())
    if name in available_skus():
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                f"SKU **{name}** already exists — pick it from the list.",
                gr.update())

    (GOLDEN_DIR / name).mkdir(parents=True, exist_ok=True)
    (TEST_DIR / name).mkdir(parents=True, exist_ok=True)
    skus, value = _refresh_skus(name)
    msg = (f"Created **{name}**.\n\n"
           f"Next: upload golden images below and **Fit subspace**, then "
           f"define its checks in step 2. Uploaded images are copied into "
           f"`normal_images/{name}/` when you save.")
    return (gr.update(choices=skus, value=value),
            gr.update(choices=skus, value=value),
            gr.update(choices=skus, value=value),
            gr.update(value=value), msg, gr.update(value=""))


def do_sku_delete(sku, confirm, state):
    """Delete a SKU and everything belonging to it. Guarded by a checkbox."""
    if not sku:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                "No SKU selected.")
    if not confirm:
        targets = "\n".join(f"- `{p}`" for p in _sku_paths(sku) if p.exists())
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                f"**This permanently deletes {sku}**, including:\n\n{targets}"
                f"\n\nTick *yes, delete it* and press the button again.")

    import shutil

    removed = []
    for p in _sku_paths(sku):
        if not p.exists():
            continue
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed.append(str(p))
        except OSError as exc:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    f"Could not remove `{p}`: {exc}")

    skus, value = _refresh_skus(None)
    listing = "\n".join(f"- `{r}`" for r in removed)
    return (gr.update(choices=skus, value=value),
            gr.update(choices=skus, value=value),
            gr.update(choices=skus, value=value),
            gr.update(value=value or ""),
            f"Deleted **{sku}**:\n\n{listing}")


def do_sku_import(files, sku):
    """Copy the uploaded golden images into this SKU's folder.

    Fitting alone does not persist anything: the images live in Gradio's
    temporary upload area and vanish. A SKU that must survive a restart needs
    its reference photos on disk.
    """
    if not sku:
        return "Select or create a SKU first."
    paths = [Path(f if isinstance(f, str) else f.name) for f in (files or [])]
    if not paths:
        return "No uploaded images to save."

    import shutil

    dest = GOLDEN_DIR / sku
    dest.mkdir(parents=True, exist_ok=True)
    copied, replaced, unchanged = [], [], []
    for p in paths:
        if not p.exists():
            continue
        target = dest / p.name
        if target.exists():
            # Gradio serves preloaded files from its own temp directory, so
            # "same path" is not enough to recognise a file that is already
            # here -- compare size and bytes, or every save re-copies the
            # part's own images over themselves and reports it as new work.
            same = (target.resolve() == p.resolve()
                    or (target.stat().st_size == p.stat().st_size
                        and target.read_bytes() == p.read_bytes()))
            if same:
                unchanged.append(p.name)
                continue
            shutil.copy2(p, target)
            replaced.append(p.name)
            continue
        shutil.copy2(p, target)
        copied.append(p.name)

    parts = []
    if copied:
        parts.append(f"Added **{len(copied)}** new image(s) to "
                     f"`normal_images/{sku}/`.")
    if replaced:
        parts.append(f"Replaced **{len(replaced)}** existing image(s).")
    if unchanged and not (copied or replaced):
        parts.append(f"Nothing to do — all {len(unchanged)} image(s) are "
                     f"already in `normal_images/{sku}/`.")
    elif unchanged:
        parts.append(f"{len(unchanged)} already in place.")
    return " ".join(parts) or "Nothing to copy."


def build_ui():
    skus = available_skus()
    with gr.Blocks(title="Training-free defect inspection") as app:
        gr.Markdown(
            "# Training-free defect inspection\n"
            "Frozen DINOv2 features + a PCA subspace of normal variation. "
            "No gradient training anywhere -- a SKU's model is a mean vector "
            "and an eigenbasis."
        )
        state = gr.State({})

        # The part number is the one thing all three tabs share, so it sits
        # above them rather than inside any one of them. Not gated behind
        # `if skus:` -- with no part registered there would otherwise be no
        # way to make the first one.
        with gr.Row():
            sku_pick = gr.Dropdown(
                skus, value=skus[0] if skus else None,
                label="part number (SKU)", scale=3,
                info="pick an existing part, or add a new one")
            # The field and its button share a Row of their own, so
            # Add sits against the box it acts on rather than being
            # stretched across a column of the outer Row.
            with gr.Column(scale=2):
                new_sku_box = gr.Textbox(
                    label="add a new part number",
                    placeholder="type sku/part no...")
                sku_add_btn = gr.Button("Add", size="sm")
        # Status strip for whichever part is selected -- what exists for it,
        # visible from all three tabs without switching back.
        with gr.Row(elem_id="sku_row"):
            sku_info = gr.Markdown(
                sku_summary(skus[0]) if skus else
                "*No part yet.* Type a part number above and press **Add**.",
                scale=0)
            # Re-reads sku.json: the table is the last thing checked before
            # inspecting, and a save made in another tab is otherwise
            # invisible here.
            sku_refresh_btn = gr.Button("\u21bb", elem_id="sku_refresh",
                                        scale=0, size="sm")

        # No sidebar: every remaining setting belongs to one stage and lives
        # inside it. `threshold` is never read by register_sku, only by
        # inspection and the metrics sweep.
        with gr.Row():
            with gr.Column():
              # Two stages, three groups. The old flat tab strip mixed
              # "teach it" with "judge a part" and put each button's output in
              # a different tab from the button, which is most of what made it
              # hard to follow.
              with gr.Tab("① REGISTRATION — teach it a good part"):
                gr.Markdown(
                    "Runs **once per part**: **1** fit on golden photos → "
                    "**2** draw the sheet's checks on the canvas that fit "
                    "produced → **3** *(only if a check is small)* measure "
                    "the resolution those checks need, then re-fit.\n\n"
                    "Step 1 comes first because ROI boxes are coordinates on "
                    "the canonical canvas, and that canvas does not exist "
                    "until something has been fitted."
                )
                with gr.Tab("1. Golden images → fit"):
                    gr.Markdown(
                        "Selecting a part above loads whatever golden images "
                        "it already has. Upload more (or different ones) and "
                        "re-fit at any time."
                    )
                    # Preloaded with the first part's images. `sku_pick.change`
                    # only fires on a real change, so without this the box sits
                    # empty at launch while the summary above says 6 images.
                    _first_golden = load_examples(skus[0], "golden") if skus else []
                    ref_files = gr.File(file_count="multiple",
                                        file_types=["image"],
                                        label="normal / golden images",
                                        value=_first_golden or None)
                    ref_gallery = gr.Gallery(label="uploaded", columns=6,
                                             height=180,
                                             value=_first_golden or None)
                    with gr.Row():
                        sku_save_imgs = gr.Button(
                            "save these images to the part")
                        sku_import_status = gr.Markdown()

                    fit_btn = gr.Button("Fit subspace", variant="primary")
                    fit_status = gr.Textbox(label="status", lines=6)
                    fit_stats = gr.Textbox(label="fit statistics", lines=10)


                with gr.Tab("2. Inspection-sheet checks (ROIs)"):
                    gr.Markdown(
                        "Draw one box per check on the **inspection sheet**. "
                        "The sheet is the source of truth for what this "
                        "station may reject on — anything not on it is out of "
                        "scope."
                    )
                    sheet_btn = gr.Button("show inspection sheet")
                    # Defaults to ROI mode: it is what production does, and it
                    # is what this tab is for. Defaulting to whole-part meant
                    # boxes drawn here were then ignored by the fit and by
                    # inspection -- the setting quietly undid the work.
                    roi_mode = gr.Radio(
                        [ROI_MODE, WHOLE_PART_MODE], value=ROI_MODE,
                        label="scoring scope",
                        info="ROI boxes: score only the checks below, each "
                             "against its own limit — what production does. "
                             "Whole part: ignore the boxes and score the "
                             "entire part as one region, which is the "
                             "comparison baseline, not a working mode.")
                    roi_mode_note = gr.Markdown(scope_note(ROI_MODE))

                    with gr.Row():
                        sheet_img = gr.Image(
                            label="inspection sheet — WHAT to check",
                            height=430)
                        # Taller than the sheet beside it: this is the one
                        # you have to click accurately on, and a small canvas
                        # makes every box a fight.
                        # elem_id is the hook the drag overlay attaches to.
                        roi_preview = gr.Image(
                            label="aligned reference — WHERE. "
                                  "Click one corner, then the opposite one.",
                            height=560, interactive=False)

                    # This panel is empty until a fit exists, and an unlabelled
                    # empty box reads as a bug rather than a missing step.
                    roi_status = gr.Markdown(
                        "The **aligned reference** on the right is the average "
                        "of this part's golden photos, all warped into one "
                        "standard frame. Boxes drawn on it are coordinates in "
                        "that frame, so they keep pointing at the same feature "
                        "on every future part.\n\n"
                        "*It is blank until you have run step 1 — the frame "
                        "does not exist until something has been fitted.*"
                    )

                    # The single editor for everything about a check. Typing
                    # a coordinate redraws the box immediately, so the numbers
                    # and the picture never disagree.
                    roi_table = gr.Dataframe(
                        headers=["id", "type", "x", "y", "w", "h",
                                 "description (human read only, "
                                 "not used by the model)"],
                        datatype=["str", "str", "number", "number",
                                  "number", "number", "str"],
                        column_count=(7, "fixed"),
                        label="checks — every column is editable; the box "
                              "redraws as you type",
                        interactive=True, wrap=True,
                    )
                    gr.Markdown(
                        "Draw a box on the reference to add a row, then set "
                        "its **id**, **type** (`appearance` or `geometry`) "
                        "and description here. Editing **x / y / w / h** "
                        "moves the box straight away. Delete a row with the "
                        "table's own row controls."
                    )

                    with gr.Row():
                        # Follows the part picked at the top of Registration,
                        # so ROIs cannot be saved onto a different SKU than
                        # the one whose reference image they were drawn on.
                        roi_save_sku = gr.Textbox(
                            label="saving to part",
                            value=skus[0] if skus else "",
                            interactive=False, scale=2)
                        roi_save_btn = gr.Button("Save sku.json",
                                                 variant="primary", scale=1)
                    roi_save_status = gr.Markdown()

                    with gr.Accordion("advanced: edit as text", open=False):
                        roi_text = gr.Textbox(
                            label="one per line: id, type, x, y, w, h",
                            lines=5,
                            placeholder="tab_top_right, appearance, "
                                        "930, 80, 280, 230",
                        )
                        preview_btn = gr.Button("Preview typed ROIs")

                # Third and last, because it needs what steps 1 and 2 produce:
                # golden images to fit on, and an ROI to measure inside. It
                # was previously above step 1, where it could only ever fail.
                with gr.Tab("3. Tuning with NG / OK"):
                    gr.Markdown(
                        "Every part scores differently, so these two settings "
                        "have to be measured for **this** part. Neither can "
                        "be worked out from good parts alone:\n\n"
                        "- **input resolution** — how finely the part is "
                        "sampled. Good parts score about the same at any "
                        "resolution, so only a bad part shows which one "
                        "actually separates good from bad.\n"
                        "- **score threshold** — the pass/fail line. It has "
                        "to sit above every good part and below every bad "
                        "one, so you need at least one of each to place it.\n\n"
                        "Upload photos of **known-bad parts** below — at "
                        "least one per check on the inspection sheet — and "
                        "both are measured instead of guessed. Add "
                        "**known-good parts** as well and the threshold is "
                        "placed inside a gap measured on both sides, rather "
                        "than estimated from the bad parts alone. Needs steps "
                        "1 and 2 first."
                    )
                    ng_files = gr.File(
                        file_count="multiple", file_types=["image"],
                        label="known-bad parts (NG) — parts that fail a check "
                              "on the inspection sheet",
                        height=120)
                    ng_gallery = gr.Gallery(label="NG set for this part",
                                            columns=6, height=150)
                    with gr.Row():
                        ng_save_btn = gr.Button("save these to the part")
                        ng_status = gr.Markdown()

                    gr.Markdown(
                        "Upload photos of **known-good parts** below — "
                        "ordinary parts that pass, not the golden ones from "
                        "step 1. The subspace is *fitted on* the golden set, "
                        "so those score near zero and cannot show where "
                        "everyday good parts land; without real good parts "
                        "the threshold is only an upper bound. Optional, but "
                        "it turns the suggestion into a measured window."
                    )
                    ok_files = gr.File(
                        file_count="multiple", file_types=["image"],
                        label="known-good parts (OK) — parts that pass every "
                              "check on the inspection sheet",
                        height=120)
                    ok_gallery = gr.Gallery(label="OK set for this part",
                                            columns=6, height=150)
                    with gr.Row():
                        ok_save_btn = gr.Button("save these to the part")
                        ok_status = gr.Markdown()

                    tune_ready = gr.Markdown()
                    tune_btn = gr.Button("Measure input resolution and "
                                         "threshold", variant="primary")
                    tune_status = gr.Markdown()

                    with gr.Row():
                        input_size = gr.Dropdown(
                            [448, 560, 672, 784, 896, 1008, 1120, 1232, 1344],
                            value=672, label="input resolution", scale=1,
                            info="patch grid = this / 14 along the long side; "
                                 "step 1 fits at it and inspection reuses it",
                            allow_custom_value=True)
                        # Editable, and mirrored to the Inspection slider so
                        # there is still one number: display-only left it with
                        # no loading bar of its own while the resolution beside
                        # it had one, so the right half looked frozen.
                        tune_thr = gr.Number(
                            label="score threshold", value=6.0, scale=1,
                            info="an ROI fails above this; the same number "
                                 "the Inspection tab uses and sku.json saves")
                    save_tuned_btn = gr.Button(
                        "Save input res and threshold", variant="primary")
                    save_tuned_status = gr.Markdown()
                    gr.Markdown(
                        "*Measuring refits at several resolutions — minutes, "
                        "not seconds. It only fills the two boxes above; "
                        "nothing reaches `sku.json` until you save. If the "
                        "resolution changed, re-fit in step 1 afterwards.*"
                    )

                # Outside the numbered tabs, so it stays put while they change.
                # A rule and a heading say so -- without them it reads as part
                # of whichever step happens to be open.
                gr.HTML(
                    '<hr style="margin:28px 0 8px;border:0;'
                    'border-top:1px solid var(--border-color-primary)">'
                )
                gr.Markdown(
                    "#### Whole part — applies to every step above\n"
                    "These stay put whichever numbered tab you are on."
                )
                # Deleting is not part of fitting, so it does not belong
                # inside step 1: it removes the part's images, model, ROIs
                # and labels together.
                with gr.Accordion("delete this part", open=False):
                    gr.Markdown(
                        "Removes this part's golden images, test images, "
                        "fitted model and NG labels. There is no undo."
                    )
                    sku_del_confirm = gr.Checkbox(
                        False, label="yes, delete it")
                    sku_del_btn = gr.Button("Delete part", variant="stop")
                    sku_del_status = gr.Markdown()

                # Defaults are deliberate and rarely want changing, so this
                # stays collapsed -- subordinate to the steps above, not
                # another one.
                with gr.Accordion("advanced model settings — rarely changed",
                                  open=False):
                    gr.Markdown(
                        "These change how the model is built, so a change "
                        "here needs a re-fit. Inspection inherits whatever "
                        "was fitted. The defaults are the ones the "
                        "experiments settled on."
                    )
                    with gr.Row():
                        with gr.Column():
                            segmenter = gr.Dropdown(
                                ["birefnet", "classical", "auto"],
                                value="birefnet", label="segmenter")
                            backbone = gr.Dropdown(
                                ["vits14", "vitb14", "vitl14", "vitg14"],
                                value="vitl14", label="backbone")
                            aggregation = gr.Dropdown(
                                ["max", "top-3 mean", "top-1% TVaR"],
                                value="max", label="aggregation",
                                info="max / top-3 mean suit small defects; "
                                     "TVaR is the paper's choice and dilutes "
                                     "them")
                        with gr.Column():
                            tau = gr.Slider(0.90, 0.99, value=0.97, step=0.01,
                                            label="PCA variance threshold tau")
                            coverage = gr.Slider(
                                0.5, 1.0, value=0.9, step=0.05,
                                label="patch coverage for in-mask test")
                            position_norm = gr.Checkbox(
                                True, label="per-position normalisation")
                            ink_on = gr.Checkbox(
                                True, label="blue-ink masking")
                            align_on = gr.Checkbox(
                                True, label="alignment",
                                info="turn off to see the no-alignment "
                                     "failure mode")

              with gr.Tab("② INSPECTION — judge one part"):
                gr.Markdown(
                    "Score a part against the fitted model. **The verdict "
                    "appears directly below the button** — no tab switching."
                )
                # The pass/fail line. It lives here rather than with the fit
                # settings because it is applied to scores that already exist,
                # so moving it never needs a re-fit -- which is also what lets
                # the Accuracy tab recompute instantly from cached scores.
                threshold = gr.Slider(
                    0.5, 40.0, value=6.0, step=0.5,
                    label="verdict threshold",
                    info="an ROI fails above this score. Put it in the gap "
                         "between the worst good part and the best bad one; "
                         "the Accuracy tab shows where that gap is.")
                with gr.Accordion("where should this be set?", open=False):
                    thr_note = gr.Markdown(
                        "*No fitted model yet.* Fit in step 1 — the "
                        "recommendation comes from what good parts scored."
                    )
                    thr_apply_btn = gr.Button("use the suggested value",
                                              size="sm")
                with gr.Accordion("settings inherited from the fit",
                                  open=False):
                    fitted_note = gr.Markdown(
                        "*No model fitted yet.* Run **Registration → Fit "
                        "subspace** first; these settings then come from "
                        "that fit."
                    )
                # Two ways in, one destination. Uploading sets `test_file`
                # directly; picking from the gallery sets it too -- previously
                # the gallery was decorative and clicking it did nothing.
                test_file = gr.File(
                    file_types=["image"], label="part to inspect",
                    height=120)
                load_test = gr.Button(
                    "…or load this part's saved test images")
                # `columns` is a maximum, not a fixed count: with one saved
                # image a 6-column grid gave it a sixth of the width and
                # cropped it to a strip. `object_fit="contain"` shows the whole
                # part in whatever cell it gets, and the CSS below caps the
                # cell height so a small set does not fill the screen either.
                # allow_preview: clicking opens the image full-size, which is
                # the only way to actually see a defect before inspecting. The
                # `select` handler still fires, so a click both previews the
                # image and selects it as the part to inspect.
                test_gallery = gr.Gallery(
                    label="saved test images — click one to view it full size "
                          "and select it for inspection",
                    columns=4, height="auto", object_fit="contain",
                    elem_id="test_gallery", allow_preview=True,
                    preview=False)
                with gr.Row():
                    inspect_btn = gr.Button("Inspect one by one",
                                            variant="primary")
                    inspect_all_btn = gr.Button("Inspect all")

                with gr.Group():
                    verdict_md = gr.Markdown()
                    timing_box = gr.Textbox(label="timings and alignment",
                                            lines=6)
                    # No fixed `height`: it crops to fill. The CSS above caps
                    # the size in viewport units and fits the whole frame.
                    heat_img = gr.Image(label="anomaly heatmap on aligned part",
                                        elem_id="heat_img")
                    stages_img = gr.Image(label="stages: raw -> mask -> ink -> "
                                                "aligned -> ROI crops",
                                          elem_id="stages_img")

              with gr.Tab("③ ACCURACY — measure it"):
                    gr.Markdown(
                        "Scores every **labelled** image in "
                        "`test_images/<sku>/` against "
                        "`data/labels/<sku>.csv` (`image,label`, 1 = NG).\n\n"
                        "Scoring runs once; after that the threshold slider "
                        "updates everything instantly. RETAKE images are "
                        "excluded — an unusable photo says nothing about the "
                        "part."
                    )
                    eval_btn = gr.Button("Evaluate", variant="primary")
                    eval_status = gr.Markdown()
                    metrics_table = gr.Dataframe(
                        headers=["Metric", "Engineering name", "Formula",
                                 "Value", "Counts", "Want"],
                        label="the three numbers that decide if this is usable",
                        wrap=True, interactive=False,
                    )
                    metrics_note = gr.Markdown()
                    with gr.Row():
                        cm_img = gr.Image(label="confusion matrix", height=380)
                        roc_img = gr.Image(label="ROC", height=380)
                    sweep_img = gr.Image(label="capture rate vs false alarm "
                                               "rate across thresholds",
                                         height=380)
                    dist_img = gr.Image(label="score of every part, by true "
                                              "label", height=340)

        ctrl = [backbone, input_size, tau, aggregation, position_norm, ink_on,
                align_on, segmenter, coverage, threshold]

        # The threshold suggestion comes from the fitted model's golden
        # scores, so it refreshes whenever a fit lands or the part changes.
        def _thr_note(state, sku):
            return recommend_threshold(state, sku)[1]

        def _thr_apply(state, sku, current):
            value = recommend_threshold(state, sku)[0]
            return current if value is None else value

        thr_apply_btn.click(_thr_apply, [state, sku_pick, threshold], threshold)

        ref_files.change(lambda fs: [f if isinstance(f, str) else f.name
                                     for f in (fs or [])],
                         ref_files, ref_gallery)

        # --- SKU management -------------------------------------------------
        # Every picker points at the same list, so adding or deleting a part
        # cannot leave one tab offering a SKU another has already dropped.
        # One picker, so nothing can drift out of sync.
        sku_selectors = [sku_pick]

        # Selecting a part loads what it already has: its golden images, its
        # inspection sheet, its saved ROIs and a summary of what exists.
        sku_pick.change(lambda s: load_examples(s, "golden"),
                        sku_pick, ref_files)
        sku_pick.change(inspection_sheet, sku_pick, sheet_img)
        sku_pick.change(sku_summary, sku_pick, sku_info)
        sku_pick.change(lambda s: gr.update(value=s or ""),
                        sku_pick, roi_save_sku)
        sku_pick.change(do_roi_load, [sku_pick, state],
                        [state, roi_preview, roi_table, roi_status])

        _add_outputs = sku_selectors + [roi_save_sku, sku_info, new_sku_box]
        sku_add_btn.click(do_sku_create, [new_sku_box, state], _add_outputs)
        # Enter in the field does the same thing as pressing Add.
        new_sku_box.submit(do_sku_create, [new_sku_box, state], _add_outputs)
        sku_save_imgs.click(do_sku_import, [ref_files, sku_pick],
                            sku_import_status).then(
                                sku_summary, sku_pick, sku_info).then(
                                    # Step 3's readiness line counts golden
                                    # images too, so saving them here changes
                                    # what it should say.
                                    _tune_ready, sku_pick, tune_ready)
        sku_del_btn.click(do_sku_delete, [sku_pick, sku_del_confirm, state],
                          sku_selectors + [roi_save_sku, sku_del_status]).then(
                              sku_summary, sku_pick, sku_info)

        def _test_list(sku):
            return load_examples(sku, "test") if sku else []

        def _test_gallery_label(sku):
            """Say how many saved test images this part has.

            The gallery reads from test_images/<sku>/ on disk, so an empty one
            means the folder is empty -- not that a button still needs
            pressing. Saying so removes the guess.
            """
            n = len(_test_list(sku))
            if not n:
                return gr.update(
                    label="saved test images — none in test_images/ for this "
                          "part; upload one above instead")
            return gr.update(
                label=f"saved test images — {n} for this part, click one to "
                      f"inspect it")

        load_test.click(_test_list, sku_pick, test_gallery)
        # Loaded up front and on every part switch, like the NG and OK
        # galleries. Behind a button it looked empty until pressed, and after
        # switching parts it kept showing the previous part's images.
        for _fn, _out in ((_test_list, test_gallery),
                          (_test_gallery_label, test_gallery)):
            app.load(_fn, sku_pick, _out)
            sku_pick.change(_fn, sku_pick, _out)

        def _pick_test(sku, evt: gr.SelectData):
            """A gallery click selects that image as the part to inspect."""
            files = load_examples(sku, "test")
            if evt.index is None or evt.index >= len(files):
                return gr.update()
            return files[evt.index]

        test_gallery.select(_pick_test, sku_pick, test_file)
        sheet_btn.click(inspection_sheet, sku_pick, sheet_img)

        fit_btn.click(
            do_fit,
            [ref_files] + ctrl + [roi_mode, roi_text, state, sku_pick],
            [state, fit_status, roi_preview, stages_img, fit_stats, roi_text,
             roi_table],
        ).then(fitted_settings_note, [state, sku_pick], fitted_note
        ).then(sku_summary, sku_pick, sku_info
        ).then(_thr_note, [state, sku_pick], thr_note)
        preview_btn.click(do_preview_rois, [roi_mode, roi_text, state],
                          [roi_preview, roi_status])

        # Click-to-place ROI editor. Two clicks on the reference define a box;
        # the result is written to sku.json, which is what the pipeline reads.
        roi_preview.select(
            do_roi_click, state,
            [state, roi_preview, roi_table, roi_status],
        )
        # Editing any cell rebuilds the ROI list and redraws the canvas, so a
        # changed coordinate moves its box immediately.
        roi_table.input(do_roi_table_edit, [roi_table, state],
                        [state, roi_preview, roi_status])
        roi_mode.change(scope_note, roi_mode, roi_mode_note)

        roi_save_btn.click(do_roi_save, [roi_save_sku, threshold, state],
                           [state, roi_save_status]).then(
                               sku_summary, sku_pick, sku_info)
        # Writes nothing: applying means re-fitting and saving, both explicit.
        ng_files.change(lambda fs: [f if isinstance(f, str) else f.name
                                    for f in (fs or [])], ng_files, ng_gallery)
        ng_save_btn.click(do_ng_import, [ng_files, sku_pick],
                          ng_status).then(_ng_list, sku_pick, ng_gallery).then(
                              _ng_gallery_label, sku_pick, ng_gallery).then(
                                  _tune_ready, sku_pick, tune_ready)
        ok_files.change(lambda fs: [f if isinstance(f, str) else f.name
                                    for f in (fs or [])], ok_files, ok_gallery)
        ok_save_btn.click(do_ok_import, [ok_files, sku_pick],
                          ok_status).then(_ok_list, sku_pick, ok_gallery).then(
                              _ok_gallery_label, sku_pick, ok_gallery).then(
                                  _tune_ready, sku_pick, tune_ready)

        # Both galleries show what is saved on disk for the selected part, so
        # they have to follow the picker and be populated at launch. Without
        # this they showed the previous part's photos -- or nothing at all --
        # while the measurement quietly used whatever was on disk, and photos
        # already in place looked like they still needed uploading.
        for _fn, _out in ((_ng_list, ng_gallery), (_ok_list, ok_gallery),
                          (_ng_gallery_label, ng_gallery),
                          (_ok_gallery_label, ok_gallery),
                          (_tune_ready, tune_ready)):
            app.load(_fn, sku_pick, _out)
            sku_pick.change(_fn, sku_pick, _out)
        # Three outputs, matching do_tune's three return values. The Inspection
        # slider is deliberately not listed: tune_thr mirrors into it below, so
        # naming it here only desynchronises the arity from the handler.
        tune_btn.click(do_tune, [ref_files, sku_pick, state],
                       [input_size, tune_thr, tune_status])
        tune_thr.change(lambda v: v, tune_thr, threshold)
        threshold.change(lambda v: v, threshold, tune_thr)
        inspect_btn.click(
            do_inspect,
            [test_file, roi_mode, roi_text, threshold, state, sku_pick],
            [heat_img, stages_img, verdict_md, timing_box],
        )
        inspect_all_btn.click(
            do_inspect_all,
            [test_gallery, roi_mode, roi_text, threshold, state, sku_pick],
            [heat_img, stages_img, verdict_md, timing_box],
        )

        metric_out = [cm_img, metrics_table, sweep_img, dist_img, roc_img,
                      metrics_note]
        eval_btn.click(
            do_evaluate,
            [sku_pick, roi_mode, roi_text, threshold, state],
            [state, eval_status] + metric_out,
        )
        # Moving the threshold recomputes from cached scores -- no re-scoring,
        # which is what makes the trade-off explorable rather than a wait.
        threshold.change(do_rescore, [threshold, state], metric_out)

        # Populate everything for the part shown at startup. The `.change`
        # handlers above only fire when the picker actually changes, so a
        # fresh page would otherwise show a part in the dropdown with an empty
        # upload box and no sheet -- contradicting the summary right above it.
        def _initial(sku):
            imgs = load_examples(sku, "golden") if sku else []
            return (imgs or None, imgs or None, inspection_sheet(sku),
                    sku_summary(sku), sku or "")

        app.load(_initial, sku_pick,
                 [ref_files, ref_gallery, sheet_img, sku_info, roi_save_sku])
        app.load(do_roi_load, [sku_pick, state],
                 [state, roi_preview, roi_table, roi_status])
        app.load(_thr_note, [state, sku_pick], thr_note)
        sku_pick.change(_thr_note, [state, sku_pick], thr_note)

        def _size_saved(sku):
            """Show the resolution this part is actually registered at.

            Leaving the dropdown on 672 while sku.json says 896 means a fit
            from the UI silently produces a model whose patch grid disagrees
            with the config -- which disables position normalisation at
            inference without telling anyone.
            """
            try:
                cfg = SKUConfig.load(sku) if sku else None
            except FileNotFoundError:
                return gr.update()
            if cfg is None or cfg.input_size is None:
                return gr.update()
            return cfg.input_size

        app.load(_size_saved, sku_pick, input_size)
        sku_pick.change(_size_saved, sku_pick, input_size)

        def _thr_saved(sku):
            """Start the slider at what this part actually uses, not 6.0."""
            try:
                return SKUConfig.load(sku).threshold if sku else gr.update()
            except FileNotFoundError:
                return gr.update()

        # Save the two measured values, then pull every display back in step
        # so the table, the panel and the sliders cannot disagree.
        save_tuned_btn.click(
            do_save_tuned, [sku_pick, input_size, threshold, state],
            save_tuned_status
        ).then(sku_summary, sku_pick, sku_info
        ).then(_thr_note, [state, sku_pick], thr_note)

        # The summary table is the last thing checked before inspecting.
        sku_refresh_btn.click(sku_summary, sku_pick, sku_info
        ).then(_thr_note, [state, sku_pick], thr_note
        ).then(_thr_saved, sku_pick, threshold
        ).then(_size_saved, sku_pick, input_size)

        app.load(_thr_saved, sku_pick, threshold)
        sku_pick.change(_thr_saved, sku_pick, threshold)
        # The inherited-settings panel reads the saved model too, so it must
        # follow the part picker as well as a fresh fit.
        app.load(fitted_settings_note, [state, sku_pick], fitted_note)
        sku_pick.change(fitted_settings_note, [state, sku_pick], fitted_note)
    return app


if __name__ == "__main__":
    print("\nStarting the demo UI - open http://127.0.0.1:7860 "
          "(the first run downloads model weights, which takes a few "
          "minutes)\n", flush=True)
    build_ui().launch(css=UI_CSS)
