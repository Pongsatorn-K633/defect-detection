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
import time
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from inspector import metrics as met
from inspector import metricsviz as mviz
from inspector import subspace as sub_mod
from inspector import viz
from inspector.config import (DATA_DIR, GOLDEN_DIR, TEST_DIR, PipelineConfig,
                              fit_signature)
from inspector.inspect import inspect_image
from inspector.pipeline import load_image
from inspector.register import register_sku
from inspector.roi import ROI, SKUConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("demo")

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


def parse_rois(text: str, canvas) -> list[ROI]:
    """ROI table: one per line, `id, type, x, y, w, h`."""
    rois = []
    for i, line in enumerate(text.splitlines(), 1):
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
           roi_text, state, progress=gr.Progress()):
    """Fit the subspace on the uploaded reference images."""
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

    try:
        images = [load_image(p) for p in paths]
        t0 = time.time()
        reg = register_sku("demo", images, cfg, [Path(p).name for p in paths],
                           parse_rois(roi_text, (10_000, 10_000))
                           if roi_mode == ROI_MODE else [],
                           progress=say)
        fit_s = time.time() - t0
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

    status = "\n".join(msgs[-6:] + [""] + warn) if warn else "\n".join(msgs[-6:])
    default_roi = roi_text
    if roi_mode == ROI_MODE and not roi_text.strip():
        cw, ch = canvas
        default_roi = (f"# id, type, x, y, w, h   (canvas {cw} x {ch})\n"
                       f"example, appearance, {cw // 4}, {ch // 4}, "
                       f"{cw // 4}, {ch // 4}\n")

    draft = state.get("draft_rois", [])
    return (state, status, _roi_canvas(reg, draft), _to_rgb(sheet), stats,
            gr.update(value=default_roi), _roi_rows(draft))


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


def _roi_canvas(reg, rois, pending=None):
    """Aligned reference with the saved boxes drawn, plus any pending corner."""
    img = viz.draw_rois(reg["ref_image"], [r.to_dict() for r in rois])
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


def do_roi_click(rid, rtype, note, state, evt: gr.SelectData):
    """Two clicks define a box: first corner, then the opposite one.

    Click-to-place rather than drag: Gradio's editor returns painted pixels
    rather than coordinates, so a drag would mean detecting a rectangle from a
    bitmap. Two clicks give exact numbers with nothing to infer.
    """
    reg = state.get("reg")
    if reg is None:
        return state, None, [], "Fit the subspace first (Panel 1)."

    x, y = int(evt.index[0]), int(evt.index[1])
    state = dict(state)
    rois = list(state.get("draft_rois", []))
    first = state.get("roi_pending")

    if first is None:
        state["roi_pending"] = (x, y)
        return (state, _roi_canvas(reg, rois, (x, y)), _roi_rows(rois),
                f"Corner 1 at ({x}, {y}). Now click the opposite corner.")

    x0, y0 = first
    state["roi_pending"] = None
    bx, by = min(x0, x), min(y0, y)
    bw, bh = abs(x - x0), abs(y - y0)
    if bw < 8 or bh < 8:
        return (state, _roi_canvas(reg, rois), _roi_rows(rois),
                "That box is too small — start again with two corners further "
                "apart.")

    name = (rid or "").strip() or f"check_{len(rois) + 1}"
    if any(r.id == name for r in rois):
        return (state, _roi_canvas(reg, rois), _roi_rows(rois),
                f"An ROI called '{name}' already exists — give this one a "
                f"different name.")

    kind = "geometry" if str(rtype).startswith("geometry") else "appearance"
    tol = {"area_pct": 10.0, "fill_ratio_pct": 12.0,
           "corner_angle_deg": 15.0} if kind == "geometry" else {}
    rois.append(ROI(name, kind, (bx, by, bw, bh), tolerance=tol,
                    note=(note or "").strip()))
    state["draft_rois"] = rois
    return (state, _roi_canvas(reg, rois), _roi_rows(rois),
            f"Added **{name}** [{kind}] at ({bx}, {by}, {bw}, {bh}). "
            f"{len(rois)} ROI(s) defined.")


def do_roi_undo(state):
    reg = state.get("reg")
    if reg is None:
        return state, None, [], "Fit the subspace first."
    state = dict(state)
    rois = list(state.get("draft_rois", []))
    state["roi_pending"] = None
    if not rois:
        return state, _roi_canvas(reg, rois), [], "Nothing to undo."
    gone = rois.pop()
    state["draft_rois"] = rois
    return (state, _roi_canvas(reg, rois), _roi_rows(rois),
            f"Removed **{gone.id}**. {len(rois)} left.")


def do_roi_clear(state):
    reg = state.get("reg")
    state = dict(state)
    state["draft_rois"] = []
    state["roi_pending"] = None
    canvas = _roi_canvas(reg, []) if reg else None
    return state, canvas, [], "Cleared all ROIs."


def do_roi_load(sku, state):
    """Load an existing sku.json so it can be edited rather than retyped."""
    reg = state.get("reg")
    try:
        cfg = SKUConfig.load(sku)
    except FileNotFoundError:
        return (state, _roi_canvas(reg, []) if reg else None, [],
                f"No saved config for {sku} yet — draw boxes and save.")
    state = dict(state)
    state["draft_rois"] = list(cfg.rois)
    state["roi_pending"] = None
    return (state, _roi_canvas(reg, cfg.rois) if reg else None,
            _roi_rows(cfg.rois),
            f"Loaded {len(cfg.rois)} ROI(s) from {sku}, threshold "
            f"{cfg.threshold}. Edit and save to overwrite.")


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
        input_size = existing.input_size
    except FileNotFoundError:
        input_size = None

    cfg = SKUConfig(sku=sku, canvas_size=reg["canvas_size"],
                    ref_area_px=reg["ref_stats"].get("area", 0.0),
                    rois=rois, threshold=float(threshold),
                    input_size=input_size)
    path = cfg.save()
    return (state,
            f"Saved **{len(rois)} ROI(s)** to `{path}`.\n\n"
            f"Geometry ROIs need their nominals measured, so re-run "
            f"registration before inspecting:\n\n"
            f"```\npython -m inspector.register --sku {sku} "
            f"--segmenter birefnet\n```")


# -------------------------------------------------------------------- panel 3

def do_inspect(test_file, roi_mode, roi_text, threshold, state):
    """Score one test image against the cached subspace. Never refits."""
    reg = state.get("reg")
    if reg is None:
        return (None, None, "Fit the subspace first (Panel 1).", "")
    if test_file is None:
        return (None, None, "Upload a test image.", "")

    cfg = state["cfg"]
    canvas = reg["canvas_size"]
    try:
        rois = ([] if roi_mode == WHOLE_PART_MODE
                else parse_rois(roi_text, canvas))
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
    if not GOLDEN_DIR.exists():
        return []
    return sorted(p.name for p in GOLDEN_DIR.iterdir() if p.is_dir())


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

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                segmenter = gr.Dropdown(["birefnet", "classical", "auto"],
                                        value="birefnet", label="segmenter")
                backbone = gr.Dropdown(["vits14", "vitb14", "vitl14", "vitg14"],
                                       value="vitl14", label="backbone")
                input_size = gr.Dropdown(
                    [448, 512, 672, 896, 1120], value=672,
                    label="input resolution",
                    info="higher = more patches per ROI. A small check on a "
                         "thin part can need 1120; see SKU 8983980631")
                tau = gr.Slider(0.90, 0.99, value=0.97, step=0.01,
                                label="PCA variance threshold tau")
                aggregation = gr.Dropdown(
                    ["max", "top-3 mean", "top-1% TVaR"], value="max",
                    label="aggregation",
                    info="max / top-3 mean suit small defects; TVaR is the "
                         "paper's choice and dilutes them")
                coverage = gr.Slider(0.5, 1.0, value=0.9, step=0.05,
                                     label="patch coverage for in-mask test")
                position_norm = gr.Checkbox(True, label="per-position normalisation")
                ink_on = gr.Checkbox(True, label="blue-ink masking")
                align_on = gr.Checkbox(True, label="alignment",
                                       info="turn off to see the "
                                            "no-alignment failure mode")
                threshold = gr.Slider(0.5, 40.0, value=6.0, step=0.5,
                                      label="verdict threshold")

            with gr.Column(scale=3):
                with gr.Tab("1. Reference images"):
                    ref_files = gr.File(file_count="multiple",
                                        file_types=["image"],
                                        label="normal / golden images")
                    ref_gallery = gr.Gallery(label="uploaded", columns=6,
                                             height=180)
                    if skus:
                        gr.Markdown("Load a sample set:")
                        with gr.Row():
                            sku_pick = gr.Dropdown(skus, value=skus[0],
                                                   label="SKU", scale=2)
                            load_golden = gr.Button("load golden", scale=1)
                    fit_btn = gr.Button("Fit subspace", variant="primary")
                    fit_status = gr.Textbox(label="status", lines=6)
                    fit_stats = gr.Textbox(label="fit statistics", lines=10)

                with gr.Tab("2. ROI selection"):
                    gr.Markdown(
                        "Draw one box per check on the **inspection sheet**. "
                        "The sheet is the source of truth for what this "
                        "station may reject on — anything not on it is out of "
                        "scope."
                    )
                    if skus:
                        with gr.Row():
                            sheet_sku = gr.Dropdown(
                                skus, value=skus[0], label="SKU", scale=2)
                            sheet_btn = gr.Button("show inspection sheet",
                                                  scale=1)
                    roi_mode = gr.Radio([ROI_MODE, WHOLE_PART_MODE],
                                        value=WHOLE_PART_MODE,
                                        label="scoring scope")

                    with gr.Row():
                        sheet_img = gr.Image(
                            label="inspection sheet — WHAT to check",
                            height=430)
                        roi_preview = gr.Image(
                            label="aligned reference — WHERE. "
                                  "Click two opposite corners to add a box.",
                            height=430, interactive=False)

                    roi_status = gr.Markdown()

                    with gr.Row():
                        roi_id = gr.Textbox(label="check name", scale=2,
                                            placeholder="tab_top_right")
                        roi_type = gr.Dropdown(
                            ["appearance — texture, marks, scratches",
                             "geometry — silhouette shape, missing tabs"],
                            value="appearance — texture, marks, scratches",
                            label="check type", scale=3)
                    roi_note = gr.Textbox(
                        label="description — quote the inspection sheet",
                        placeholder="ชิ้นงานต้องมีปีกอยู่ด้านขวาบน  /  "
                                    "the part must have the tab at top right")

                    with gr.Row():
                        roi_undo_btn = gr.Button("undo last box")
                        roi_clear_btn = gr.Button("clear all")
                        roi_load_btn = gr.Button("load saved sku.json")

                    roi_table = gr.Dataframe(
                        headers=["id", "type", "x", "y", "w", "h",
                                 "description"],
                        label="ROIs defined so far",
                        interactive=False, wrap=True,
                    )

                    with gr.Row():
                        roi_save_sku = gr.Textbox(
                            label="save as SKU",
                            value=skus[0] if skus else "",
                            scale=2)
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

                with gr.Tab("3. Test image"):
                    test_file = gr.File(file_types=["image"],
                                        label="part to inspect")
                    if skus:
                        with gr.Row():
                            test_sku = gr.Dropdown(skus, value=skus[0],
                                                   label="SKU", scale=2)
                            load_test = gr.Button("list test images", scale=1)
                        test_gallery = gr.Gallery(label="test set", columns=6,
                                                  height=160)
                    inspect_btn = gr.Button("Inspect", variant="primary")

                with gr.Tab("4. Output"):
                    verdict_md = gr.Markdown()
                    timing_box = gr.Textbox(label="timings and alignment",
                                            lines=6)
                    heat_img = gr.Image(label="anomaly heatmap on aligned part",
                                        height=420)
                    stages_img = gr.Image(label="stages: raw -> mask -> ink -> "
                                                "aligned -> ROI crops",
                                          height=260)

                with gr.Tab("5. Metrics"):
                    gr.Markdown(
                        "Scores every **labelled** image in "
                        "`test_images/<sku>/` against "
                        "`data/labels/<sku>.csv` (`image,label`, 1 = NG).\n\n"
                        "Scoring runs once; after that the threshold slider "
                        "updates everything instantly. RETAKE images are "
                        "excluded — an unusable photo says nothing about the "
                        "part."
                    )
                    with gr.Row():
                        eval_sku = gr.Dropdown(skus, value=skus[0] if skus else None,
                                               label="SKU", scale=2)
                        eval_btn = gr.Button("Evaluate", variant="primary",
                                             scale=1)
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

        ref_files.change(lambda fs: [f if isinstance(f, str) else f.name
                                     for f in (fs or [])],
                         ref_files, ref_gallery)
        if skus:
            load_golden.click(lambda s: load_examples(s, "golden"),
                              sku_pick, ref_files)
            load_test.click(lambda s: load_examples(s, "test"),
                            test_sku, test_gallery)
            sheet_btn.click(inspection_sheet, sheet_sku, sheet_img)
            # Loading a sample SKU's golden set also pulls up its sheet, so
            # the two are side by side without a second click.
            load_golden.click(inspection_sheet, sku_pick, sheet_img)
            sku_pick.change(inspection_sheet, sku_pick, sheet_img)

        fit_btn.click(
            do_fit,
            [ref_files] + ctrl + [roi_mode, roi_text, state],
            [state, fit_status, roi_preview, stages_img, fit_stats, roi_text,
             roi_table],
        )
        preview_btn.click(do_preview_rois, [roi_mode, roi_text, state],
                          [roi_preview, roi_status])

        # Click-to-place ROI editor. Two clicks on the reference define a box;
        # the result is written to sku.json, which is what the pipeline reads.
        roi_preview.select(
            do_roi_click, [roi_id, roi_type, roi_note, state],
            [state, roi_preview, roi_table, roi_status],
        )
        roi_undo_btn.click(do_roi_undo, state,
                           [state, roi_preview, roi_table, roi_status])
        roi_clear_btn.click(do_roi_clear, state,
                            [state, roi_preview, roi_table, roi_status])
        roi_load_btn.click(do_roi_load, [roi_save_sku, state],
                           [state, roi_preview, roi_table, roi_status])
        roi_save_btn.click(do_roi_save, [roi_save_sku, threshold, state],
                           [state, roi_save_status])
        inspect_btn.click(
            do_inspect,
            [test_file, roi_mode, roi_text, threshold, state],
            [heat_img, stages_img, verdict_md, timing_box],
        )

        metric_out = [cm_img, metrics_table, sweep_img, dist_img, roc_img,
                      metrics_note]
        eval_btn.click(
            do_evaluate,
            [eval_sku, roi_mode, roi_text, threshold, state],
            [state, eval_status] + metric_out,
        )
        # Moving the threshold recomputes from cached scores -- no re-scoring,
        # which is what makes the trade-off explorable rather than a wait.
        threshold.change(do_rescore, [threshold, state], metric_out)
    return app


if __name__ == "__main__":
    build_ui().launch()
