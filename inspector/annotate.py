"""ROI annotation: draw the boxes that scope each inspection-sheet check.

This is configuration, not labelling -- it takes minutes per SKU and needs no
defective examples.

    python -m inspector.annotate --sku 8983864641

Controls: drag to draw a box, then type an id at the prompt and pick the
check type. `u` undoes, `s` saves, `q` quits without saving.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from . import viz
from .register import load_registration
from .roi import ROI, SKUConfig

log = logging.getLogger(__name__)


def annotate_interactive(sku: str) -> int:
    reg = load_registration(sku)
    ref = reg["ref_image"]
    try:
        sku_cfg = SKUConfig.load(sku)
    except FileNotFoundError:
        sku_cfg = SKUConfig(sku=sku, canvas_size=reg["canvas_size"])

    # Scale to something that fits a laptop screen; boxes are stored in
    # canvas coordinates so the annotation stays valid at any display size.
    h, w = ref.shape[:2]
    scale = min(1.0, 1100 / max(w, h))
    disp_size = (int(w * scale), int(h * scale))

    rois: list[ROI] = list(sku_cfg.rois)
    state = {"drag": None, "box": None}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["box"] = (*state["drag"], x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0 = state["drag"]
            state["drag"] = None
            if abs(x - x0) > 5 and abs(y - y0) > 5:
                state["box"] = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            else:
                state["box"] = None

    win = f"ROIs - {sku}"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__)

    while True:
        canvas = cv2.resize(ref, disp_size, interpolation=cv2.INTER_AREA)
        for r in rois:
            x, y, bw, bh = [int(v * scale) for v in r.box]
            color = (0, 200, 255) if r.type == "geometry" else (0, 220, 0)
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
            canvas = viz.label(canvas, f"{r.id} [{r.type[:4]}]",
                               (x + 3, max(y - 6, 14)), color, 0.5)
        if state["box"] and len(state["box"]) == 4 and state["drag"]:
            x0, y0, x1, y1 = state["box"]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)

        canvas = viz.label(canvas, f"{len(rois)} ROI(s)   drag=new  "
                                   f"u=undo  s=save  q=quit",
                           (8, disp_size[1] - 12))
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if state["box"] and not state["drag"] and len(state["box"]) == 4:
            bx, by, bw, bh = state["box"]
            state["box"] = None
            box = tuple(int(v / scale) for v in (bx, by, bw, bh))
            rid = input("  ROI id (blank to discard): ").strip()
            if rid:
                kind = input("  type [a]ppearance / [g]eometry (default a): ")
                kind = "geometry" if kind.strip().lower().startswith("g") \
                    else "appearance"
                tol = {}
                if kind == "geometry":
                    tol = {"area_pct": 8.0, "fill_ratio_pct": 10.0,
                           "corner_angle_deg": 12.0}
                rois.append(ROI(rid, kind, box, tolerance=tol))
                print(f"  added {rid} [{kind}] at {box}")

        if key == ord("q"):
            print("quit without saving")
            break
        if key == ord("u") and rois:
            print(f"  removed {rois.pop().id}")
        if key == ord("s"):
            sku_cfg.rois = rois
            sku_cfg.canvas_size = reg["canvas_size"]
            path = sku_cfg.save()
            print(f"saved {len(rois)} ROI(s) to {path}")
            print("Re-run register so geometry nominals are measured for the "
                  "new ROIs.")
            break

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Draw ROI boxes for a SKU")
    ap.add_argument("--sku", required=True)
    ap.add_argument("--list", action="store_true", help="print ROIs and exit")
    args = ap.parse_args()

    if args.list:
        cfg = SKUConfig.load(args.sku)
        print(f"SKU {cfg.sku}  canvas {cfg.canvas_size}  "
              f"threshold {cfg.threshold}")
        for r in cfg.rois:
            print(f"  {r.id:<16}{r.type:<12}{r.box}  {r.tolerance or ''}")
        return 0

    return annotate_interactive(args.sku)


if __name__ == "__main__":
    raise SystemExit(main())
