"""Stage 1-3 sanity dump: segmentation, ink masking, alignment.

Writes a contact sheet per SKU plus an aligned-overlay image, so the masks and
the aligned stack can be inspected visually before any scoring is trusted.

    python -m experiments.check_stages --sku 8983864641 --segmenter classical
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspector import viz
from inspector.config import GOLDEN_DIR, OUTPUT_DIR, TEST_DIR, PipelineConfig
from inspector.pipeline import build_reference, load_image, preprocess


def gather(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in exts and not p.name.startswith("IS_"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="8983864641")
    ap.add_argument("--segmenter", default="classical",
                    choices=["classical", "birefnet", "auto"])
    ap.add_argument("--no-ecc", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.segment.method = args.segmenter
    cfg.align.refine_homography = not args.no_ecc

    golden_dir = GOLDEN_DIR / args.sku
    files = gather(golden_dir)
    if not files:
        print(f"no golden images in {golden_dir}")
        return 1
    print(f"SKU {args.sku}: {len(files)} golden images, segmenter={args.segmenter}")

    images = [load_image(p) for p in files]
    ref_stats, ref_img, ref_mask, results = build_reference(
        images, cfg, [p.name for p in files]
    )
    print(f"canonical canvas: {ref_stats['canvas_size']}  "
          f"ref area {ref_stats['area']:.0f} px  "
          f"elongation {ref_stats['elongation']:.2f}")

    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "stage_check" / args.sku
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- golden set ---")
    print(f"{'image':<18}{'area':>9}{'ink%':>7}{'IoU':>7}{'drift':>8}{'ecc':>6}  notes")
    ious, drifts = [], []
    for r in results:
        iou = r.info.get("iou")
        drift = r.info.get("centroid_drift_px")
        if iou is not None:
            ious.append(iou)
            drifts.append(drift)
        print(f"{Path(r.source).name:<18}"
              f"{r.info.get('area', 0):>9.0f}"
              f"{100 * r.info.get('ink_fraction', 0):>7.2f}"
              f"{iou if iou is not None else float('nan'):>7.3f}"
              f"{drift if drift is not None else float('nan'):>8.2f}"
              f"{str(r.info.get('ecc_ok', '-')):>6}  "
              f"{'' if r.ok else r.reject_reason}")

    if ious:
        print(f"\nmean IoU {np.mean(ious):.4f}   min {np.min(ious):.4f}")
        print(f"mean centroid drift {np.mean(drifts):.2f} px   "
              f"max {np.max(drifts):.2f} px")
        ok = np.mean(ious) > 0.95 and np.max(drifts) < 5
        print(f"Experiment-1 criterion (IoU>0.95, drift<5px): "
              f"{'PASS' if ok else 'FAIL'}")

    # Per-image stage strips.
    for r in results:
        sheet = viz.stage_sheet(r, Path(r.source).stem)
        cv2.imwrite(str(out_dir / f"stages_{Path(r.source).stem}.png"), sheet)

    # Aligned stack: how tightly do the silhouettes agree?
    stack = np.zeros(ref_mask.shape, np.float32)
    for r in results:
        if r.aligned_part_mask is not None:
            stack += (r.aligned_part_mask > 0).astype(np.float32)
    stack /= max(len(results), 1)
    heat = cv2.applyColorMap((stack * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(out_dir / "aligned_overlay.png"),
                viz.label(heat, "aligned mask agreement (yellow = all agree)"))

    mean_img = np.mean([r.aligned_image.astype(np.float32) for r in results
                        if r.aligned_image is not None], axis=0)
    cv2.imwrite(str(out_dir / "aligned_mean.png"), mean_img.astype(np.uint8))
    cv2.imwrite(str(out_dir / "aligned_grid.png"),
                viz.grid([(r.aligned_image, Path(r.source).stem) for r in results],
                         cols=4))

    # Test images through the identical path.
    test_dir = TEST_DIR / args.sku
    if test_dir.exists():
        tfiles = gather(test_dir)
        if tfiles:
            print(f"\n--- test set ({len(tfiles)}) ---")
            tres = []
            for p in tfiles:
                r = preprocess(load_image(p), cfg, ref_stats=ref_stats,
                               ref_mask=ref_mask, ref_image=ref_img, source=p.name)
                tres.append(r)
                print(f"{p.name:<16}"
                      f"IoU {r.info.get('iou', float('nan')):>6.3f}  "
                      f"drift {r.info.get('centroid_drift_px', float('nan')):>6.2f}  "
                      f"ink {100 * r.info.get('ink_fraction', 0):>5.2f}%  "
                      f"{'OK' if r.ok else 'REJECT: ' + r.reject_reason}")
                cv2.imwrite(str(out_dir / f"test_stages_{p.stem}.png"),
                            viz.stage_sheet(r, p.stem))
            cv2.imwrite(str(out_dir / "test_aligned_grid.png"),
                        viz.grid([(r.aligned_image, Path(r.source).stem)
                                  for r in tres], cols=4))

    print(f"\nwrote visualisations to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
