"""Single source of truth: every pipeline sub-stage, one consistent run.

Replaces the earlier split measurement (bench.py + bench_pre.py +
bench_align.py), which stitched numbers from separate processes with
different warmup states. Here each sub-stage is timed inside one loop,
after a discarded warmup pass, so the parts sum to the whole.

Execution-unit labels are derived from what the code actually calls:
  GPU      - torch forward pass on cuda
  CPU-MT   - OpenCV op that honours cv2.setNumThreads
  CPU-ST   - effectively serial (ECC iterative solve)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from inspector import align as align_mod  # noqa: E402
from inspector import inkmask, segment as seg_mod  # noqa: E402
from inspector import features as feat_mod  # noqa: E402
from inspector import subspace as sub_mod  # noqa: E402
from inspector.pipeline import load_image  # noqa: E402
from inspector.register import load_registration  # noqa: E402

# stage -> execution unit. Kept next to the timing so the two cannot drift.
UNIT = {
    "load":            "CPU-ST  (disk + JPEG decode)",
    "segment":         "GPU     (BiRefNet fwd)",
    "ink":             "CPU-MT  (HSV threshold)",
    "align_similarity": "CPU-MT  (moments + warp)",
    "align_ecc":       "CPU-ST  (ECC iterative solve)",
    "align_warp_final": "CPU-MT  (2x warpPerspective)",
    "features":        "GPU     (DINOv2 fwd)",
    "score":           "CPU-MT  (PCA residual)",
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Run:
    """Times sub-stages of one full inspection, reproducing pipeline.py."""

    def __init__(self, img, cfg, reg):
        self.img, self.cfg, self.reg = img, cfg, reg
        self.t = {}

    def _m(self, name, fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        self.t[name] = (time.perf_counter() - t0) * 1000
        return out

    def go(self, path):
        cfg, reg = self.cfg, self.reg
        img = self._m("load", lambda: load_image(path))

        part = self._m("segment",
                       lambda: seg_mod.segment_part(img, cfg.segment))
        ink = self._m("ink", lambda: inkmask.blue_ink_mask(img, cfg.ink))

        ref_stats = reg["ref_stats"]
        canvas = ref_stats["canvas_size"]
        ref_mask, ref_image = reg["ref_mask"], reg["ref_image"]

        def _sim():
            M, _ = align_mod.similarity_to_canonical(part, ref_stats, ref_mask)
            ai = align_mod.warp_image(img, M, canvas)
            ap = align_mod.warp_image(part, M, canvas, nearest=True)
            return M, ai, ap

        M, aligned_img, aligned_part = self._m("align_similarity", _sim)

        H_ok = False
        if cfg.align.refine_homography:
            def _ecc():
                return align_mod.refine_homography(
                    aligned_img, ref_image, aligned_part, ref_mask, cfg.align)
            H, ecc_info = self._m("align_ecc", _ecc)
            H_ok = bool(ecc_info.get("ecc_ok"))
        else:
            self.t["align_ecc"] = 0.0
            H = None

        def _final():
            m = M
            if H_ok:
                m = align_mod.compose(M, H)
                ai = align_mod.warp_image(img, m, canvas)
                ap = align_mod.warp_image(part, m, canvas, nearest=True)
            else:
                ai, ap = aligned_img, aligned_part
            aink = align_mod.warp_image(ink, m, canvas, nearest=True)
            return ai, ap, aink

        ai, ap, aink = self._m("align_warp_final", _final)
        valid = inkmask.valid_mask(ap, aink)

        fmap = self._m("features", lambda: feat_mod.extract_patch_features(
            ai, cfg.features))
        gh, gw = fmap.shape[:2]
        # position_stats were fitted on the SKU's own grid; reuse them only
        # when the grid still matches, else the normalisation cannot apply.
        ps = reg.get("position_stats")
        if ps and tuple(reg["grid"]) != (gh, gw):
            ps = None
        self._m("score", lambda: sub_mod.score_maps(
            fmap, reg["model"], cfg.subspace, ps))
        self.grid = (gh, gw)
        self.ecc_ok = H_ok
        return self.t


def bench(sku, input_size, repeats, warmup=2):
    reg = load_registration(sku)
    cfg = reg["config"]
    fitted = cfg.features.input_size
    cfg.features.input_size = input_size
    imgs = []
    for sub in ("test_images", "ng_images", "ok_images", "normal_images"):
        d = REPO / sub / sku
        if d.is_dir():
            imgs += sorted(p for p in d.iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        return None

    feat_mod.load_backbone(cfg.features)
    for _ in range(warmup):
        Run(None, cfg, reg).go(imgs[0])

    rows, grid, ecc_n = [], None, 0
    peak = 0
    # Sweep every image, repeats each: ECC cost depends on initial pose
    # error, so a single photo understates the spread.
    for path in imgs:
        for _ in range(repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            r = Run(None, cfg, reg)
            try:
                rows.append(dict(r.go(path)))
            except Exception as exc:
                print(f"    skip {path.name}: {type(exc).__name__} {exc}")
                break
            grid = r.grid
            ecc_n += int(r.ecc_ok)
            if torch.cuda.is_available():
                peak = max(peak, torch.cuda.max_memory_allocated())
    ecc_ok = f"{ecc_n}/{len(rows)}"

    agg = {k: statistics.median([r[k] for r in rows]) for k in rows[0]}
    return {
        "sku": sku, "input_size": input_size, "fitted_size": fitted,
        "is_shipping_config": input_size == fitted, "grid": grid,
        "grid_patches": grid[0] * grid[1], "ecc_ok": ecc_ok,
        "n_images": len(imgs), "n": len(rows),
        "peak_vram_gb": peak / 1e9, "stages": agg,
        "total": sum(agg.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sizes", default="fitted",
                    help="'fitted' (each SKU's own input_size) or a "
                         "comma-separated list, e.g. 896,1120")
    args = ap.parse_args()

    print(f"opencv {cv2.__version__} threads={cv2.getNumThreads()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU {torch.cuda.get_device_name(0)} "
              f"{p.total_memory/1e9:.1f}GB {p.multi_processor_count}SM")
    print(f"torch {torch.__version__}  median of {args.repeats}, "
          f"2 warmup discarded\n")

    out = []
    skus = sorted(m.parent.name
                  for m in (REPO / "data" / "models").glob("*/model.npz"))
    for sku in skus:
        if args.sizes == "fitted":
            sizes = [load_registration(sku)["config"].features.input_size]
        else:
            sizes = [int(v) for v in args.sizes.split(",")]
        for size in sizes:
            r = bench(sku, size, args.repeats)
            if not r:
                continue
            out.append(r)
            print(f"[{r['sku']}] input={size} grid={r['grid'][0]}x"
                  f"{r['grid'][1]}={r['grid_patches']} ecc_ok={r['ecc_ok']} "
                  f"vram={r['peak_vram_gb']:.2f}GB")
            for k, v in r["stages"].items():
                print(f"    {k:<18}{v:8.1f} ms   {UNIT[k]}")
            print(f"    {'TOTAL':<18}{r['total']:8.1f} ms\n")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
