"""Stage 1 -- registration. Build a SKU's normal model from golden photos.

    python -m inspector.register --sku 8983864641 --images "normal_images/8983864641/*.jpeg"
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import cancel
from . import features as feat_mod
from . import geometry as geom_mod
from . import subspace as sub_mod
from .config import GOLDEN_DIR, PipelineConfig
from .pipeline import IMAGE_SUFFIXES, build_reference, load_image
from .roi import ROI, SKUConfig, model_path, patch_grid_box, sku_dir

log = logging.getLogger(__name__)


def _valid_patch_grid(res, cfg, grid_h, grid_w):
    return feat_mod.patch_mask_from_pixel_mask(
        res.aligned_valid_mask, grid_h, grid_w, cfg.features.patch_coverage
    )


def register_sku(
    sku: str,
    images: list[np.ndarray],
    cfg: PipelineConfig,
    sources: list[str] | None = None,
    rois: list[ROI] | None = None,
    progress=None,
    prepared=None,
) -> dict:
    """Fit the normal subspace for one SKU. No gradient training anywhere:
    the resulting "model" is a mean vector and an eigenbasis.

    `prepared` is a `build_reference` result to reuse. Segmentation and
    alignment do not depend on the feature resolution, so a sweep over input
    sizes does them once and passes them in, instead of re-segmenting every
    golden photo (twice, for the reference refinement) at every size.
    """
    t0 = time.time()
    sources = sources or [f"image_{i}" for i in range(len(images))]

    def say(msg):
        log.info(msg)
        if progress:
            progress(msg)

    if prepared is None:
        say(f"preprocessing {len(images)} golden images")
        prepared = build_reference(images, cfg, sources)
    ref_stats, ref_img, ref_mask, results = prepared
    canvas = tuple(ref_stats["canvas_size"])

    posed = [r for r in results if r.info.get("pose_rejected")]
    if posed:
        say("WARNING: left out " + ", ".join(
            f"{Path(r.source).name} (overlap {r.info.get('iou', 0):.2f})"
            for r in posed) +
            f" -- the part is not lying like the other golden photos "
            f"(needs {cfg.align.min_iou:.2f}). Retake them flat, as on the "
            f"inspection sheet, or remove them.")
    usable = [r for r in results if r.aligned_valid_mask is not None
              and r.aligned_valid_mask.max() > 0
              and not r.info.get("pose_rejected")]
    if not usable:
        raise ValueError("no golden image survived preprocessing")
    # Pose rejections were already named above, with the reason.
    dropped = [Path(r.source).name for r in results
               if r not in usable and r not in posed]
    if dropped:
        say(f"dropped {len(dropped)} unusable golden image(s): {dropped}")

    say(f"extracting DINOv2 {cfg.features.backbone} features")
    fmaps, vgrids = [], []
    for r in usable:
        cancel.check()                    # features are the other slow step
        f = feat_mod.extract_patch_features(r.aligned_image, cfg.features)
        gh, gw = f.shape[:2]
        fmaps.append(f)
        vgrids.append(_valid_patch_grid(r, cfg, gh, gw))
    grid_h, grid_w = fmaps[0].shape[:2]

    per_image = [f[v] for f, v in zip(fmaps, vgrids)]
    counts = [len(p) for p in per_image]
    say(f"patch grid {grid_h}x{grid_w}, valid patches per image: "
        f"min {min(counts)} max {max(counts)}")

    # Golden-set contamination screen: a defective golden part would put its
    # defect permanently inside the normal subspace.
    screen = {"rejected": []}
    if cfg.subspace.loo_enabled and len(per_image) >= 3:
        screen = sub_mod.leave_one_out_screen(per_image, cfg.subspace)
        if screen["rejected"]:
            names = [Path(usable[i].source).name for i in screen["rejected"]]
            say(f"leave-one-out screen rejected {names} as outliers")
            keep = [i for i in range(len(per_image)) if i not in screen["rejected"]]
            if len(keep) >= 2:
                usable = [usable[i] for i in keep]
                fmaps = [fmaps[i] for i in keep]
                vgrids = [vgrids[i] for i in keep]
                per_image = [per_image[i] for i in keep]
            else:
                say("too few golden images would remain; keeping all")
                screen["rejected"] = []

    dupes = sub_mod.near_duplicate_pairs(per_image)
    if dupes:
        say(f"{len(dupes)} near-duplicate golden pair(s): these cost fit time "
            f"without widening the subspace")

    stacked = np.concatenate(per_image, axis=0)
    say(f"fitting subspace on {len(stacked)} valid patches "
        f"({stacked.shape[1]}-D)")
    model = sub_mod.fit_subspace(stacked, cfg.subspace.tau,
                                 cfg.subspace.max_components)
    say(f"retained r={model['r']} components "
        f"({100 * model['explained']:.1f}% variance)")

    # Per-position statistics of the golden residuals. Edges, holes and formed
    # features always score high on normal parts; standardising them away
    # drops the noise floor substantially.
    pos_stats = None
    if cfg.subspace.position_normalize:
        resid_maps = np.stack([sub_mod.residual_score(f, model) for f in fmaps])
        stats = {"residual": sub_mod.fit_position_stats(
            resid_maps, np.stack(vgrids), cfg.subspace.position_sigma_floor)}
        if cfg.subspace.use_mahalanobis:
            maha_maps = np.stack([sub_mod.mahalanobis_score(f, model)
                                  for f in fmaps])
            stats["mahalanobis"] = sub_mod.fit_position_stats(
                maha_maps, np.stack(vgrids), cfg.subspace.position_sigma_floor)
        pos_stats = stats

    geom_nominals = measure_geometry_nominals(
        rois, [r.aligned_part_mask for r in usable], say)

    # Golden score distribution: what "normal" looks like on this SKU, which
    # is what any sensible threshold has to be set against.
    # One score map per golden photo, shared by every check and the whole
    # part: it does not depend on the ROI, and recomputing it per ROI cost
    # (checks + 1) full Mahalanobis passes per photo.
    score_maps = []
    for f in fmaps:
        cancel.check()
        score_maps.append(
            sub_mod.score_maps(f, model, cfg.subspace, pos_stats)["score"])

    golden_scores = {}
    eff_rois = rois or []
    for roi in eff_rois:
        if roi.type != "appearance":
            continue
        gy0, gy1, gx0, gx1 = patch_grid_box(roi.box, canvas, grid_h, grid_w)
        vals = []
        for s, v in zip(score_maps, vgrids):
            vsub = v[gy0:gy1, gx0:gx1]
            if vsub.any():
                vals.append(sub_mod.aggregate(s[gy0:gy1, gx0:gx1], vsub,
                                              cfg.subspace.aggregation))
        golden_scores[roi.id] = vals

    golden_scores["whole_part"] = [
        sub_mod.aggregate(s, v, cfg.subspace.aggregation)
        for s, v in zip(score_maps, vgrids)]

    # Last chance: a cancel during the subspace fit is honoured here, before
    # the caller gets a model it would save.
    cancel.check()
    elapsed = time.time() - t0
    say(f"registration finished in {elapsed:.1f}s")

    return {
        "sku": sku,
        "model": model,
        "position_stats": pos_stats,
        "ref_stats": ref_stats,
        "ref_image": ref_img,
        "ref_mask": ref_mask,
        "canvas_size": canvas,
        "grid": (grid_h, grid_w),
        "results": usable,
        # Features of `results`, in the same order, for callers that score the
        # golden parts again (the resolution sweep). Not saved to disk.
        "golden_features": list(zip(fmaps, vgrids)),
        "n_valid_patches": int(len(stacked)),
        "per_image_patches": counts,
        "geometry_nominals": geom_nominals,
        "golden_scores": golden_scores,
        "loo_screen": screen,
        "near_duplicates": dupes,
        "fit_seconds": elapsed,
        "config": cfg,
        "sources": [Path(r.source).name for r in usable],
    }


def save_registration(reg: dict, sku_cfg: SKUConfig | None = None) -> Path:
    """Persist the model. It is small: a mean vector, an eigenbasis and stats."""
    sku = reg["sku"]
    d = sku_dir(sku)
    d.mkdir(parents=True, exist_ok=True)

    payload = {
        "mu": reg["model"]["mu"],
        "C": reg["model"]["C"],
        "eigenvalues": reg["model"]["eigenvalues"],
        "r": reg["model"]["r"],
        "canvas_size": np.array(reg["canvas_size"]),
        "grid": np.array(reg["grid"]),
        "ref_mask": reg["ref_mask"],
        "ref_image": reg["ref_image"],
        "ref_stats_json": json.dumps(reg["ref_stats"]),
        "config_json": json.dumps(reg["config"].to_dict()),
        "geometry_nominals_json": json.dumps(reg["geometry_nominals"]),
        "golden_scores_json": json.dumps(reg["golden_scores"]),
        "sources_json": json.dumps(reg["sources"]),
        "n_valid_patches": reg["n_valid_patches"],
    }
    # The aligned golden outlines, so a box drawn or moved later can have its
    # shape check measured from disk -- after a restart too -- instead of
    # needing a re-fit. Stored 0/1: geometry only ever reads mask > 0, and a
    # binary mask compresses to a few KB.
    if reg.get("results"):
        payload["golden_masks"] = np.stack(
            [(r.aligned_part_mask > 0) for r in reg["results"]]
        ).astype(np.uint8)
    if reg["position_stats"]:
        for key, st in reg["position_stats"].items():
            payload[f"pos_{key}_mean"] = st["mean"]
            payload[f"pos_{key}_std"] = st["std"]
            payload[f"pos_{key}_counts"] = st["counts"]

    np.savez_compressed(model_path(sku), **payload)

    # Anything the fit determines is written back; anything the operator chose
    # is preserved. Reading the config off disk when the caller did not supply
    # one stops a re-fit from wiping ROIs and the threshold -- work that takes
    # NG photos to establish and is not recoverable from the model.
    if sku_cfg is None:
        try:
            sku_cfg = SKUConfig.load(sku)
        except FileNotFoundError:
            sku_cfg = SKUConfig(sku=sku, canvas_size=reg["canvas_size"],
                                ref_area_px=reg["ref_stats"]["area"])
    cfg = sku_cfg
    cfg.canvas_size = reg["canvas_size"]
    cfg.ref_area_px = reg["ref_stats"]["area"]
    # input_size is a property OF the fit, not a preference alongside it: the
    # subspace and the position statistics are tied to the patch grid they
    # were fitted on, and inspection has to use the same one. Recording what
    # was actually fitted keeps sku.json and model.npz from drifting apart --
    # a mismatch silently disables position normalisation at inference.
    cfg.input_size = int(reg["config"].features.input_size)
    cfg.save()

    cv2.imwrite(str(d / "reference_aligned.png"), reg["ref_image"])
    cv2.imwrite(str(d / "reference_mask.png"), reg["ref_mask"])
    return model_path(sku)


def measure_geometry_nominals(rois, aligned_masks, say=None) -> dict:
    """Nominal outline measurements per geometry ROI, from the golden masks.

    Shared by a full fit and by `update_geometry_nominals`, so a box drawn
    after fitting is measured exactly as the fit itself would measure it.
    """
    say = say or log.info
    nominals = {}
    for roi in (rois or []):
        # A presence check is an absolute limit and needs no nominal; the
        # flat-part warning below is also exactly wrong for it -- a spot that
        # is solid on every good part is what "must be metal" checks.
        if roi.type != "geometry" or geom_mod.is_presence(roi.tolerance):
            continue
        nominal = geom_mod.fit_nominal(
            [geom_mod.measure_roi_geometry(m, roi.box) for m in aligned_masks])
        # The nominal only means something for this exact box. Recording it
        # lets inspection refuse to compare a moved box against numbers
        # measured somewhere else, instead of scoring noise until the re-fit.
        nominal["box"] = [int(v) for v in roi.box]
        nominals[roi.id] = nominal

        # A geometry ROI only means something where the silhouette has an edge
        # to measure. Sitting on solid interior it fills the box on every
        # golden part, so it can never catch a defect -- an embossed mark or a
        # scratch does not move the outline -- yet it still fires whenever
        # segmentation wobbles, failing a good part on a check that was
        # measuring nothing. Cheap to leave in (~0.03 ms), but the false
        # rejects are not cheap, so say so here where it can still be removed.
        # Threshold set from the sample SKUs: a box straddling a real outline
        # measures 0.40 (tab) or 0.62 (chamfered corner), while a box on flat
        # interior measures 0.97 (the embossed RH mark). 0.95 sits clear of
        # both, and the check is advisory -- it never blocks registration.
        frac = nominal.get("area_frac", float("nan"))
        if np.isfinite(frac) and (frac > 0.95 or frac < 0.05):
            where = "solid part" if frac > 0.5 else "background"
            say(f"WARNING: geometry ROI {roi.id!r} sits on flat {where} "
                f"(area {100 * frac:.1f}% of its box on every golden part). "
                f"The silhouette has no edge here, so this check cannot catch "
                f"a defect -- it can only misfire on segmentation noise. If "
                f"the check is a surface mark, delete this ROI and keep the "
                f"appearance one.")
    return nominals


def update_geometry_nominals(reg: dict, rois, say=None) -> bool:
    """Measure shape checks for new or moved boxes without re-fitting.

    Boxes are drawn on the canvas a fit produces, so a first-time setup used
    to need fit -> draw -> fit again just to measure the outline inside the
    boxes. The fit already aligned every golden photo, and the outline is all
    a shape check needs, so it is measured from those here and written into
    the saved model. The appearance model is untouched.

    The outlines come from the fit held in memory, or from the saved model
    (`golden_masks`), so this works after a restart too. Returns False --
    changing nothing -- only for a model fitted before outlines were saved,
    or when the saved model is from a different fit; a re-fit is the answer.
    """
    results = reg.get("results")
    if results:
        masks = [r.aligned_part_mask for r in results]
    elif reg.get("golden_masks") is not None:
        masks = list(reg["golden_masks"])
    else:
        return False
    p = model_path(reg["sku"])
    if not p.exists():
        return False
    with np.load(p, allow_pickle=False) as z:
        saved = {k: z[k] for k in z.files}
    # Only ever amend the fit these masks came from.
    if json.loads(str(saved["sources_json"])) != list(reg["sources"]) or \
            tuple(int(v) for v in saved["canvas_size"]) != tuple(reg["canvas_size"]):
        return False
    nominals = measure_geometry_nominals(rois, masks, say)
    saved["geometry_nominals_json"] = np.array(json.dumps(nominals))
    np.savez_compressed(p, **saved)
    reg["geometry_nominals"] = nominals
    return True


def load_registration(sku: str) -> dict:
    """Load a saved SKU model."""
    p = model_path(sku)
    if not p.exists():
        raise FileNotFoundError(
            f"no model for SKU {sku}; run `python -m inspector.register --sku {sku}`"
        )
    z = np.load(p, allow_pickle=False)
    model = {"mu": z["mu"], "C": z["C"], "eigenvalues": z["eigenvalues"],
             "r": int(z["r"])}

    pos = {}
    for key in ("residual", "mahalanobis"):
        if f"pos_{key}_mean" in z:
            pos[key] = {"mean": z[f"pos_{key}_mean"],
                        "std": z[f"pos_{key}_std"],
                        "counts": z[f"pos_{key}_counts"]}

    return {
        "sku": sku,
        "model": model,
        "position_stats": pos or None,
        "canvas_size": tuple(int(v) for v in z["canvas_size"]),
        "grid": tuple(int(v) for v in z["grid"]),
        "ref_mask": z["ref_mask"],
        "ref_image": z["ref_image"],
        "ref_stats": json.loads(str(z["ref_stats_json"])),
        "config": PipelineConfig.from_dict(json.loads(str(z["config_json"]))),
        "geometry_nominals": json.loads(str(z["geometry_nominals_json"])),
        "golden_scores": json.loads(str(z["golden_scores_json"])),
        "sources": json.loads(str(z["sources_json"])),
        "n_valid_patches": int(z["n_valid_patches"]),
        # Absent from models fitted before the outlines were saved.
        "golden_masks": z["golden_masks"] if "golden_masks" in z else None,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Register a SKU from golden photos")
    ap.add_argument("--sku", required=True)
    ap.add_argument("--images", nargs="*", default=None,
                    help="glob(s); defaults to normal_images/<sku>/")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--segmenter", default=None,
                    choices=["classical", "birefnet", "auto"])
    args = ap.parse_args()

    cfg = PipelineConfig()
    if args.backbone:
        cfg.features.backbone = args.backbone
    if args.input_size:
        cfg.features.input_size = args.input_size
    if args.tau:
        cfg.subspace.tau = args.tau
    if args.segmenter:
        cfg.segment.method = args.segmenter

    if args.images:
        paths = [Path(p) for pat in args.images for p in glob.glob(pat)]
    else:
        folder = GOLDEN_DIR / args.sku
        paths = [p for p in sorted(folder.glob("*"))
                 if p.suffix.lower() in IMAGE_SUFFIXES
                 and not p.name.startswith("IS_")]
    if not paths:
        print("no golden images found")
        return 1

    try:
        sku_cfg = SKUConfig.load(args.sku)
        rois = sku_cfg.rois
    except FileNotFoundError:
        sku_cfg, rois = None, []

    if sku_cfg is not None:
        # Per-SKU overrides (e.g. a higher input resolution for an
        # under-resolved check) must apply here as well as at inspection --
        # fitting at one resolution and testing at another changes the patch
        # grid and silently invalidates the position statistics.
        cfg = sku_cfg.apply_to(cfg)
        if sku_cfg.input_size:
            print(f"SKU override: input_size={sku_cfg.input_size}")

    images = [load_image(p) for p in paths]
    reg = register_sku(args.sku, images, cfg, [p.name for p in paths], rois)
    out = save_registration(reg, sku_cfg)

    print(f"\nsaved {out}")
    print(f"  canvas       {reg['canvas_size']}")
    print(f"  patch grid   {reg['grid']}")
    print(f"  valid patches {reg['n_valid_patches']}")
    print(f"  components r  {reg['model']['r']}")
    print(f"  fit time     {reg['fit_seconds']:.1f}s")
    gs = reg["golden_scores"].get("whole_part", [])
    if gs:
        print(f"  golden whole-part scores: min {min(gs):.2f} "
              f"max {max(gs):.2f}  (a threshold must sit above max)")
    if not rois:
        print(f"\nNo ROIs defined yet. Run: "
              f"python -m inspector.annotate --sku {args.sku}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
