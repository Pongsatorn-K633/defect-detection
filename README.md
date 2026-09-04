# Training-Free Defect Inspection

Visual inspection of stamped metal automotive brackets photographed at a manual
station. Frozen DINOv2 patch features plus a PCA subspace of normal variation
(SubspaceAD); the anomaly score is the reconstruction residual off that
subspace. **No gradient training anywhere** — a per-SKU model is a mean vector
and an eigenbasis.

## Setup

```bash
conda env create -f environment.yml
conda activate defect-detection
```

BiRefNet weights (~900 MB) download from HuggingFace on first use, DINOv2
(~1.1 GB for ViT-L) from torch.hub. Both are cached afterwards.

## Quick start

```bash
# 1. Check alignment holds. Nothing downstream matters if this fails.
python -m experiments.exp1_alignment_repeatability --sku 8983864641

# 2. Fit the normal subspace from the golden photos
python -m inspector.register --sku 8983864641 --segmenter birefnet

# 3. Draw the ROI boxes for the inspection-sheet checks
python -m inspector.annotate --sku 8983864641

# 4. Inspect
python -m inspector.inspect --sku 8983864641 --image part.jpg --out result/
python -m inspector.inspect --sku 8983864641 --dir test_images/8983864641 --csv r.csv

# Or use the experimentation UI
python demo.py
```

## Results on the sample data (SKU 8983864641)

Six golden images, nine test images, two of them defective (square top-left
corner and a bent-up top-right tab — both checks on the inspection sheet).

| Experiment | Result |
|---|---|
| 1 — alignment repeatability | **PASS**: mean IoU 0.967, mean centroid drift 1.75 px, max 3.15 px (criterion: IoU > 0.95, drift < 5 px) |
| 2 — ROI vs whole part vs full frame | **ROI-cropped AUROC 1.000** vs whole-part 0.667 vs full-frame 0.667 |
| 3 — threshold sweep | 100% recall at **0% false-call rate**, part threshold ≈ 18.6 (set to 16.0 for margin) |
| 4 — specular sensitivity | not run — needs the varied-lighting photo set |
| 5 — reference count | at k=3, **varied images give AUROC 1.000 and near-duplicates 0.333** |

Final verdicts at threshold 16.0: both defective parts caught, all six good
parts pass, and the one photo with the part running off the frame edge is
flagged INVALID rather than FAIL.

Experiment 2 is the go/no-go for the whole approach, and it is unambiguous:
**ROI cropping is what makes this work.** Scoring the whole part is barely
better than chance here, because the defect signal is swamped by edge and
lighting variation across the rest of the silhouette.

## The non-negotiable design rules, and where they live

1. **One preprocessing path.** [`pipeline.preprocess()`](inspector/pipeline.py)
   is the only implementation; registration and inspection both call it. There
   is no second copy to drift out of sync.
2. **The mask reaches the scoring.** [`features.patch_mask_from_pixel_mask()`](inspector/features.py)
   drops any patch less than 90% inside the valid mask, from the PCA fit *and*
   from test scoring.
3. **No per-image min-max normalisation.** Scores are raw residuals; the only
   standardisation is against per-position statistics measured once at
   registration, which is a fixed transform. Heatmap colour scales are anchored
   on the threshold, never on the image's own range.
4. **Rotation augmentation off by default** (`n_aug = 0`). Alignment handles pose.
5. **Aggregation is `max` or top-3 mean**, not top-1% TVaR. TVaR is available
   in the UI for comparison and it does dilute these defects.

## Layout

```
inspector/
  config.py      every knob that changes a result
  segment.py     BiRefNet or classical segmentation -> part mask
  inkmask.py     blue marker-pen exclusion
  align.py       canonical pose normalisation  <- the load-bearing module
  pipeline.py    THE shared preprocessing path
  features.py    frozen DINOv2 patch features
  subspace.py    PCA fit, residual + Mahalanobis scoring, LOO screen
  geometry.py    silhouette measurements for shape checks
  roi.py         SKU config and ROI definitions
  register.py    stage 1 — build a SKU model
  inspect.py     stage 2 — score a part
  annotate.py    draw ROI boxes
  viz.py         overlays, heatmaps, contact sheets
demo.py          Gradio experimentation UI
experiments/     exp1..exp5 plus check_stages.py
data/
  models/<sku>/  sku.json + model.npz + reference images
  labels/<sku>.csv   ground truth for the test set (image,label; 1 = defective)
  specular/<sku>/    good-part photos with varied lighting, for Experiment 4
normal_images/<sku>/   golden photos + IS_*.jpg inspection sheet
test_images/<sku>/     parts to inspect
```

## Notes and caveats

**The labels are provisional.** `data/labels/8983864641.csv` was read off the
aligned corner crops by eye against the inspection sheet, not supplied by QC.
Every accuracy number above rests on it. Replace it with the real verdicts
before quoting anything externally.

**The sample size is small.** Two defective and six good parts. AUROC 1.000 on
eight images means the method separates *these* defects cleanly; it is not a
production accuracy estimate, and the perfect score should be read as "no
overlap yet", not "solved".

**The subspace is still starving at k=6.** Experiment 5 shows the mean test
score falling 342 → 115 → 21 as k goes 1 → 2 → 4, with no sign of a plateau.
More golden images — spanning lighting and highlight positions, not
near-duplicates — should lower the absolute scores and widen the margin.
The spec's expected useful range of 8–16 looks right; we cannot yet reach it.

**Coverage beats count, measurably.** At the same k=3, deliberately varied
images gave AUROC 1.000 while near-duplicates gave 0.333 — worse than chance.
This is the single most actionable finding for how to collect golden sets.

**Specular sensitivity is untested.** Experiment 4 has no data. This is the
spec's predicted dominant failure mode and it remains unmeasured. Photograph
one good part ~20 times with the light and camera deliberately moved, drop
them in `data/specular/<sku>/`, and run it.

**SKU 8983980631 is registered but unvalidated.** Alignment passes (IoU 0.986,
drift 0.34 px) and the `mark_rh` ROI is positioned on the stamp, but there are
no test images yet. That check is the hardest in the set — a low-contrast
raised stamp only about four DINOv2 patches wide at 672 px input. If recall is
poor, the fallbacks are tiling the ROI or moving that check to classical CV.

**Two ECC rejections per fit are expected.** The homography refinement is
validated by corner displacement and falls back to the similarity transform
when it diverges; the warning is the safety net working, not a failure.

**Geometry checks are the right tool for silhouette defects.** A missing or
bent tab is a "logical anomaly", SubspaceAD's documented weak spot. The
geometry ROIs measure the aligned mask contour directly and caught the corner
defect independently of the appearance score.
