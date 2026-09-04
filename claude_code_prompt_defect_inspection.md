# Build Spec: Training-Free Defect Inspection Pipeline

## Context

I need a training-free visual defect inspection system for stamped/pressed metal automotive
brackets. Target scope is 200 SKUs across 6 production lines. Inspection happens at a **manual
station** — an operator photographs a part with a phone/camera on a dark workbench. There is no
conveyor, no fixture, and no controlled lighting. Framing, distance, in-plane rotation, and
lighting all vary shot to shot.

The core method is **SubspaceAD** (arXiv 2602.23013): frozen DINOv2 patch features + PCA subspace
of normal variation, anomaly score = reconstruction residual off that subspace. No gradient
training anywhere in the system. Per-SKU "model" is just a mean vector and an eigenbasis.

Checks come from existing paper inspection sheets, e.g. "mark RH must be present at this
location", "top-right tab must exist", "top-left corner must be chamfered".

**Build this as a working experimental codebase, not a production service.** I need to run
experiments and see numbers before committing to the architecture.

---

## Non-negotiable design rules

These are the rules that make or break the system. Violating any of them silently destroys
accuracy, so enforce them in code, not in comments.

1. **Registration and inspection must run an identical preprocessing path.** Same segmentation,
   same alignment, same crop, same resize, same masking. Write this once as a shared function
   and call it from both paths. Never duplicate the logic.
2. **The mask must reach the scoring, not just the display.** Patches outside the part mask are
   excluded from the PCA fit AND from test scoring. Do not grey out the background and feed the
   full image — boundary patches still vary and will produce a halo of false positives.
3. **No min-max normalization of the anomaly map.** The paper normalizes per-image for its
   figures. That makes a fixed threshold impossible. Score on raw residual.
4. **Rotation augmentation off by default** (`n_aug = 0`, configurable). Alignment handles pose.
   The paper's 30 random rotations only widen the subspace and hide defects.
5. **Image-level aggregation is `max` or top-3 mean**, not the paper's top-1% TVaR. Our defects
   are tiny relative to the ROI; averaging over the top 1% of patches dilutes them away.

---

## Architecture

### Stage 1 — Registration (once per SKU)

```
8-16 golden part photos
  -> BiRefNet segmentation           -> part mask
  -> blue-ink exclusion mask         -> valid mask
  -> canonical alignment (from mask) -> warped image + warped mask
  -> operator draws ROI boxes        -> saved to SKU config
  -> per appearance ROI: crop -> resize -> DINOv2 -> in-mask patches -> PCA fit -> save mu, C
  -> per geometry ROI:  measure mask contour -> save nominal + tolerance
```

### Stage 2 — Inspection (per part)

```
1 photo
  -> identical preprocessing path
  -> apply SKU's ROI boxes
  -> appearance ROI: DINOv2 -> in-mask patches -> residual vs saved subspace -> ROI score
  -> geometry ROI:   measure mask contour -> compare to tolerance
  -> part verdict = fail if any ROI exceeds threshold
  -> output: verdict + per-ROI scores + heatmap overlay
```

---

## Module specification

### `segment.py`

```python
def segment_part(image: np.ndarray) -> np.ndarray:
    """BiRefNet foreground segmentation -> binary mask (uint8, 0/255).
    Keep largest connected component only. Fill holes smaller than 0.5% of part area
    (stamped holes in the part should NOT be filled — they are real geometry;
    only fill segmentation noise). Morphological open/close to clean edges."""
```

- Model: `ZhengPeng7/BiRefNet` via HuggingFace transformers, or RMBG-2.0 as fallback.
- Frozen, zero-shot, never fine-tuned.
- Cache the loaded model at module level; do not reload per image.

### `inkmask.py`

Operators mark parts with blue marker pen. Ink position and shape vary per part and must be
excluded from scoring or it becomes the dominant false positive.

```python
def blue_ink_mask(image: np.ndarray) -> np.ndarray:
    """HSV threshold for blue marker ink -> binary mask of ink pixels.
    Dilate slightly to catch anti-aliased edges. Returns mask to EXCLUDE."""
```

- Tune HSV bounds against the sample images; expose them in config.
- Final valid mask = `part_mask AND NOT ink_mask`.

### `align.py`

This is the most important module. Get it right before anything else.

```python
def canonical_transform(mask: np.ndarray, ref_stats: dict) -> np.ndarray:
    """Compute 2x3 similarity transform mapping this part to canonical pose.
    - centroid from image moments        -> translation
    - principal axis from PCA on mask pixel coordinates -> rotation
    - sqrt(ref_area / this_area)         -> scale
    Returns affine matrix."""

def resolve_axis_ambiguity(mask, angle, ref_mask) -> float:
    """The principal axis is 180-degree ambiguous. Resolve by testing both
    orientations and picking the one with higher IoU against the reference mask."""

def refine_homography(warped_img, ref_img, warped_mask, ref_mask) -> np.ndarray:
    """Polish the similarity alignment into a homography to absorb mild
    out-of-plane tilt. Use cv2.findTransformECC (MOTION_HOMOGRAPHY) on masked
    grayscale. If ECC fails to converge, fall back to the similarity transform
    and log a warning."""
```

- `ref_stats` (reference area, canonical canvas size) is saved at registration from the first
  golden image and reused for every subsequent part of that SKU.
- Canonical canvas: fixed pixel size per SKU, e.g. long side 1024 px.

### `features.py`

```python
def extract_patch_features(image: np.ndarray, layers=(22,23,24,25,26,27,28)) -> np.ndarray:
    """Frozen DINOv2. Mean-pool tokens across the given layers (paper Sec 3.2).
    Returns (h, w, D) patch feature map."""

def patch_mask_from_pixel_mask(mask: np.ndarray, h: int, w: int, coverage=0.9) -> np.ndarray:
    """Downsample pixel mask to patch grid. A patch is valid only if at least
    `coverage` of its pixels are inside the mask. Partial-coverage boundary
    patches are EXCLUDED — they are a major false positive source."""
```

- Backbone configurable: `dinov2_vits14 / vitb14 / vitl14 / vitg14`. Default ViT-L for
  experiments (fast), ViT-G for final numbers.
- Input resolution configurable, default 672.

### `subspace.py`

```python
def fit_subspace(features: np.ndarray, tau: float = 0.97) -> dict:
    """Stack valid patch features from all golden images -> mean mu, covariance,
    eigendecomposition. Retain r components where cumulative explained variance >= tau.
    Returns {'mu': (D,), 'C': (D, r), 'eigenvalues': (r,)}."""

def residual_score(features, model) -> np.ndarray:
    """S(x) = ||x - (mu + C C^T (x - mu))||^2 per patch. Returns (h, w) map."""

def mahalanobis_score(features, model) -> np.ndarray:
    """In-subspace distance: ||Lambda^-0.5 C^T (x - mu)||^2.
    Catches defects that lie inside the subspace but far from the mean."""

def fit_position_stats(golden_score_maps: np.ndarray) -> dict:
    """Per-patch-position mean and std of residual across golden images.
    Edges, holes and stamped features always score high on normal parts;
    normalizing them away drops the noise floor substantially."""

def normalized_score(score_map, position_stats) -> np.ndarray:
    """z(p) = (S(p) - mu_res[p]) / sigma_res[p]"""
```

- Final ROI score: `max(z_residual, z_mahalanobis)`, aggregated over the ROI by `max` or
  top-3 mean (configurable).

### `geometry.py`

Some checks from the inspection sheets are shape questions, not appearance questions
(e.g. "top-left corner is chamfered", "tab present"). Measuring the aligned mask contour is
cheaper and more reliable than asking an anomaly model.

```python
def measure_roi_geometry(mask, roi_box) -> dict:
    """Return area, contour length, corner angles, bounding box fill ratio
    for the mask region inside the ROI."""

def check_geometry(measured: dict, nominal: dict, tolerance: dict) -> bool
```

### `roi.py` + `annotate.py`

- SKU config as JSON:

```json
{
  "sku": "AAPICO-898386-4641",
  "canvas_size": [1024, 768],
  "ref_area_px": 184320,
  "rois": [
    {"id": "mark_rh",    "type": "appearance", "box": [412, 388, 96, 96],  "threshold": null},
    {"id": "tab_top_r",  "type": "geometry",   "box": [640, 200, 160, 140],"tolerance": {"area_pct": 8}}
  ]
}
```

- `annotate.py`: simple OpenCV or Gradio window showing the aligned golden image, operator
  drags boxes, types an id, picks appearance/geometry. Saves the JSON. This is configuration,
  not labeling — it takes minutes per SKU.

---

## CLI

```bash
# Register a SKU from golden photos
python -m inspector.register --sku AAPICO-898386-4641 --images data/golden/898386-4641/*.jpg

# Draw / edit ROI boxes
python -m inspector.annotate --sku AAPICO-898386-4641

# Inspect one part
python -m inspector.inspect --sku AAPICO-898386-4641 --image test.jpg --out result/

# Batch inspect a folder
python -m inspector.inspect --sku ... --dir data/test/ --csv results.csv
```

Outputs per inspection: verdict, per-ROI scores, and a heatmap overlay PNG showing which ROI
triggered.

---

## Demo UI

Build a simple Gradio app (`demo.py`) so I can experiment without the CLI and show the
approach to the client. Keep it minimal — this is an experimentation tool, not the production
operator interface.

**Layout:**

- **Panel 1 — Reference (normal) images.** Multi-file upload, accepts 1 to N images. Show
  thumbnails with a per-image remove button, and a live count. A "Fit subspace" button runs
  registration on whatever is currently uploaded.
- **Panel 2 — ROI selection.** After fitting, display the aligned reference image with an
  interactive box-draw tool. Operator drags one or more boxes, names each, picks
  appearance/geometry. Also allow **"whole part"** as a mode so I can compare ROI-scoped
  against full-part scoring without leaving the UI.
- **Panel 3 — Test image.** Single upload, normal or defective. A "Inspect" button scores it
  against the fitted subspace.
- **Panel 4 — Output.** Verdict (pass/fail), per-ROI numeric scores, and the anomaly heatmap
  overlaid on the aligned test image. Also show the intermediate stages side by side:
  raw photo -> BiRefNet mask -> ink mask -> aligned canvas -> ROI crops. I need to see where
  it goes wrong, not just the final answer.

**Controls to expose as sliders/dropdowns** (all of these change results and I want to feel
the effect directly):

- backbone (`vits14 / vitb14 / vitl14 / vitg14`)
- input resolution (448 / 512 / 672)
- PCA variance threshold tau (0.90 - 0.99)
- aggregation (`max` / `top-3 mean` / `top-1% TVaR`)
- per-position normalization on/off
- ink masking on/off
- alignment on/off (so the no-alignment failure mode is visible for comparison)

**Behaviour requirements:**

- Cache the fitted subspace in session state. Changing the test image must NOT refit —
  refitting is the slow part and it only needs to happen when the reference set or the
  backbone/resolution/tau changes.
- Show fit time and inference time separately in the UI. These get conflated otherwise and
  the fit cost is not representative of production per-part cost.
- Display the number of valid (in-mask) patches used for the fit, and the number of retained
  PCA components `r`. When something behaves oddly these two numbers usually explain it.

### On "more reference images = better recall?"

Not straightforwardly, and the UI should let me measure it rather than assume it.

More golden images widen the subspace. That is what you want for *lighting and specular
variation*, because highlight positions then count as normal instead of as defects — this
mainly cuts **false calls**. But a wider subspace also reconstructs more things well,
including subtle defects, which can *reduce* sensitivity. So recall and false-call rate move
in opposite directions past some point.

The paper's own numbers show the gain from more shots is modest (MVTec image AUROC 98.0 at
k=1 vs 98.4 at k=4). My expectation for this application is that the useful range is roughly
8-16 golden images, chosen to span lighting and highlight positions rather than just adding
near-duplicates — but this is an empirical question, hence Experiment 5.

Add a warning in the UI if the reference set contains images that are near-duplicates
(very similar aligned appearance), since those add fit time without adding coverage.

---

## Repo layout

```
inspector/
  segment.py     inkmask.py    align.py      features.py
  subspace.py    geometry.py   roi.py        pipeline.py   # the shared preprocessing path
  register.py    inspect.py    annotate.py   config.py
demo.py
experiments/
  exp1_alignment_repeatability.py
  exp2_roi_vs_fullframe.py
  exp3_threshold_sweep.py
  exp4_specular_sensitivity.py
  exp5_reference_count_sweep.py
data/
  golden/<sku>/   test/<sku>/   models/<sku>/
```

---

## Experiments — build these, they gate the whole project

### Experiment 1: Does alignment hold? (build first, nothing works without it)

Take 20 photos of the same good part at varied distance, angle and rotation. Run alignment.
Overlay all aligned masks and report mean IoU against the reference, plus per-image centroid
drift in pixels.

**Pass criterion: mean IoU > 0.95, centroid drift < 5 px on the canonical canvas.**
If this fails, stop and fix alignment. Every downstream number is meaningless otherwise.

### Experiment 2: Does the ROI crop actually help?

One SKU, good parts vs defective parts. Score three ways:
1. full frame, no segmentation (the naive baseline)
2. segmented + aligned, whole part scored
3. segmented + aligned + ROI cropped

Report ROC / AUROC and the score separation between good and defective for each.

**This is the go/no-go for the whole approach.** If ROI cropping doesn't clearly separate the
defect, the method is wrong for this defect class and we need classical CV for it instead.

### Experiment 3: Threshold sweep

Plot recall vs false-call rate for the ROI-cropped configuration. Report the false-call rate at
99% recall. Report per-ROI, not just per-part.

### Experiment 4: Specular sensitivity

Photograph one good part 20x, deliberately moving the light/camera so highlights land in
different places. Score all against the subspace. Report how many false-fail at the
Experiment 3 threshold.

**This is my biggest expected failure mode.** Painted and bare metal both throw moving
specular hotspots that can out-score a real defect. If this experiment fails, the fixes in
priority order are: more golden images covering highlight positions, per-position
normalization, a polarizing filter on the lens, then a diffused light source at the station.

### Experiment 5: How many reference images?

Sweep k = 1, 2, 4, 8, 16, 32 golden images. For each k, refit and plot **both** recall and
false-call rate on the same axes, plus the number of retained PCA components `r` and the fit
time.

I expect false calls to fall with k while recall flattens or slightly declines, giving an
optimum rather than "more is always better". Report where that optimum sits. Also compare
16 near-duplicate images against 16 deliberately varied ones (different lighting, highlight
positions, distances) at the same k, to confirm that coverage matters more than count.

---

## Known risks — surface them rather than papering over them

- **Specular highlights** — see Experiment 4. Expect this to be the dominant false positive.
- **Blue marker ink** — varies per part. Must be masked. Also determine whether parts are
  photographed before or after marking, and keep golden and test consistent.
- **Out-of-plane tilt** — the homography refinement absorbs mild tilt only. Log the estimated
  tilt magnitude per image so we can see when it exceeds what alignment can handle.
- **Logical anomalies** (missing mark, missing tab) are SubspaceAD's documented weak spot —
  a missing component can read as normal background texture. This is exactly why we crop to
  ROIs. Verify per-check in Experiment 2, not in aggregate.
- **Small defects** — if a defect is only a few pixels inside its ROI, the patch residual may
  not clear threshold. Fallback is tiling the ROI, or moving that check to classical CV.
- **Golden set contamination** — one defective golden part puts that defect inside the normal
  subspace permanently. Implement a leave-one-out screen at registration: score each candidate
  golden part against a subspace built from the others, and reject outliers before saving.

---

## What I want from you

Build the codebase and the four experiment scripts. Use the sample images I provide to
validate stages 1-3 end to end (segmentation, ink masking, alignment) and show me the
intermediate visualizations before moving on to scoring — I want to see the masks and the
aligned overlays, not just final numbers.

Ask me before making an architectural decision that isn't specified here. Don't add training,
don't add a detector, don't fine-tune anything.
