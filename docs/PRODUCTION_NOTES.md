# Production deployment notes

**Status: not implemented. This is a record of constraints, not a to-do list.**

This repo is an experimental codebase for validating the method. Everything in
it assumes **one part per photo, photographed close up** — which is what the
sample images are. Production is different in two ways that matter, and this
document records them so they are settled before they become expensive.

Nothing here changes the demo or the experiment scripts.

---

## 1. Camera resolution — decide this before buying hardware

This is the constraint that cannot be fixed in software later, so it comes
first.

The real station photographs a **600 × 1400 mm** area. Measured against the
printed ruler on `IS_8983980631.jpg`, the bracket is about **330 mm** long and
the RH mark about **18 mm** across.

A DINOv2 patch is 14 × 14 px. A feature smaller than one patch cannot be seen
by the model at all; a feature spanning about 3 × 3 patches is the least that
produces a stable multi-patch signature rather than one noisy patch (see the
Mark RH investigation in the git history — at 16 patches the verdict rode on
the single worst one and a good part tripped it).

| Camera over 1400 mm | mm/px | Part | RH mark | Patches | Verdict |
|---|---|---|---|---|---|
| 640×480 (our samples) | 2.188 | 151 px | 8 px | 0.6 | **impossible** |
| 1920×1080 | 0.729 | 453 px | 25 px | 1.8 | **marginal** |
| 3840×2160 (4K) | 0.365 | 905 px | 49 px | 3.5 | workable |
| 4032×3024 (12 MP) | 0.347 | 950 px | 52 px | 3.7 | workable |
| 8000×6000 (48 MP) | 0.175 | 1886 px | 103 px | 7.3 | comfortable |

**The close-up photos this repo is validated on work out at 0.367 mm/px.**
4K across the full station area gives 0.365 mm/px — effectively identical. So:

> **4K is the minimum that preserves the accuracy demonstrated here.**
> 1080p does not: the RH mark would span under two patches, which is the
> regime we already showed to be unreliable.

Caveats worth stating:

- 0.365 mm/px assumes the camera actually resolves to its sensor count. Lens
  quality, focus across a 1400 mm field, and JPEG compression all erode this.
  Treat 4K as the floor, not the target; 12 MP buys margin for little cost.
- Depth of field must cover the whole area with parts of varying height.
- The RH mark is a **low-contrast raised stamp**, so lighting matters as much
  as resolution. Raking/angled light makes an embossed mark far more legible
  than flat illumination.
- If a future SKU has a check smaller than ~18 mm, redo this table for it.
  The arithmetic is `mm_per_px = 1400 / sensor_width_px`, and the feature
  needs ≥ 42 px.

**Fallback if 4K is not available:** a second close-up shot for the mark
checks only, with the wide shot used for presence/shape checks. This splits
inspection into two captures and is more operational work, but it removes the
resolution constraint entirely.

---

## 2. Multiple parts per photo — a missing pipeline stage

Staff place **many parts of the same SKU** in the area at once. Estimated
capacity at 600 × 1400 mm for a 180 × 120 mm part: roughly **18–28 parts per
shot** depending on spacing.

### What breaks today

Two places assume a single part, and both fail silently rather than erroring:

1. **`segment.py` keeps only the largest connected component.** With 20 parts
   in frame, 19 are discarded without warning.
2. **`align.py` computes one centroid and one principal axis for the whole
   image.** For a scattered group these are meaningless — the "principal
   axis" of 20 parts is an artefact of how they happen to be arranged.

Silent failure is the dangerous part: the pipeline would return a confident
verdict for one arbitrary part and report nothing about the rest.

### The shape of the fix

A **multi-instance stage** between segmentation and alignment:

```
photo
  -> segment (all components, not just the largest)
  -> filter components by expected part area  ± tolerance
  -> for each surviving component:
         crop with margin -> EXISTING per-part pipeline unchanged
  -> one verdict per part, plus a per-photo summary
```

The important property: **everything downstream stays as it is.** Alignment,
features, subspace, ROIs, geometry and all four verdicts work unmodified once
each part is isolated in its own crop. This is an addition, not a rewrite.

### Details that will need deciding

- **Expected part area** comes from the golden set, which registration already
  measures (`ref_stats["area"]`). Components far from it are debris, a hand, a
  tool, or two parts touching.
- **Touching parts** merge into one component. Watershed or a distance-
  transform split handles the common case; a component whose area is ~2× the
  nominal is the signal to attempt it.
- **Partial parts at the frame edge** already have handling — `touches_border`
  → RETAKE — but per-part rather than per-photo. A part at the edge should not
  invalidate the other 19.
- **Per-photo reporting.** One photo now yields ~20 verdicts. The operator
  needs an overlay showing which physical part failed, not a list of scores.
  Part identity should be positional (row/column or centroid), since there is
  no other way to point at one.
- **Throughput.** ~1.2 s per part today, so ~25 s for 20 parts. Batching the
  DINOv2 forward pass across crops would cut this substantially; the backbone
  is already loaded once and the crops are independent.
- **Golden set.** Registration should keep using single-part close-ups, or the
  multi-instance stage must run over golden photos too. The former is simpler
  and keeps the reference clean.

---

## 3. Canvas sizing — a related weakness worth revisiting

Current behaviour is **half-adaptive**: the canonical canvas long side is
fixed (`canvas_long_side = 1024`, padded to ~1250 by `part_fill_frac`), while
the short side follows the part's own aspect ratio.

Consequence on the two sample SKUs:

| SKU | Shape | Canvas | Result |
|---|---|---|---|
| 8983864641 | chunky bracket | 1250 × 988 | ROIs had ample patches |
| 8983980631 | long thin bar | 1250 × 438 | Mark RH ROI had only ~16 patches |

The thin part gets less than half the pixel area, so a small feature on it
lands on far fewer patches. That is why SKU 8983980631 needed
`input_size: 1120` while the other SKU is fine at the 672 default.

**Better rule:** size the canvas for constant **pixels per mm** across SKUs,
rather than constant long-side pixels. Then a 18 mm feature occupies the same
number of patches whatever the part's shape, and per-SKU `input_size`
overrides mostly stop being necessary.

This needs a real-world dimension per SKU. Two sources exist already: the part
drawing, or the printed ruler visible in the `IS_*.jpg` inspection sheets.
Not worth doing for two SKUs; likely worth doing before 200.

---

## 4. Inspection sheets — read by hand today

**Now:** the demo *displays* the sheet in the ROI tab, beside the aligned
reference. Nothing parses it. The ROI boxes in both `sku.json` files were read
off the sheets by a human and typed in.

Sheets are excluded from the golden set everywhere else, and must stay that
way: a sheet is a photo of a document on a blue board with a ruler and printed
annotations. Segmentation would find the board, alignment would inherit the
error, and PCA would learn blue background as normal.

**Improvable:** an LLM can reliably extract *how many* checks a sheet has and
*what each says* — "tab at top-right", "corner chamfered". That is the tedious
part across 200 SKUs and it is genuinely automatable.

**Not improvable as easily:** placing the box. The red circles are in sheet
coordinates (part on a board, one angle, annotations drawn over the part); ROI
boxes need canonical canvas coordinates. Mapping between them means running
the whole pipeline on an image it was not designed for, and the annotation
overlaps the feature you would align against.

So the realistic split is **sheet → text checks (automated) → box placement
(human, ~30 s each once you know what to look for)**.

---

## 5. Other open items

- **REVIEW threshold is provisional.** `review_floor = 60.0` was set against
  the observed distribution on 15 images, not derived. The proper basis is the
  golden set's own off-ROI scores, but that only works once golden sets are
  8–16 varied images — currently golden parts score 1.5–2.0 off-ROI while
  known-good test parts score 14–30, so the golden baseline is far too
  optimistic. **Retune on real production parts.**
- **Golden sets are too small.** 4–6 images per SKU. Experiment 5 shows scores
  still falling steeply with k and no plateau, and that coverage beats count
  (varied k=3 gave AUROC 1.000, near-duplicate k=3 gave 0.333). Collect 8–16
  spanning lighting and highlight positions.
- **Experiment 4 (specular sensitivity) has never been run** — no data. This
  is the predicted dominant failure mode in a station with uncontrolled
  lighting, and it is currently unmeasured.
- **Thresholds are per-SKU and hand-set** (12.0 and 20.0 on the two sample
  SKUs). At 200 SKUs these must be derived automatically from the golden
  distribution, with a handful of known-NG parts per SKU to verify.
- **Two sample SKUs, 15 test images, 6 NG parts total.** Every accuracy claim
  in the README rests on that. It demonstrates the method separates these
  defects; it is not a production accuracy estimate.
