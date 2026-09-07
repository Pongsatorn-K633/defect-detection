# Using the demo UI

```bash
python demo.py
```

Opens at **http://127.0.0.1:7860** (Gradio's default port; 127.0.0.1 means only
your machine can reach it). Leave the terminal running — closing it stops the
server.

---

## The one thing to understand first

The demo has **two separate steps**, and confusing them is the usual reason it
feels stuck:

| Step | What it does | How long |
|---|---|---|
| **Fit subspace** | learns what a good part looks like, from the reference images | ~2 minutes |
| **Inspect** | scores one test part against that | ~1 second |

**You fit once, then inspect as many test images as you like.** Changing the
test image never re-fits. A re-fit only happens if you change the reference
images, or the backbone / resolution / tau / masking settings.

If you press **Inspect** before **Fit subspace**, you get
*"Fit the subspace first (Panel 1)"*. That is the most common first stumble.

---

## Fastest path to seeing something work

1. Open the **1. Reference images** tab.
2. Under *Load a sample set*, pick SKU **8983864641** and press **load golden**.
   Six thumbnails appear.
3. Press **Fit subspace**. Wait ~2 minutes. The *fit statistics* box fills in.
4. Go to **3. Test image**. Pick the same SKU, press **list test images** to see
   what is available, then upload one from `test_images/8983864641/` — try
   `image_2.jpg` (a defective part).
5. Press **Inspect**.
6. Go to **4. Output**. You get the verdict, a score table, the heatmap, and
   the stage-by-stage strip.

Then upload `image_7.jpeg` (a good part) and press **Inspect** again — note it
returns in about a second, because it does not re-fit.

> **Note:** with no ROI boxes defined you are in *whole part* mode, which
> scores the entire part. That deliberately performs worse than ROI mode
> (AUROC 0.867 vs 1.000). To reproduce the headline results, set up ROIs as
> below.

---

## The five tabs

### 1. Reference images

Upload photos of **good parts only**. This is what the system learns "normal"
from.

- **load golden** pulls a sample SKU's images from `normal_images/`.
- **Fit subspace** runs registration on whatever is currently uploaded.
- **status** shows progress and warnings.
- **fit statistics** is the box worth reading:

```
Fit time            105.6 s
Reference images    6
Canonical canvas    1250 x 988
Patch grid          38 x 48
Valid patches (fit) 5895        <- how much data the model was fitted on
Per image           min 959  max 1012
Retained components r = 180     <- how wide the "normal" subspace is
Golden whole-part   min 2.01  max 2.04
```

**When something behaves oddly, `valid patches` and `r` usually explain it.**
A threshold must sit above the golden max, or good parts fail.

Warnings you may see, and what they mean:

- *near-duplicate pair(s)* — those images add fit time without widening the
  subspace. Vary the lighting instead of taking more of the same shot.
- *leave-one-out screen rejected image(s)* — one reference part looks unlike
  the others. If it is genuinely defective, remove it: a defective golden part
  puts its defect permanently inside "normal" and the system goes blind to it.
- *Only N reference image(s)* — below 8 the subspace is too narrow and ordinary
  lighting variation reads as anomalous.

### 2. ROI selection

Where to look on the part — and the tab now shows the **inspection sheet**
beside the aligned reference, because drawing boxes means translating the
sheet's checks into coordinates.

Press **show inspection sheet** (or load a sample SKU's golden set, which
pulls it up automatically). It reads `normal_images/<sku>/IS_*.jpg`. Everywhere
else in the pipeline these files are deliberately excluded — a sheet is a photo
of a document, and letting one into the golden set would put a blue background
and printed Thai inside the model's idea of a normal part.

The two panels answer different halves of the same question: the sheet says
**what** to check, your boxes say **where**.

Two modes:

- **whole part** — scores everything. Useful as a baseline, worse in practice.
- **ROI boxes** — scores only inside the boxes you define. This is what makes
  the method work.

The table is one box per line: `id, type, x, y, w, h`

```
corner_top_left, appearance, 90, 90, 260, 220
tab_top_right, appearance, 930, 80, 280, 230
```

- `id` — your name for the check
- `type` — `appearance` (texture, marks, scratches) or `geometry` (silhouette
  shape: missing tabs, wrong corners)
- `x, y, w, h` — in **canonical canvas** coordinates, not photo coordinates.
  The canvas size is shown in *fit statistics*.

Press **Preview ROIs on reference** to see the boxes drawn on the aligned
reference image. Do this before inspecting — a box in the wrong place is the
second most common cause of confusing results.

> **Finding coordinates:** the UI takes typed numbers, not mouse-drag. To drag
> boxes visually use `python -m inspector.annotate --sku <sku>`, which opens an
> OpenCV window. Ready-made boxes for both sample SKUs are already in
> `data/models/<sku>/sku.json`; `python -m inspector.annotate --sku <sku> --list`
> prints them for copying.

### 3. Test image

Upload **one** part to inspect — good or defective. **list test images** shows
what is in `test_images/<sku>/` but does not select one; you still upload via
the file box.

### 4. Output

- **Verdict** — PASS / FAIL / RETAKE, plus a per-ROI score table
- **timings and alignment** — inference time, alignment IoU, centroid drift,
  ECC status, ink coverage
- **heatmap** — anomaly score on the aligned part, with ROI boxes drawn.
  Red inside a box means that check failed.
- **stages** — raw → mask → ink → aligned → valid mask → ROI crops

**Always look at the heatmap, not just the verdict.** If a part FAILs but the
hot spot is not on the defect, the verdict is right by luck. That is exactly
how the Mark RH resolution problem was found.

### 5. Metrics

Answers "is this good enough to put on a line?" with numbers rather than
impressions. Needs ground truth in `data/labels/<sku>.csv` — one line per
image, `image,label`, where 1 = NG and 0 = OK. Both sample SKUs already have
one.

Pick a SKU, press **Evaluate**. It scores every labelled test image once, then
**the threshold slider updates everything instantly** — no re-scoring. That is
what makes the trade-off explorable rather than a two-minute wait per setting.

The three numbers that decide usability:

| Metric | Engineering name | Formula | In words | Want |
|---|---|---|---|---|
| Capture rate | Recall / sensitivity / TPR | TP / (TP + FN) | of the bad parts, how many we caught | **HIGHER** |
| False alarm rate | FPR / 1 − specificity | FP / (FP + TN) | of the good parts, how many we wrongly stopped | **LOWER** |
| Review load | Flag rate / positive rate | (TP + FP) / total | of everything, how much lands in the queue | LOWER |

**Capture rate up, false alarm rate down — and they pull against each other.**
Lowering the threshold catches more defects and also stops more good parts, so
every setting is a compromise. The sweep chart shows exactly where yours sits.

Four charts:

- **confusion matrix** — TP/FP/FN/TN at the current threshold. Colour marks
  right versus wrong, not magnitude: a large TN and a large FP are both big
  numbers but only one is good.
- **sweep** — capture and false alarm rates across every threshold, with your
  current one marked and a suggested one that catches every NG.
- **score distribution** — every part's score by true label. **The most useful
  chart here.** Its title names the closest OK and closest NG and the gap
  between them; a healthy AUROC can sit on top of a margin of almost nothing,
  and this is where you see that.
- **ROC** — the standard threshold-independent summary, with AUROC.

RETAKE images are excluded and listed by name. An unusable photo says nothing
about the part, so counting it either way would invent an outcome.

The panel also names any **missed defect** outright, because a missed NG
normally costs far more than a false alarm.

---

## The controls, and what each one actually does

| Control | Default | What it changes |
|---|---|---|
| **segmenter** | birefnet | How the part is separated from the bench. `birefnet` is a downloaded model and more accurate (IoU 0.967 vs 0.935); `classical` needs no download. |
| **backbone** | vitl14 | DINOv2 model size. `vitl14` is the right default. **Avoid `vitg14`** — a 4.2 GB download and ~3× slower for little gain. |
| **input resolution** | 672 | Bigger = more patches per ROI. Raise it when an ROI has too few patches to be stable. |
| **PCA variance tau** | 0.97 | How much normal variation to keep. Higher = wider subspace = fewer false alarms but potentially blind to subtle defects. |
| **aggregation** | max | How patch scores become one ROI score. `max` suits small defects; `top-1% TVaR` averages them away. |
| **patch coverage** | 0.9 | A patch counts only if this fraction is inside the part. Lower lets boundary patches in, which adds false alarms. |
| **per-position normalisation** | on | Standardises each patch position against the golden set. Edges and holes always score high; this removes that baseline. |
| **blue-ink masking** | on | Excludes operator pen marks. Turn off to see them become the top anomaly. |
| **alignment** | on | Pose normalisation. Turn off to see the failure mode the whole design guards against. |
| **verdict threshold** | 6.0 | Score above which an ROI fails. Sample SKUs use 12.0 and 20.0. |

**Changing any of segmenter / backbone / resolution / tau / coverage /
normalisation / ink / alignment forces a re-fit.** Only the threshold is free
to change — it is applied at inspection.

---

## Four experiments worth running in the UI

Each one takes a couple of minutes and shows something the numbers alone don't.

**1. Why alignment matters.** Fit normally, inspect a good part, note the score.
Now untick **alignment**, re-fit, inspect the same part. Scores explode: without
pose normalisation, patches are compared against the wrong locations.

**2. Why ROI cropping matters.** Inspect a defective part in **whole part**
mode, then again in **ROI boxes** mode with proper boxes. The whole-part
heatmap has hot spots scattered over every edge; the ROI score isolates the
real defect.

**3. Why aggregation matters.** Switch from `max` to `top-1% TVaR` and re-inspect
a defective part. The score drops sharply — averaging over the top 1% of patches
dilutes a small defect back into the noise.

**4. Why ink masking matters.** Untick **blue-ink masking**, re-fit, inspect.
Pen strokes become the strongest anomaly on the part.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| *"Fit the subspace first (Panel 1)"* | Press **Fit subspace** before **Inspect**. |
| Fit takes ~2 minutes every time | Expected on the first fit. It should be instant on re-press if nothing changed; if it re-fits anyway, you changed a setting that invalidates the subspace. |
| Everything FAILs | Threshold too low, or the golden set does not cover the test lighting. Check *Golden whole-part max* in fit statistics — the threshold must sit above it. |
| Good parts flagged | Usually too few or too uniform reference images. Add varied ones. |
| One patch spikes, scores unstable | The ROI holds too few patches. Raise **input resolution**. |
| RETAKE on a good part | The part touches the frame edge. Re-shoot with the whole part visible. |
| Page won't load | Check the terminal for the actual URL — if 7860 was busy Gradio uses 7861. |
| Slow, or `cuda False` | CPU-only PyTorch. See Setup in the README. On a laptop, plug in — GPUs clock down on battery. |
| First run stalls at 0% | Downloading BiRefNet (~900 MB) or DINOv2 (~1.1 GB). One time only. |

---

## What the demo is not

It is an experimentation tool, not the operator interface. It handles **one
part per photo**, has no database, no user accounts, and no multi-part
handling. Production constraints are recorded in
[PRODUCTION_NOTES.md](PRODUCTION_NOTES.md).
