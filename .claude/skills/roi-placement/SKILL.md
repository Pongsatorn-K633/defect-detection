---
name: roi-placement
description: "Place inspection-sheet checks (ROIs) on a SKU's aligned reference: which type to use, how tightly to frame the box, and how to tell a working check from one that only looks like it works. Use when adding or fixing checks in sku.json, when a good part fails, or when a defect is missed."
risk: safe
source: project
source_type: project
date_added: "2026-09-09"
tags: [defect-detection, roi, inspection-sheet, geometry, appearance]
tools: [claude-code]
---

# Placing inspection checks

An ROI is one line off the paper inspection sheet, drawn as a box on the
SKU's canonical canvas. It has a type, a box, and a limit. Getting the type
and the framing right matters more than any threshold: a badly framed box
fails good parts or misses real defects, and both look like a threshold
problem until you measure.

## 1. Which type

Ask: **does the defect change the part's outline?**

- **No — the silhouette is identical.** Appearance only. An embossed stamp,
  a printed character, a scratch, a stain, wrong finish. The metal is all
  still there; only the surface differs.
- **Yes — metal is missing, bent, or shaped differently.** Both types. A
  missing tab, a chamfered vs square corner, a bent flange.

Both types on one region is the normal case for a shape feature, because
they fail on opposite things:

- appearance sees a feature that is **present but wrong** — scratched,
  stained, badly finished
- geometry sees a feature that is **missing or misshapen**

Appearance is blind to a missing tab: the absent metal reads as ordinary
background texture, which the model has seen in every golden image. This is
the "logical anomaly" case `inspector/geometry.py` documents. Geometry is
blind to a scratch, because a scratch does not move the outline.

Every ROI is scored on every inspection and any failure fails the part
(`inspect.py`), so a pair widens coverage rather than duplicating work. A
geometry ROI costs ~0.03 ms — compute is never the reason to drop one.

## 2. How tight to frame it

**Appearance: tight. Geometry: loose.** They measure different things and
want opposite framing.

Appearance scores the **surface inside** the box, so background is dead
weight — it dilutes the anomaly signal and drags in variation that has
nothing to do with the check.

Geometry measures the **outline crossing** the box, so it needs room around
the feature. Framed too close, the box's own edges start being measured
instead of the part.

Measured on SKU 8983864641, changing only the box:

| change | appearance margin | geometry |
|---|---|---|
| tab box 40% → 56% part | 1.15× → 1.51× | 0.82× → 1.16× |
| corner box height 220 → 180 | 1.51× → **2.33×** | good part scored **1.19** — false fail |

The corner case is the one to remember: the tighter box was clearly better
for appearance and unusable for geometry. Because the two are separate rows
in `sku.json`, they do not have to share a box — give each the framing it
wants.

## 3. Check the box before trusting it

Three failures that look like something else:

**Mostly background.** Below ~45% part, every measure taken inside is
diluted. `demo.py` warns on this when a box is drawn or edited
(`box_fill`).

**Geometry on flat interior.** If the silhouette has no edge inside the box
(area fraction above ~0.95 or below ~0.05), a geometry check there cannot
catch anything — but it still fires on segmentation noise, so it only ever
costs you good parts. `geometry_useless_here` warns; registration warns
again when it fits the nominal.

**Too few patches.** With `max` aggregation an appearance ROI below ~25
valid patches rides on one or two of them, and a good part trips it on an
unlucky draw. Fix with a per-SKU `input_size`, measured in the Tuning with
NG tab — never guessed.

## 4. Verify with real parts, not by eye

A box is only good if it separates. Score the known-bad photos in
`ng_images/<sku>/` against the golden set and compare:

- **appearance** — `best_NG / worst_OK`. Above ~1.5 is workable; 1.15× is
  one bad photo away from failing.
- **geometry** — the score is tolerance usage, so 1.0 *is* the limit. Every
  good part must land below 1.0 and every bad part above it. A geometry
  check where bad parts score under 1.0 is not catching the defect, whatever
  its ratio looks like.

If a measure is unstable across *identical good parts*, the framing is not
the problem and no box will fix it. On SKU 8983864641 `corner_angle`
measured 31.9° to 88.1° across six golden parts of the same shape — that
measure cannot be trusted in that region, and a check resting on it passes
or fails by luck. Drop that tolerance for the ROI and keep the stable ones
(`area_pct`, `fill_ratio_pct`) rather than tuning the box around noise.

## 5. After changing any box

Geometry nominals are measured at registration, so a moved box needs a
re-fit before its geometry check means anything:

```
python -m inspector.register --sku <SKU> --segmenter birefnet
python -m inspector.inspect  --sku <SKU> --dir test_images/<SKU>
```

Re-fitting also invalidates the threshold — it is a measurement against one
fitted model, not a preference. `sku.json` records the patch grid it was
measured on and the app flags it stale; re-measure in Tuning with NG.

## What not to do

- Do not tune a threshold to paper over a badly framed box. Fix the box.
- Do not read a defect's absence as "the check works" — confirm on a known
  bad part that the check actually fires.
- Do not add a geometry ROI to a surface-only check. It measures nothing
  and can only misfire.
- Do not edit `input_size` in `sku.json` by hand. It must match the model,
  and it is written back by registration.
