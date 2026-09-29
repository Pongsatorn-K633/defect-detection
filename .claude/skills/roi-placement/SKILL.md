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

**One box, drawn around the feature, used by both checks.** The demo editor
works this way: the operator draws one box per inspection-sheet item, it is
stored as `<name>` (appearance) and `<name>_geom` (geometry) over the same
box, and the geometry half is added only when the outline clearly runs
through the box (15–85% part, `outline_through`). Keep it that way — the
operator should never have to know the two checks apart.

Frame the box around the **whole feature, including where it ends**. On SKU
8983864641 the wing check stopped at the corner where a good wing ends, so a
wing that ran on too long sat entirely outside it and passed. Moving the box
to 900,90,200,140 covered the wing's end and caught it (appearance 21.6 vs
worst good part 15.1).

Do not pad the geometry box to "give it room". Measured on the same SKU,
padding each side of the drawn box only hurt:

| geometry box | top-left corner | top-right wing |
|---|---|---|
| as drawn | every good part < 0.70 | missing / raised / too-long wing: 1.19 / 1.13 / 1.86 — all caught |
| +10% each side | fine | 1.10 / 1.07 / 3.19 |
| +20% each side | a good part scored **4.61** — false fail | 0.52 / 0.87 — **missing and raised wing pass** |

Padding pulls more of the part into the box, so the feature is a smaller share
of what is measured. Too tight is also bad: a corner box cut from height 220
to 180 made a good part score 1.19. Draw around the feature itself.

## 3. Check the box before trusting it

Three failures that look like something else:

**Mostly background.** Below ~45% part, every measure taken inside is
diluted. `demo.py` warns on this when a box is drawn or edited
(`box_fill`).

**Geometry on flat interior.** If the silhouette has no edge inside the box
(area fraction above ~0.95 or below ~0.05), a geometry check there cannot
catch anything — but it still fires on segmentation noise, so it only ever
costs you good parts. The editor leaves it out automatically
(`outline_through`); registration still warns when a hand-written one is
fitted.

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
