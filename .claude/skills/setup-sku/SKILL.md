---
name: setup-sku
description: "Set up and prove every inspection-sheet check for one part (SKU) end to end: read the sheet, place the boxes, choose each check type, write sku.json, and verify on the labelled and test photos. Use when the user types /setup-sku <SKU> or asks Claude to find the boxes for a part."
argument-hint: "<SKU>"
risk: safe
source: project
source_type: project
date_added: "2026-10-01"
tags: [defect-detection, roi, inspection-sheet, sku-setup]
tools: [claude-code]
---

# Set up the checks for part $ARGUMENTS

The user cannot place the boxes by hand. Do the whole job, then report.
If no part number was given, ask for it and stop.

Load the `roi-placement` skill too: it holds the rules for framing a box.

## Inputs

- Inspection sheet: `normal_images/<SKU>/IS_<SKU>.*`. Read it (it's in Thai)
  and list every item it asks to check.
- Photos: golden `normal_images/<SKU>/`, good `ok_images/<SKU>/`, bad
  `ng_images/<SKU>/`.
- Model: `data/models/<SKU>/`. If `model.npz` is missing, stop and tell the
  user to press **Fit subspace** in step 1 of the app.
- **Never** use `test_images/<SKU>/` to choose boxes or limits. Use it only
  for the final blind check.

## Steps

1. **Sanity check.** Is the aligned reference (`reference_aligned.png`) the
   same way round as the sheet? Mirrored is fine if every photo shows the back
   side. List photos that are RETAKE, or aligned oddly (camera off-centre
   shifts raised features against the outline).
2. **Locate** each sheet item on the reference. Boxes are `x, y, w, h` in its
   pixels.
3. **Build and prove one check per item** with `inspector/boxsearch.py`: run
   `score_photos` once, then `search()` and `check_boxes()`. Pick the type by
   the defect:
   - **A hole you can see the table through:** geometry hole check
     (`min_hole_px`, `hole_px`, plus shape limits from
     `geometry.hole_shape_limits`, so an oval must stay oval and a round hole
     round). One box per hole, never two holes in one box. Leave room for the
     camera moving the hole.
   - **A feature whose absence opens a spot** (a tab over a window): geometry
     `min_area_frac` ("must be metal"). **A feature that must not be
     there:** `max_area_frac` ("must be empty").
   - **Surface defects** (scratch, hole not punched over solid metal):
     appearance, with its own `threshold`. The box goes on the feature only,
     away from shiny edges and the outline.
   - Prefer suggestions that are right on every photo they had not seen.
     Reject any suggestion that no longer covers the feature on the sheet.
     Drop the appearance half of a check if it is what fails good photos.
4. **Write** `data/models/<SKU>/sku.json`. Back up the old one to the
   scratchpad first. Keep `canvas_size`, `ref_area_px` and `input_size`. Name
   each check after its sheet item, with a plain-words note.
5. **Verify with the saved file.** Golden, good and bad photos must all be
   judged right (`check_boxes`). Then run `inspect_image` on every photo in
   `test_images/<SKU>/` and report each verdict with its reason.
6. **Report:**
   - per check: box, type, and worst good vs limit vs weakest bad
   - which sheet items are proven, and which are not (no bad photo shows them)
   - the remaining risks
   - then tell the user: pick the part and press **↻ Reload checks from
     sku.json** (step 2), or ↻ beside the part picker. Restart the app only
     if code changed. Don't press Save sku.json, Fit or Measure for it
     before reloading. Skip step 3 (Tuning) when every check is geometry.

Don't commit. Don't change code unless a needed check type doesn't exist; if
so, explain first.
