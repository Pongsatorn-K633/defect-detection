# Performance — single source of truth

Every per-stage timing in this repo comes from this document. If a number
appears elsewhere and disagrees with the table below, this table is right.

**Measured on:** RTX 4070 Laptop GPU (8.6 GB, 36 SM), OpenCV 4.11 with 20
threads, torch 2.6.0+cu124, Windows 11.
**Method:** each sub-stage timed inside one process with `torch.cuda.synchronize()`
around it, 2 warmup passes discarded, median across every available image
(`test_images/`, `ng_images/`, `ok_images/`, `normal_images/`) × 3 repeats.
**Reproduce:** `python experiments/bench_stages.py --repeats 3`
(defaults to each SKU's fitted `input_size`; `--sizes 896,1120` to sweep)

---

## The table

Both SKUs at their **own fitted `input_size`** — the configuration that
actually ships, read from `data/models/<sku>/model.npz`.

| # | Stage | Runs on | Parallel? | 8983864641<br>`input_size=896` | 8983980631<br>`input_size=1120` | Code |
|---|---|---|---|---|---|---|
| 1 | load | **CPU** | single-thread | 3 ms | 3 ms | `pipeline.load_image` |
| 2 | segment (BiRefNet) | **GPU** | — | 253 ms | 249 ms | `segment.segment_part` |
| 3 | ink mask | **CPU** | multi-thread | 4 ms | 3 ms | `inkmask.blue_ink_mask` |
| 4 | align — similarity | **CPU** | multi-thread | 34 ms | 29 ms | `align.similarity_to_canonical` |
| 5 | **align — ECC refine** | **CPU** | **effectively serial** | **479 ms** | **153 ms** | `align.refine_homography` |
| 6 | align — final warps | **CPU** | multi-thread | 10 ms | 7 ms | `align.warp_image` ×2 |
| 7 | features (DINOv2-L) | **GPU** | — | 486 ms | 299 ms | `features.extract_patch_features` |
| 8 | score (PCA residual) | **CPU** | multi-thread | 65 ms | 47 ms | `subspace.score_maps` |
| | **TOTAL** | | | **1334 ms** | **790 ms** | |

Sample size: n=30 (10 images × 3) and n=24 (8 images × 3). ECC converged on
every run (`ecc_ok` 30/30 and 24/24).

### Where the time goes

| | 8983864641 | 8983980631 |
|---|---|---|
| GPU stages (2, 7) | 739 ms — **55%** | 548 ms — **69%** |
| CPU stages (1, 3, 4, 5, 6, 8) | 595 ms — **45%** | 242 ms — **31%** |
| — of which ECC alone | 479 ms — **36% of total** | 153 ms — **19% of total** |

Peak VRAM: **2.93 GB** in every configuration measured, including
`input_size=1120`.

---

## Why stage 5 is called out

`refine_homography` is `cv2.findTransformECC` with `MOTION_HOMOGRAPHY`, an
iterative solve capped at `ecc_iterations=200`. It is the largest single CPU
cost and the largest single stage on the chunky-bracket SKU.

Measured directly against OpenCV's thread count (SKU 8983864641, one image,
median of 5):

| `cv2.setNumThreads` | ECC time |
|---|---|
| 20 | 755 ms |
| 8 | 967 ms |
| 4 | 729 ms |
| 1 | 1024 ms |

Going from 1 thread to 20 buys about 25%, not 20×. **ECC does not parallelise
meaningfully** — it is latency-bound on a serial iteration loop, so it scales
with single-thread CPU performance. That matters for any ARM SoC, where
single-thread performance is the weakest dimension.

### The downscale lever

ECC cost is quadratic in canvas area, and the refinement it produces is a
homography that can simply be scaled back up:

| ECC input | Resolution | Time | Speedup |
|---|---|---|---|
| full canvas | 1250×988 | 804 ms | 1.0× |
| half | 625×494 | 128 ms | **6.3×** |
| 0.35 | 437×345 | 50 ms | **16×** |

This is the single highest-leverage optimization available, worth roughly
400 ms/part on SKU 8983864641.

**Validated against Experiment 1** (pass: golden mean IoU > 0.95, max drift
< 5 px). ECC solved on a downscaled canvas, homography conjugated back as
`S⁻¹ H S`, then the real Exp-1 criterion re-run:

| scale | ECC ms | 8983864641<br>mean IoU / max drift | 8983980631<br>mean IoU / max drift |
|---|---|---|---|
| 1.00 (today) | 1253 / 537 | 0.9707 / 2.90 px | 0.9864 / 0.43 px |
| 0.60 | 547 / 198 | 0.9707 / 2.86 px | 0.9863 / 0.53 px |
| **0.50** | **273 / 98** | **0.9706 / 2.70 px** | **0.9865 / 0.45 px** |
| 0.35 | 160 / 88 | 0.9705 / 2.36 px | 0.9866 / 0.51 px |
| 0.25 | 115 / 44 | 0.9698 / 1.58 px | 0.9862 / 0.93 px |

**Every scale passes on both SKUs**, and accuracy is flat to four decimal
places down to 0.35. The large `corner_shift` differences between scales
(65.4 px full vs 21.9 px at 0.35) reflect the *reported diagnostic*, not
alignment quality — measured IoU and drift are unchanged.

**Recommended: `scale = 0.5`** — a 4.6× ECC speedup with no measurable
accuracy cost, and two further steps of margin before anything degrades.

> **Full-res ECC is not the accurate option — it is the slow one.**
> This is the most common misreading of the table above, so it is worth
> stating directly: accuracy is **identical to three decimal places** across
> every scale, and max drift *improves* as scale drops (2.90 px → 2.36 px on
> 8983864641). At 0.35 the second SKU scores fractionally **better** than
> full-res. You are paying 479 ms for accuracy you already have at ~104 ms.
>
> **Why:** with `ecc_on="mask"` (the default), ECC aligns silhouettes through
> a signed distance transform, not image texture. A part outline is a smooth,
> large-scale shape — locating it does not need 1250×988. Downscaling discards
> pixels that were never carrying alignment information in the first place.
>
> The practical consequence for hardware: **there is no accuracy argument for
> buying a faster CPU.** Downscale ECC, then choose on GPU.

With ECC at 0.5, per-part on SKU 8983864641 falls **1334 ms → ~960 ms**, and
the CPU/GPU balance inverts to **77% GPU / 23% CPU**. This is why GPU, not
CPU, should drive hardware selection — but only *after* this fix lands. The
ECC timings above are from the Exp-1 harness (cold, single-image) and run
higher than the SSOT table's steady-state medians; the *ratios* are what
transfer.

---

## Scaling notes for hardware selection

The split above is what makes the hardware question non-obvious. **Which
component to buy depends on whether the ECC fix has landed:**

| | GPU share | CPU share | Buy for |
|---|---|---|---|
| Today (ECC full-res) | 55% | **45%** | CPU single-thread matters as much as GPU |
| With ECC @ 0.5 | **77%** | 23% | GPU dominates; CPU spec matters much less |

A machine chosen on GPU alone will disappoint on *today's* code. A machine
chosen on CPU will be overspent once the fix lands. **Do the fix first, then
buy for GPU.**

- Stages 2 and 7 scale with **GPU** fp16 throughput.
- Stage 5 scales with **single-thread CPU** performance.
- Stages 3, 4, 6, 8 scale with **multi-thread CPU**, but are small.

### ECC is the smaller half of the Orin gap

Stage 5 being CPU-bound is a real reason Orin NX scores poorly — its ARM
cores are weak at single-thread work. But it is **not the main reason**, and
attributing the ranking to ECC leads to the wrong purchase. Split after the
ECC fix, SKU 8983864641 @ 896:

| | Orin NX 16 GB | RTX 4060 Ti | gap |
|---|---|---|---|
| CPU (ECC @ 0.5 + stages 1, 3, 4, 6, 8) | ~620 ms | ~220 ms | **~0.4 s** |
| GPU (segment + DINOv2) | ~2,960 ms | ~700 ms | **~2.3 s** |

The GPU gap is roughly **6× larger** than the CPU gap. Orin ranks last mainly
because its integrated Ampere GPU is ~4× slower on ViT-L inference — ECC
contributes, but fixing ECC alone does not close the distance.

### Candidate machines

1 line, 1 camera, 500+ SKUs. Per-part estimates for SKU 8983864641 @ 896,
**assuming ECC @ 0.5**, scaled from the measured RTX 4070 Laptop stages by
relative GPU throughput and single-thread CPU.

| # | Machine | GPU | VRAM | 1 part | 20-part tray | Verdict |
|---|---|---|---|---|---|---|
| **1** | **RTX 4060 Ti 16 GB + i5-14400** | Ada, 4352 CUDA | **16 GB** | **0.92 s** | 18.4 s | **Best balance** |
| 2 | RTX 4070 12 GB + i5-14500 | Ada, 5888 CUDA | 12 GB | 0.75 s | 15.0 s | Fastest sane option |
| 3 | RTX 4060 8 GB mini + i7-13620H | Ada, 3072 CUDA | 8 GB | 1.06 s | 21.2 s | Smallest footprint |
| 4 | RTX A2000 12 GB SFF | Ampere pro | 12 GB | 1.49 s | 29.7 s | If certified hardware is required |
| 5 | Jetson Orin NX 16 GB (J4012) | Ampere, 1024 CUDA | 16 GB unified | 3.57 s | 71.4 s | Only for fanless / 10–25 W |

Tray figures assume the multi-instance stage exists and parts run serially;
batching DINOv2 and amortising segmentation per photo improves all rows
roughly 2× (#1 → ~8.8 s).

**Three facts that should shape the purchase:**

- **500+ SKUs is not a hardware factor.** A SKU model is 1.5–2.2 MB measured
  (`data/models/<sku>/model.npz`), so 500 SKUs is ~1 GB on disk and ~700 KB
  resident for the active SKU. There is no per-SKU network to load.
- **VRAM is not the constraint.** Peak 2.93 GB measured. Buy VRAM for future
  batching headroom, not for today's workload.
- **Orin NX ranks last on throughput specifically because of stage 5.** Its
  weakest dimension is single-thread CPU, which is exactly what ECC needs.
  Its figures above already assume the ECC fix; without it, expect ~4 s/part.
  Budget for the ARM port (JetPack, ARM wheels, likely TensorRT).

### Why #1, and when to pick another

**RTX 4060 Ti 16 GB + i5-14400, 32 GB RAM, 1 TB NVMe.**

- **16 GB VRAM at the price of 8.** The cheapest modern card with this much
  memory. Not needed today (2.93 GB peak), but it is what lets you batch 20
  crops in one forward pass later, and what absorbs a larger backbone.
- **Comfortably under 1 s/part** after the ECC fix, without paying for a 4070.
- **Standard mini tower**, x86. BiRefNet, `torch.hub` DINOv2 and OpenCV all
  run with zero porting.
- **The i5 is deliberate, not a compromise.** CPU work is ~220 ms/part after
  the fix, and ECC's serial portion cares about single-thread clock, which
  the 14400 has in abundance. A pricier CPU buys nothing here.

**#2** if you want maximum headroom and the ~15% price delta is irrelevant —
faster, at the cost of 4 GB VRAM.
**#3** if the enclosure must be genuinely small; 8 GB makes aggressive
batching tighter, though fine for today's 2.93 GB.
**#5 only** for fanless/industrial mounting or a 10–25 W budget.

### Buy alongside

- **32 GB system RAM.** Not for the model — for OpenCV buffers and holding a
  4K frame plus ~20 crops. 16 GB works; 32 GB is cheap insurance.
- **1 TB NVMe.** The 500 SKU models are ~1 GB, but golden sets, logs and
  captured frames accumulate.
- **4K camera minimum.** `PRODUCTION_NOTES.md` establishes this from the
  18 mm RH mark against a 1400 mm field. **Independent of the PC and not
  fixable in software later.** Confirm the capture path sustains your frame
  rate.
- **UPS.** A mid-inspection power loss is an operational problem, not a
  compute one.

---

## Other notes

`input_size` affects **only stage 7**. Going 896 → 1120 on SKU 8983864641
costs **+451 ms** of DINOv2 time; every other stage moves by under 14 ms,
which is measurement noise. Segmentation is fixed cost at
`birefnet_input=1024` regardless.

---

## Known gaps

- **One camera, one part per photo.** These numbers are per-part on
  single-part close-ups. The multi-instance stage described in
  `PRODUCTION_NOTES.md` does not exist yet; per-photo throughput for a tray of
  ~20 parts is an extrapolation, not a measurement.
- **Measured on x86 + discrete GPU only.** No ARM/Jetson numbers here. Any
  Orin figure quoted elsewhere is a scaled estimate, and the ECC row is the
  one most likely to be underestimated.
- **Only the RTX 4070 Laptop column is measured.** Every candidate machine
  figure is scaled by relative GPU throughput and single-thread CPU, not
  benchmarked on that silicon. **Treat the ranking as reliable and the
  absolute numbers as ±30%.** Borrowing one unit and running
  `experiments/bench_stages.py` would settle it definitively.
- **Small image corpus.** 10 and 8 images per SKU. ECC cost depends on initial
  pose error, so a wider corpus could widen the spread.
- **Two SKUs measured, not 500.** Cost varies with part shape and
  `input_size`; a thin part at 1120 behaves differently from a chunky one at
  896. Expect a spread once the real catalogue is registered.
- **The ECC downscale is validated for alignment, not implemented.** It passes
  Experiment 1 at every scale tested, but has not been run through the full
  defect-detection metrics (`inspector/metrics.py`), and no code change has
  been made. Every "@ 0.5" figure in this document is therefore a projection.
