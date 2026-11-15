# Study 4, Phase 2–3 escalation: leftover and variation diagnostics (2026-10-30)

Follows `NOTES_phase2_3.md`. Code: `synthetic_generation/realism/leftover.py`, `_study4_leftover.py`,
`_study4_variation.py`. Results: `leftover/` and `variation/` (`calibration.json`, `detector*.json`).

All runs use the same setup as Phase 2–3:
- the same fit-part sources;
- the dev real side (600 SR + all minority, seed 42), patient-grouped;
- protocol v2 with 5 seeds.

The judge part is untouched.

## Gate status: the plan's Phase 4 is blocked
`WORKFLOW_4.md` Phase 4 ports fixes that pass decision D1 (AUC ≤ 0.70 on real templates) to generated templates. **No
fix passed D1.** Every combined variant is at 1.00, so the Phase 5 gate G1 (v2 pool less detectable than v1) cannot
be met. This is recorded as a failed gate. The work below is the Phase 2 escalation ladder (D5), not Phase 4.

## Leftover: what the GP lacks (statistics, no detector)
Measured on the real V2 leftover of the sources (`leftover/calibration.json`):

| class | kurtosis (all / without the edge seconds) | edge-to-middle power | QRS-to-rest power |
|---|---|---|---|
| SR | 27.3 / 7.8 | 4.1 | 4.8 |
| AFIB | 27.7 / 16.6 | 2.8 | 2.5 |
| STACH | 26.5 / 8.5 | 2.7 | 3.5 |
| SBRAD | 29.6 / 6.8 | 4.4 | 7.4 |

- The stationary GP leftover has kurtosis 3.0, and both power ratios are 1.0.
- Its RMS distribution across recordings already matches (log-RMS SD 0.51–0.64 against real 0.53–0.59). Its spectrum
  matches too (Phase 2–3).

## Leftover ladder (detector, full band unless stated)

| variant | leftover | AUC [95 % CI] |
|---|---|---|
| R2 (Phase 2–3) | stationary GP | 0.999 [0.997, 1.001] |
| P_rand (diagnostic) | own real leftover, phase-randomised (same phases on all leads) | **0.993** [0.990, 0.996] |
| R1e | GP × edge envelope | 0.999 [0.997, 1.000] |
| R1b | GP × beat-phase envelope | 0.996 [0.993, 1.000] |
| R1 | GP × edge × beat-phase envelope | 0.998 [0.996, 1.000] |
| R1, between-beats band | | 0.989 [0.986, 0.991] |

How well R1 is calibrated:
- It reproduces the edge power (2.9–4.5) and the beat locking (2.4–5.2 against real 2.5–7.4).
- Its kurtosis rises to 6–11 overall and 4–7 in the middle, against real 27–30 and 7–17.
- The detector does not move.

**Reading:**
- P_rand keeps the real leftover's spectrum, cross-lead spectrum and per-recording scale, yet is detected at 0.993. So
  the giveaway is the leftover's **time-domain structure**.
- Matching its first-order envelopes (edges, beat phase) is not enough. The remaining structure is probably
  waveform-like: the roughly 52–55 % of each beat's deviation that the 3-parameter-per-lead V2 model does not explain,
  plus spiky transients (kurtosis 7–17 away from the edges). Neither is modulated noise.
- Plan D5's next rung is a **learned leftover model** (GPU), or a richer per-beat deviation model that moves that
  structure out of the leftover.

## Beat variation (detector, full band)

| variant | beat deviation (around own template; own V2 leftover and wander) | AUC [95 % CI] |
|---|---|---|
| V0 (Phase 2–3) | V2 deviation removed | 0.771 [0.738, 0.804] |
| **VS** | own real V2 coefficients, permuted across the recording's beats | **0.779** [0.756, 0.801] |
| **VB** (diagnostic) | real V2 coefficient vectors of a same-class donor recording | **0.833** [0.819, 0.847] |
| V2 (Phase 2–3) | Gaussian V2 model | 0.939 [0.922, 0.957] |
| V1 (Phase 2–3) | Gaussian generic PCA model | 0.985 [0.973, 0.997] |

Calibration (`variation/calibration.json`):
- Deviation energy relative to own: VS 1.00 by construction; VB 0.96–1.40.
- R-peak re-detection miss rate: VS 4–9 %, VB 5–11 %.

**Pairing confound (found in this audit).** The V2 leftover is computed *against* each beat's own fitted deviation:
leftover_k = real_k − template − d_k. Replacing d_k with anything else gives real_k + (d_new − d_k). So every variant
in this table carries a pairing mismatch, and even shuffled real coefficients (VS) re-detect beats imperfectly (4–9 %
missed). Consequences:
- **0.77 is the floor of this design, not the AUC of identical beats.** V0 (mismatch −d_k) and VS (mismatch
  d_j − d_k, twice the energy) score the same.
- The **ranking is still fair**, because all rows share the mismatch: own real ≈ none (0.77–0.78) < real from another
  template (0.83) < Gaussian (0.94) < generic Gaussian (0.99).
- VB and V2 have similar deviation energy, yet V2 is far more detectable. So the Gaussian V2 model is wrong in
  **shape** (joint distribution), not only in size.
- Beat order and RR coupling are not detectable at this floor (VS ≈ V0).
- A clean test of variation needs the leftover modelled jointly with d_k (no real leftover kept), which is the full
  generator.

## Development-detector tally (forking paths; the judge detector in Phase 6 guards against this)

| stage | detector evaluations |
|---|---|
| Phase 0 protocol checks (real vs real, before and after patient grouping) | ≈ 4 |
| Phase 0 baselines (5 variants × 5 bands) | 25 |
| Phase 2–3 (`_study4_fixes.py` + V0) | 23 |
| Leftover ladder | 5 |
| Variation diagnostics | 2 |
| **Total** | **≈ 59** |

Variant families tried: generic Gaussian variation, template-relative Gaussian variation, stationary GP leftover,
envelope GP leftover (3 forms), PCA wander.

## Options for the next decision (not started)
1. **Learned leftover model (D5 escalation, GPU).** A small conditional 1-D model of R-aligned leftover segments.
   Memorisation check required (copy correlation > 0.98).
2. **A richer per-beat deviation model**, so that less structure is left in the leftover. For example, a per-class PCA
   basis of rank K around the template (V1's basis gave leftover shares of 8–13 % against V2's 17–31 %), with
   coefficients from a non-Gaussian model (mixture or flow), checked against VB = 0.83.
3. **Stop study 4 here and write it up.** The detectability of the noise assembly is established and explained (time
   structure, not spectrum), along with the negative results.

## Control: the 1 µV quantisation grid (post hoc, prompted by code review, 2026-10-30)
- **The question.** Real PTB-XL samples lie exactly on a 1 µV grid (100 % of samples, every lead), but synthetic
  sums are continuous: about 90 % of samples are off-grid for gen and C, and 65–80 % for the V2-type rebuilds. Could
  the detector be sensing the grid?
- **The control.** Each synthetic ECG was rounded to the grid, using the train normalisation stats to go from z to
  mV, round to 1 µV and back to z. The detector was then rerun with the same protocol (`_study4_grid.py`,
  `grid/`).

| set | unrounded | on the grid |
|---|---|---|
| VS (rebuild floor) | 0.779 | 0.789 [0.757, 0.820] |
| R2 (GP leftover) | 0.999 | 0.999 [0.998, 1.000] |
| gen (current generator) | 1.000 | 1.000 [1.000, 1.000] |

- **Reading.** The grid explains none of the detectability, so every AUC in Phases 0–3 reflects signal content, not
  the storage format.
- **Pool leak in `_study4_fixes.py`, noted alongside.** `oracle2` drew real templates with replacement and was not
  grouped by template (`g_syn` = None), so the same template could fall on both sides of the detector split. Its
  1.000 may be slightly inflated. The conclusion (assembly at 1.00) does not depend on it: gen2, whose templates
  are not repeated, gives the same 1.000. The new learned-residual pools are grouped by source patient or template.

**Detector tally:** 3 more evaluations, about 62 in total.
