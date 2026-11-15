# Study 4, Phases 2–3: noise and variation fixes on real templates (2026-10-30)

Plan: `report/WORKFLOW_4.md`. Code: `synthetic_generation/realism/models.py`, `_study4_fixes.py`, and the post hoc
control `_study4_fixes_v0.py`. Results: `fixes/calibration.json`, `fixes/detector.json`, `fixes/detector_*.json`.

Data use:
- Models are fitted on the `fit` part (400 recordings per class).
- The detector's real side is the `dev` part (600 SR + all minority, seed 42, patient-grouped, 5 seeds; protocol v2).
- The `judge` part is untouched.
- Frozen code was checked at the start and end of every run.

## What each variant replaces
Each semi-synthetic variant keeps a fit-part recording's own template at its own R-peaks and swaps out one or more
parts:

| variant | beat-to-beat variation | leftover | wander |
|---|---|---|---|
| V0 (post hoc) | no amplitude/shift/width variation; the rest of the real variation (≈ 52–55 %) stays in the own leftover | own | own |
| V1 | modelled, generic PCA basis | own | own |
| V2 | modelled, template-relative | own | own |
| R2 | own | Gaussian process | own |
| V2R2 | modelled V2 | Gaussian process | own |
| W1 | own | own | PCA wander |
| full | modelled V2 | Gaussian process | PCA wander |

Fully generated variants:
- **oracle2:** real templates with a sampled rhythm and the v2 assembly (V2 + GP leftover + PCA wander).
- **gen2:** the same, with generated templates (study 1 pool).

## Detector results (AUC, mean with 95 % CI over 5 seeds; 0.5 = indistinguishable)

| variant | full | < 0.5 Hz | beats only | between beats |
|---|---|---|---|---|
| ref_C (old GP residual, baseline C) | 0.999 [0.997, 1.001] | | | |
| ref_D (old GP wander, baseline D) | 0.895 [0.867, 0.924] | 0.976 [0.964, 0.987] | | |
| **V0, template-relative variation removed (post hoc)** | **0.771 [0.738, 0.804]** | | | |
| V1, generic variation | 0.985 [0.973, 0.997] | | | |
| V2, template-relative variation | 0.939 [0.922, 0.957] | | 0.931 | |
| R2, GP leftover | 0.999 [0.997, 1.001] | | | |
| V2R2 | 0.999 | | 0.999 | 0.981 |
| **W1, PCA wander** | 0.925 [0.912, 0.938] | **0.918 [0.900, 0.937]** | | |
| full | 0.999 | 0.972 | 0.999 | 0.982 |
| oracle2 | 1.000 | 0.971 | 0.999 | 0.980 |
| gen2 | 1.000 | 0.966 | | |

The references reproduce Phase 0: C 1.000 and D 0.888 / 0.977. The pipeline is consistent.

## Findings

**1. The leftover Gaussian process is the giveaway, even after the per-beat fit.**
- R2 swaps only the leftover and scores 0.999, the same as the old residual (ref_C).
- Phase 1 expected a stationary process to be enough once the QRS-locked part was modelled per beat. **That
  expectation is refuted.**
- The spectrum is not the cause. On 100 recordings per class, real and GP leftovers have similar RMS and similar
  power shares in the bands 0–0.5, 0.5–2, 2–5, 5–15, 15–25 and 25–50 Hz (within about 0.03–0.1). Low-frequency
  content is also similar.
- So the detector sees structure a stationary Gaussian process cannot carry: non-Gaussian amplitude, time-varying
  amplitude, or residual phase-locking to the beats.

**2. Modelled beat-to-beat variation makes the ECG *more* detectable, not less.**
- With the template-relative part of the variation removed (V0), the AUC is 0.771. **Correction (2026-10-30):** V0 does
  *not* have identical beats. V2's 24 parameters explain only about 45–48 % of the deviation energy; the remainder stays
  in V0's own (real) leftover, so V0 still carries real beat-to-beat variation.
- Template-relative variation (V2) gives 0.939; generic variation (V1) gives 0.985.
- V2 beats V1 clearly (CIs do not overlap), but both are worse than V0, which keeps only the real remainder of the variation.
- Calibration explains part of this. Measured coefficient variance (synthetic over real):

  | class | V2 | V1 |
  |---|---|---|
  | SR | 1.3× | 1.7× |
  | AFIB | 1.9× | 1.7× |
  | STACH | 1.5× | 1.5× |
  | SBRAD | 1.0× | 1.4× |

  AFIB and STACH variation is too large.
- Independent Gaussian draws per lead and feature are also not what real beats do.
- **What this says about "make many different beats" (narrower than first written):** adding Gaussian
  template-relative variation on top of a leftover that already holds the real remainder raises the AUC from 0.77 to
  0.94; a truly identical-beat control was not run. Naive random variation is easily detectable (0.94–0.99). Variation has to be modelled from real joint structure (for
  example resampled real coefficient vectors, or a learned conditional model), not as independent Gaussians.

**3. PCA wander fixes wander in its own band, but not in the full ECG.**
- In the < 0.5 Hz band, W1 scores 0.918 against the old GP's 0.976; the CIs do not overlap.
- Calibration supports this. The edge-to-middle increment ratio is real 19–26, W1 25–28, old GP 2.3–3.2. Kurtosis and
  RMS quantiles match.
- Wander memorisation: nearest correlation to any training wander is at most 0.755; no copies (> 0.98).
- In the full band W1 is 0.925, against 0.895 for the old GP. The CIs touch, so there is no improvement there and
  possibly a small loss.
- Wander PCA rank 60 explains 0.977 of real wander variance.

**4. Combined, nothing moves.**
- full, oracle2 and gen2 stay at 1.000 in the full band, because the GP leftover alone already gives 0.999.
- oracle2 ≈ gen2 again. With this assembly, generated templates add no detectability of their own.
- Decision D1 (AUC ≤ 0.70) is not met by any composite.

## Audit notes
- **V1 first build failed calibration.** The R-peak detector missed 20–28 % of beats (real 0 %). The template-relative
  V2 was added in response. That build is kept in `fixes_v1_attempt/` (JSON only tracked).
- **V2 first build blew up for SBRAD.** Near-flat leads gave unregularised coefficients with variance about 7e12 and
  samples with |z| about 1e6. The fix is a per-feature ridge (`RIDGE_REL = 0.05`). After the fix, missed beats are
  3–10 % and max |x| is about 23–41. The broken run is kept in `fixes_v2_unregularised/` (its 3 reference detector
  JSONs match the final ones).
- **V0 is post hoc.** It was added after seeing V2 = 0.939, as the missing matched control: same sources, real side and
  seeds. Treat it as exploratory.
- **Some statistics are inflated by the noise model itself.** Measured wander RMS is about 0.45 in full, oracle2,
  gen2 and ref_C, against real 0.31. Coefficient variance is inflated in every GP-leftover variant. The re-measurement
  (frozen high-pass) attributes part of the added noise to wander and part to beat deviations. These calibration
  numbers are reliable only for variants with real leftovers (V0, V1, V2, W1).
- **The edge-to-middle increment ratio is noisy.** V1 and V2 keep the real wander but measure 40–42 for SR, against
  real 20. Use it as a qualitative check only.
- Semi-synthetic sources are fit-part recordings and never overlap the real (dev) side; this is asserted in code.
  Patients are grouped on both sides.

## Next (for the Phase 4 decision)
1. **Leftover first.** It caps everything at 0.999. Candidates:
   - Bootstrap real leftovers from other fit-part recordings of the class. Check this with a detector first: is a
     *borrowed real* leftover detectable?
   - A non-stationary or non-Gaussian leftover model.
2. **Beat variation:** resample real V2 coefficient vectors (jointly, from other recordings) instead of Gaussian draws.
   Compare against V0 = 0.771.
3. **Keep PCA wander.** It wins in its band. Its full-band effect needs re-checking once the leftover is fixed.
