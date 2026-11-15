# Study 4, side question F: which clinical features give synthetic ECGs away? (plan, 2026-10-30)

**Status: plan, written before any code.** CPU only, so it runs while the GPU trains the residual model. Train
split only. The judge part is untouched.

## Why
The CNN detector says "1.00" but not *what* is wrong. A reader (or a cardiologist) needs the answer in ECG terms:
heart-rate variability, QRS width, amplitudes, noise. This is also the "handcrafted-feature second opinion" of plan
decision D3. It guards against tuning to one judge: a fix that fools the CNN but not a feature model, or the
reverse, is reported as such.

## Question
For each synthetic variant, which interpretable features differ from real ECGs, and how strongly? Two measures:
1. **Univariate AUC per feature.** This is model-free (rank-based), so it needs no trained model and is easy to
   read: 0.5 means no difference, 1 or 0 means complete separation. It is reported as |AUC − 0.5| + 0.5 with the
   direction (synthetic higher or lower), plus the real and synthetic medians.
2. **Multivariate feature detector.** Gradient-boosted trees on all features, with the same patient-grouped split
   and 5 seeds as the CNN detector, giving AUC with a CI. Permutation importance comes from the held-out side.

## Data
- **Real side:** exactly the CNN detector's real rows (dev part, 600 SR + all minority, seed 42, sha matched).
- **Synthetic sets (all existing, nothing new generated):**
  - `E`: the null control; must give AUC ≈ 0.5.
  - `C` (GP residual) and `D` (GP wander).
  - `oracle` (real templates + current assembly).
  - **`gen` (the current generator, study 1 pool: the ECGs studies 2–3 trained with).**
  - Later `LR` (learned residual) when it exists.
  - Real-vs-real check: fit-part ECGs as they are, against dev, sampled like E's sources.
- Semi-synthetic sets are grouped by their source patient; pools without sources get their own groups.

## Features (per ECG, from the 8 independent leads; about 26; fixed now)
- **Rhythm**, from frozen `detect_r_peaks`:
  - HR;
  - SDNN and RMSSD of RR;
  - RR coefficient of variation;
  - number of beats.
- **Template morphology**, from frozen `recording_template`:
  - QRS width: the span around R where the 8-lead derivative energy exceeds 20 % of its peak;
  - R amplitude (lead II);
  - S depth (lead II);
  - T amplitude: max absolute value of lead II in R + 150 … 400 ms, as a signed value;
  - ST level: lead II mean at R + 80 … 120 ms;
  - template peak-to-peak, mean over leads.
- **Beat-to-beat variation:**
  - median correlation of each full beat with the template;
  - SD of R amplitude across beats (lead II), relative to the mean.
- **Residual** (hp − place(template)):
  - RMS;
  - kurtosis;
  - edge-to-middle power ratio;
  - QRS-to-rest power ratio;
  - share of power above 15 Hz.
- **Wander** (x8 − hp):
  - RMS;
  - kurtosis of increments without the edge seconds;
  - edge-to-middle increment ratio.
- **Whole signal:**
  - mean pairwise cross-lead correlation;
  - spectral entropy (0.5–40 Hz, lead II);
  - total RMS;
  - max absolute amplitude.

ECGs without a template get NaN morphology features. Gradient boosting handles NaN; univariate AUCs drop NaN, and
the NaN count is reported.

## Phases (each ends with an audit of the code just written)
- **F0, code:** `synthetic_generation/realism/features.py` (pure functions) and `_study4_features.py`.
  - Unit checks before any real use:
    - features on a synthetic sine-plus-spike signal with a known HR give that HR;
    - the QRS width of a template widened in time grows;
    - kurtosis of Gaussian noise ≈ 3;
    - the edge ratio of stationary noise ≈ 1.
- **F1, controls first:**
  - E vs real and real-vs-real must give a multivariate AUC < 0.6 and no feature with univariate AUC > 0.65.
  - **If a control fails, stop and find the leak before looking at any other set.**
- **F2, run:** all sets. Univariate table and multivariate AUC.
- **F3, read-out:** `NOTES_features.md`. For gen: the top 5 features with medians and effect direction, in clinical
  words. Compare with the CNN detector's bands, and say whether the feature judge agrees with the CNN ranking of
  variants (C > D, etc.).

## What would make this a strong result
- If gen is separable on a few named features (e.g. "RR too regular", "residual not spiky", "edge transients
  missing"), we have a concrete, clinically readable account of where current ECG generators fall short.
- If features cannot separate gen (AUC near 0.5) while the CNN can (1.00), that is also a finding: the giveaway is
  below the level of standard ECG measurements, so feature-based realism checks (common in the literature) would
  pass this generator.

## Not doing
- No tuning of feature definitions after seeing results. A feature found broken in F0/F1 can be fixed, and is
  logged.
- No use of the judge part, the validation split or the test split.
