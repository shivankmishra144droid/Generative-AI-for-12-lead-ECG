# Study 4, U2: does the noise model matter for usefulness? Results (2026-10-30)

Plan: `PLAN_usefulness.md`, amendment U2, committed (2ebc5b1) before any pool or classifier run. Code:
`_study4_useful.py`. Raw: `useful/results.json`, `useful/analysis.json`, `useful/realism.json`,
`useful/val_prob_*.npy`.

**Exploratory, validation split only.** The test split was never loaded. The analysis ran once (guarded).

## Design
Three synthetic pools differ **only** in the residual. They share the same generated templates (study 1/2
generator), the same sampled R-peaks and the same GP wander:
- **N:** no residual.
- **B:** the learned residual.
- **A:** the current GP residual.

Each adds the study 2 synthetic counts (+1,056 AFIB / +568 STACH / +446 SBRAD) to the full real training set. Ten
seeds (600–609) use the study 2 recipe (6,250 steps, deterministic, best validation checkpoint), plus a real-only
reference: 40 runs, all in one attempt.

## Realism of the three pools (before training): the corruption axis holds
| pool | beat errors, project detector (missed + spurious) | ECGs with an error, project / xqrs | feature judge AUC |
|---|---|---|---|
| A (current GP) | 12 % + 18 % = 30 % | 50 % / 98 % | 0.969 |
| B (learned) | 9 % + 4 % = 12.5 % | 45 % / 98 % | 0.996 |
| N (none) | 0 % | 0 % / 0 % | 0.972 |

Truth here is the same detector applied to the residual-free version of each ECG.

## Results (validation, mean ± SD over 10 seeds)
| condition | macro AUPRC | plain macro-F1 | tuned macro-F1 | AFIB ↔ regular confusions |
|---|---|---|---|---|
| real only | 0.8843 ± 0.0051 | 0.7887 ± 0.0074 | 0.8421 ± 0.0076 | 55.0 |
| A: + synthetic, GP noise | 0.8842 ± 0.0067 | 0.8171 ± 0.0083 | 0.8431 ± 0.0085 | 59.5 |
| B: + synthetic, learned noise | 0.8880 ± 0.0073 | 0.8181 ± 0.0102 | 0.8409 ± 0.0099 | 54.8 |
| N: + synthetic, no noise | 0.8854 ± 0.0073 | 0.8137 ± 0.0053 | 0.8419 ± 0.0080 | 58.1 |

Tuned macro-F1 uses per-class logit offsets tuned on one half of the validation patients and scored on the other
half (50 splits × 2 directions), with `_threshold_explore.tune` imported unchanged.

### Paired differences (10 seeds; unadjusted Wilcoxon p; bootstrap 95 % CI over seeds)
| comparison | macro AUPRC | plain macro-F1 | tuned macro-F1 |
|---|---|---|---|
| **A − real** | −0.0002 [−0.003, +0.003] | **+0.0284** [+0.020, +0.037], 10/10, p 0.002 | +0.0010 [−0.006, +0.007] |
| **B − real** | +0.0037 [−0.001, +0.008] | **+0.0294** [+0.020, +0.039], 10/10, p 0.002 | −0.0012 [−0.009, +0.006] |
| **N − real** | +0.0011 [−0.003, +0.006] | **+0.0250** [+0.019, +0.031], 10/10, p 0.002 | −0.0002 [−0.006, +0.006] |
| N − A (primary) | +0.0012 [−0.005, +0.008], p 0.85 | −0.0034 [−0.009, +0.002] | −0.0012 [−0.010, +0.008] |
| B − A (primary) | +0.0039 [+0.001, +0.007], p 0.049 | +0.0010 [−0.006, +0.007] | −0.0023 [−0.010, +0.005] |
| N − B | −0.0027 [−0.009, +0.004] | −0.0044 [−0.010, +0.002] | +0.0010 [−0.008, +0.010] |

Per-class AUPRC against A:

| comparison | SBRAD | STACH |
|---|---|---|
| N − A | −0.021 (2/10) | +0.015 (7/10) |
| B − A | +0.008 | +0.005 |

## Findings
1. **Synthetic augmentation's gain is entirely threshold placement, whatever the noise model.**
   - Every synthetic arm raises plain macro-F1 by +0.025 to +0.029 over real-only, in 10 of 10 seeds.
   - **None** raises macro AUPRC (all within ±0.004 of real).
   - **None** beats real-only once each model's decision offsets are tuned on held-out validation patients: tuned
     differences are −0.001 to +0.001.
   - Share of the untuned gap closed by tuning: about 100 %. Given the tuned CI (for A, −0.006 to +0.007 against an
     untuned gap of about +0.028), the plausible range is roughly 75–120 %. This is the second exploratory look at
     a validation set that also chose the checkpoints.
   - This independently replicates the threshold account (studies 2–3; THRESHOLD_EXPLORE.md) with a new pool, new
     seeds and three noise models.
2. **The choice among three noise models (none, learned, current GP) does not change usefulness on macro metrics.**
   - All three are equally unrealistic: feature-judge AUC 0.969–0.996, and the CNN detector is at 1.00. What varies
     is beat-detection corruption (0–30 % of beats) and whether there is any noise at all.
   - **Realism itself was not varied**, because no realistic noise arm exists. So this does *not* show that realism
     is irrelevant.
   - Synthetic ECGs with no noise at all (N) give the same gain as the current noise (A) and the learned noise (B).
   - N − A and N − B are within ±0.004 AUPRC with CIs spanning 0.
   - B − A is +0.004 AUPRC (p = 0.049 unadjusted, two primary tests), small and not visible in tuned F1. It is not
     treated as a real effect.
   - Pre-set reading: N ≈ B ≈ A. The N − A CI slightly exceeds the ±0.005 band (−0.005 to +0.008), so "no practical
     effect" is the honest statement, not "proven equivalence".
   - **Per-class trade-off (exploratory, unadjusted).** N − A is −0.021 SBRAD AUPRC (2 of 10 seeds favour N) and
     +0.015 STACH. Both are larger than any macro difference, and they cancel in the macro. The noise model may move
     separation between SBRAD and STACH even though the macro is unchanged.
3. **The detection-corruption mechanism is not supported.**
   - The pre-set readout required AFIB ↔ regular-rhythm confusions ordered A > B > N. Observed: 59.5 / 54.8 / 58.1,
     not ordered. AFIB precision and recall are flat (0.83–0.84 / 0.93).
   - The CNN-LSTM does not inherit the R-peak corruption that standard detectors suffer.
   - The corruption remains a real problem for any **HRV- or R-peak-based** use of these synthetic ECGs
     (NOTES_features.md). It is not why classifier gains are threshold-only.

## What it means
- The generator's most visible flaw is non-physiological noise that breaks beat detection, caught by a CNN and by
  ordinary ECG features. Across three versions of it (none, learned, current), it has no measurable effect on
  macro-level classifier training.
- Whether a *realistic* noise model would help cannot be answered yet; none was built.
- The augmentation benefit itself is fully explained by threshold placement: a real-only classifier with tuned
  decision offsets does as well.
- So, unless realism changes something these three arms do not, improving the noise model should not be expected
  to improve augmentation at the macro level. That is a reasoned expectation, not a tested result. The decisive test of
  "synthetic = threshold calibration" is the pre-registered study 5 (PREREGISTRATION_5.md), with a
  separate calibration split and the test set.

## Limitations
- Validation is used both for checkpoint selection and for the outcome, and is split for tuning. This is the same
  limitation as THRESHOLD_EXPLORE.md. It is the second exploratory use of the validation split.
- One pool (one template set, one rhythm draw). The CIs cover classifier-seed variation only.
- 100 % real data only. One generator family and one classifier.
- Exploratory; two primary comparisons, unadjusted.
