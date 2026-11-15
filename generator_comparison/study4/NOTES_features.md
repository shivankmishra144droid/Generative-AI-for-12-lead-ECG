# Study 4, side question F: which clinical features give synthetic ECGs away? (2026-10-30)

Plan: `PLAN_features.md`, written before any code. Code: `synthetic_generation/realism/features.py` (unit-tested in
`tests/test_study4_features.py`), `_study4_features.py`, `_study4_rpeak_check.py`. Results:
`features/results.json`, `features/rpeak_check.json`.

Setup:
- Train split only. The real side is the CNN detector's dev rows.
- Feature judge: gradient-boosted trees on 25 standard ECG features, patient-grouped, 5 seeds.
- Univariate AUCs are class-stratified: the per-class AUC, averaged over the 4 classes.
- **Exploratory.** Feature definitions were fixed in the plan and not changed after results.

## Controls (phase F1): pass
| set | feature-judge AUC [95 % CI] | largest univariate separation |
|---|---|---|
| real vs real (fit-part ECGs as they are) | 0.509 [0.487, 0.531] | 0.53 |
| E (real ECG rebuilt from its own parts) | 0.512 [0.496, 0.527] | 0.53 |

## Main result: 25 standard ECG features catch the current generator almost as well as the CNN
| set | feature judge | CNN detector (Phase 0) |
|---|---|---|
| **gen** (current generator, the pool studies 2–3 trained on) | **0.973** [0.970, 0.976] | 1.000 |
| oracle (real templates + current assembly) | 0.976 [0.973, 0.979] | 1.000 |
| C (real ECG, GP residual) | 0.977 [0.973, 0.982] | 1.000 |
| D (real ECG, GP wander) | 0.965 [0.960, 0.970] | 0.888 |

What the CNN sees is mostly visible in plain ECG measurements. This makes the account interpretable, and it means
feature-based realism checks would catch this generator too.

## What gives the current generator away, in clinical terms (gen vs real, medians)
| feature | real | synthetic | univariate AUC | reading |
|---|---|---|---|---|
| residual kurtosis | 29.7 | 8.8 | 0.14 | Real non-template activity is **spiky**: sharp, sparse, beat-specific deviations. Synthetic is diffuse Gaussian-like noise. Most important feature in the judge. |
| median beat-to-template correlation | 0.96 | 0.82 | 0.13 | Real beats are **more alike** than synthetic ones. The noise is spread across every beat instead of concentrated in a few places. |
| residual RMS | 0.41 | 0.70 | 0.75 | Synthetic ECGs carry about 70 % more non-template signal. |
| wander RMS | 0.27 | 0.44 | 0.72 | Baseline wander is about 65 % too strong. Part of it is GP residual leaking below 0.5 Hz (Phase 2–3 audit). |
| QRS-to-rest residual power | 4.96 | 2.35 | 0.30 | Real deviations concentrate at the QRS; synthetic deviations do not. |
| RR CV / SDNN / RMSSD | 0.043 / 36 ms / 52 ms | 0.19 / 119 ms / 160 ms | 0.67–0.69 | Synthetic sinus ECGs **look arrhythmic** to a standard R-peak detector. See the next section. |

Template morphology is mostly **not** a giveaway (values checked, not only the top-9 list):

| feature | univariate AUC |
|---|---|
| QRS width | 0.54 |
| R amplitude II | 0.49 |
| S depth II | 0.55 |
| T amplitude II | 0.44 |
| ST level II | 0.48 |

The exception is that generated templates are about 14 % smaller peak-to-peak (median 4.34 vs 5.06, AUC 0.35).
Heart rate reads higher (86.7 vs 74.1 bpm, AUC 0.60), partly because spurious beats are detected (next section).

## Finding: the noise model corrupts R-peak detection (`rpeak_check.json`)
*Detector named precisely:* "project" is the project's frozen R-peak detector, `beats.detect_r_peaks` (8-lead,
Lead-II envelope with derivative-energy refinement). "xqrs" is wfdb's `xqrs_detect` on Lead II in mV: an independent,
widely used detector, already installed, nothing downloaded. The first table below is the project detector.
C keeps each source's *own real* R-peak timing and template. Only the residual is replaced by the current Gaussian
process. Running the frozen R-peak detector on C and on its real source, with a match window of 50 ms:

| set | missed beats | spurious beats | ECGs with any error | SDNN, true → measured |
|---|---|---|---|---|
| E (real rebuild) | 0.0 % | 0.0 % | 0 % | 45 → 45 ms |
| **C (GP residual)** | **11.2 %** | **20.1 %** | **49 %** | **45 → 147 ms** |
| D (GP wander) | 0.6 % | 0.6 % | 6 % | 45 → 46 ms |

- With the current residual model, half of all synthetic ECGs have beat-detection errors, and measured heart-rate
  variability is inflated about 3.3×.
- The rhythm *model* is not at fault: C's true rhythm is real. The noise *model* makes regular rhythms look irregular
  to standard tooling.
- **Practical consequence (exploratory, not tested here):** any HRV-based use of these synthetic ECGs would be wrong.
  A classifier trained on them sees "irregular-looking" STACH / SBRAD / SR, closer to AFIB in RR terms. This is one
  candidate mechanism for why the synthetic data shifted the operating point rather than improving class separation
  (studies 2–3, THRESHOLD_EXPLORE.md). Testing it would need a new study.

## Agreement with the CNN detector
- The feature judge ranks the variants as the CNN does: C ≈ oracle ≈ gen (top) > D.
- Its D score is higher than the CNN's (0.965 vs 0.888). Its top D feature is the wander edge-transient ratio (real
  23.4 vs GP 2.6), the edge effect Phase 1 found.
- Two independent judges, one a black-box CNN and one interpretable, reach the same verdict, so the conclusion does
  not depend on one detector.

## Implications for the learned residual model (training now)
The targets are now concrete and measurable:
- residual kurtosis about 30;
- beat-to-template correlation about 0.96;
- residual RMS about 0.41;
- **R-peak detection error near 0 %**.

These are physiology checks, independent of the detector, and will be applied to the learned residual before any
detector run.

## Follow-up: generated ECGs of regular rhythms look like AFIB to a standard detector (descriptive)
- **Measurement.** RR coefficient of variation from the frozen R-peak detector, from cached features
  (`features/feat_*.npz`).
- **Threshold.** The real AFIB median, 0.209.
- **Rhythm model alone.** 400 R-peak sequences per class drawn from the class rhythm models (`fixes/models.pkl`, the
  same recipe as the gen pool's assembly), measured directly with no ECG and no detector.

| class | real ECGs: share ≥ AFIB median | generated ECGs: share ≥ AFIB median | rhythm model alone: share ≥ AFIB median (median RR CV) |
|---|---|---|---|
| SR | 3 % | **39 %** | 0 % (0.017) |
| AFIB | 49 % | 70 % | 37 % (0.179) |
| STACH | 7 % | **51 %** | 0 % (0.021) |
| SBRAD | 4 % | **30 %** | 0 % (0.015) |

**Reading:**
- The rhythm model produces regular SR / STACH / SBRAD rhythms, slightly *more* regular than real.
- Once the generated ECG is measured, 30–51 % of those regular-rhythm ECGs are as irregular as a typical AFIB
  recording.
- So the irregularity comes **entirely from the noise model** corrupting beat detection, the same mechanism quantified
  on C above (11 % missed / 20 % spurious beats, with real R-peak timing kept).
- **Limitation:** the classifier in studies 2–3 is a CNN-LSTM, not an R-peak pipeline. Whether it picks up the same
  cue is untested. The "AFIB-like synthetic STACH" mechanism stays a hypothesis.


## R-peak corruption: independent detector and mechanism (`_study4_rpeak_check.py`, extended)
Every set keeps its source's own R-peaks and template, so the detector's output on the real source is the truth.

| set (what replaces the real part) | detector | missed | spurious | ECGs with any error |
|---|---|---|---|---|
| E (nothing: real rebuild) | project / xqrs | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| C (stationary GP residual: the current generator's) | project / xqrs | 11 % / **49 %** | 20 % / **65 %** | 49 % / **≈ 99 %** |
| D (GP wander only) | project / xqrs | 0.6 % / 16 % | 0.6 % / 16 % | 6 % / see json |
| R2 (stationary GP leftover after the per-beat fit) | project | 13 % | 10 % | 56 % |
| R1 (GP leftover × edge and beat-phase envelopes) | project | 16 % | 4 % | 65 % |
| **P_rand (own real leftover, phase-randomised: exact real spectrum and scale)** | project | 12 % | 7 % | **45 %** |
| VS (own leftover, real deviation coefficients shuffled across beats) | project | 7 % | 1 % | 38 % |

**Reading:**
- **An independent detector confirms the effect, and it is stronger there.** xqrs is single-lead and more
  noise-sensitive. It flags nearly every ECG with the current noise model and none of the real rebuilds.
- **Mechanism: structure, not loudness.** P_rand keeps the real leftover's exact power spectrum and per-recording
  amplitude, yet still corrupts detection in 45 % of ECGs. The real residual is spiky and locked to the QRS.
  Spreading the same energy evenly over time (Gaussian-like) puts QRS-band bursts between beats and blurs the true
  QRS.
- **Every modelled residual tried so far does this:** stationary GP, envelope GP and phase-randomised real.
- VS errors are mostly *displaced* peaks: shuffled shift coefficients move the R fiducial by more than 50 ms. VS is a
  diagnostic, not a candidate model.
- **Consequence for the learned residual (LR):** the R-peak error rate is a physiology test that does not need the CNN
  detector, and it is the U gate in `PLAN_usefulness.md`.
