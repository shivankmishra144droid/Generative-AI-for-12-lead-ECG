# Study 4: overnight results (2026-10-29 → 30)

Plain summary first; details and numbers are in the linked notes. Everything below uses the **train split** (fit /
dev parts) or the **validation split** only. The test split and the study 4 "judge" part were not touched. Every
step was planned in writing before it ran, and every code change was audited (two bugs were found and fixed before
any affected result existed).

## Tier 1: solid findings (controls pass, reproduced two ways)

### 1. Twenty-five ordinary ECG measurements catch the current generator almost as well as the deep detector
- A gradient-boosted model on 25 standard ECG features (heart-rate variability, QRS width, amplitudes, noise
  statistics) separates the generator's ECGs from real ones with **AUC 0.973**. The CNN detector scores 1.000.
- **Controls pass:** real-vs-real gives 0.509 and real-rebuilt gives 0.512.
- **What gives it away, in clinical terms:**
  - The non-template activity is not spiky enough: kurtosis 9 vs real 30.
  - Beats are less alike than real beats: beat-to-template correlation 0.82 vs 0.96.
  - Noise and baseline wander are 65–70 % too strong.
  - Deviations are not concentrated at the QRS.
- **What doesn't give it away:** beat morphology. QRS width, R / S / T amplitude and ST level all have AUCs of
  0.44–0.55.
- `generator_comparison/study4/NOTES_features.md`

### 2. The noise model makes regular rhythms look like atrial fibrillation to R-peak detectors
- A variant that keeps each ECG's *real* beat timing and only swaps in the current noise model shows:

  | detector | missed beats | spurious beats | ECGs with any error |
  |---|---|---|---|
  | the project's own detector | 11 % | 20 % | 49 % |
  | wfdb `xqrs` (independent, standard) | 49 % | 65 % | ≈ 99 % |
  | either detector, real-ECG rebuild | 0 % | 0 % | 0 % |

  Measured heart-rate variability (SDNN) is inflated 3.3×, from 45 to 147 ms.
- In the generated pools, 30–51 % of the regular-rhythm ECGs (sinus, tachycardia, bradycardia) are as irregular as a
  typical AFIB recording. For real ECGs it is 3–7 %.
- The rhythm *model* itself is regular (0 % at AFIB level). So this comes **entirely from the noise model**.
- **Mechanism: structure, not loudness.** The real leftover noise, with its exact spectrum and amplitude but its time
  structure scrambled, still corrupts detection in 45 % of ECGs.
- `NOTES_features.md`, `features/rpeak_check.json`

### 3. The detector's verdict is not a storage artefact
- Real PTB-XL samples lie on a 1 µV grid, while synthetic samples do not.
- Snapping synthetic ECGs onto the grid changes nothing: AUC 0.78→0.79, 0.999→0.999 and 1.000→1.000.
- `NOTES_phase2_3b.md`

## Tier 2: negative but informative

### 4. A learned noise model (conditional diffusion) did not fix realism
- **Setup:** cross-fitted on two patient halves, 20,000 steps each. A schedule bug was found by a code review at step
  7,000, fixed, and training restarted; it is logged.
- **Detector:** still **1.000**.
- **Better than the current noise:**
  - 3× fewer beat-detection errors (10 % vs 31 % of beats);
  - less slow-drift leakage.
- **Still wrong:**
  - kurtosis 4–9 vs real 23–39;
  - a high-frequency noise floor of 14–25 % of power above 25 Hz, vs 3–4 % real.
- **Diagnosis:** the model's denoising at intermediate noise levels is too weak. A second sampler gave the same result.
- No memorisation: the maximum correlation to training data is 0.80.
- `NOTES_residual.md`

### 5. Earlier escalations also failed, with reasons
- **Envelope-modulated noise:** 0.998.
- **Template-relative beat variation:** 0.94, worse than none (0.77).
- **Real beat variation borrowed from another patient:** 0.83. Real variation transfers better than modelled.
- **Phase 4 of the plan** (port fixes to generated templates) is formally blocked, because no fix met the AUC ≤ 0.70
  bar.
- `NOTES_phase2_3.md`, `NOTES_phase2_3b.md`

## Tier 3: the usefulness test (finished 2026-10-30 ~08:20; exploratory, validation only)

### 6. Does the noise model matter for usefulness? (U2, `PLAN_usefulness.md` amendment)
- **Setup:** classifiers are trained with synthetic ECGs that differ **only** in their noise:
  - none;
  - the learned noise;
  - the current noise.

  Plus a real-only baseline. 10 seeds, validation split, analysis run once.
- **Mechanism readout fixed in advance:** if detection corruption carries over to the classifier, AFIB ↔
  regular-rhythm confusions should rank current > learned > none.
- **Results** (`generator_comparison/study4/NOTES_usefulness.md`):

  | synthetic arm | plain macro-F1 vs real | class separation (AUPRC) vs real | tuned macro-F1 vs real |
  |---|---|---|---|
  | current noise | +0.028 (10/10) | −0.000 | +0.001 |
  | learned noise | +0.029 (10/10) | +0.004 | −0.001 |
  | **no noise at all** | +0.025 (10/10) | +0.001 | −0.000 |

  - **All of the gain is threshold placement.** Each arm wins on plain macro-F1 in every seed. None improves class
    separation, and none beats a real-only model whose decision offsets are tuned on held-out validation patients.
  - **The noise model does not matter at the macro level.** Even noise-free synthetic ECGs work as well, with
    arm-vs-arm differences ≤ 0.004 AUPRC.
    - Caveat: all three arms are equally unrealistic (feature judge 0.97–1.00). What varied was beat-detection
      corruption (0–30 %), not realism.
    - Per class there may be a trade-off (no noise: −0.021 SBRAD, +0.015 STACH AUPRC; exploratory).
  - **The mechanism we proposed (beat-detection corruption → AFIB confusion) is not supported.** The pre-set
    confusion ordering did not appear (59.5 / 54.8 / 58.1). The classifier does not inherit the R-peak problem.

## What this means for the paper
- **For the generator story:** the diffusion model's beats are fine. The weak part is the hand-built noise
  assembly, and it fails in specific, measurable, clinically readable ways. The most practical of these is that it
  breaks beat detection.
- **For the augmentation story:** the generator's most visible flaw (noise that a CNN, plain ECG features and beat
  detectors all catch) has no measurable macro-level effect on classifier training. This holds across three versions
  of it (none, learned, current) spanning 0–30 % beat-detection corruption. A *realistic* noise arm was never
  available, so "realism is irrelevant" is not shown.
  - The augmentation gain replicates once more as a pure threshold effect: +0.03 untuned, ≈ 0 tuned, ≈ 0 AUPRC.
  - This matches studies 2–3 and the threshold exploration.
  - The confirmatory test is the pre-registered study 5 (a separate calibration split, then the
    test set).
- **Candidate headline (to be confirmed by study 5):** *"Synthetic ECG augmentation acts as decision-threshold
  calibration. Its gain vanishes against a tuned real-only baseline, and it is unchanged across three noise models,
  including none, spanning 0–30 % beat-detection corruption."* Our evidence for it so far:
  - studies 2–3 (AUPRC flat);
  - the threshold exploration (70–85 % of the gap closed);
  - U2: about 100 % of the gap closed, plausible range 75–120 %; three noise models, including none. This is the
    second exploratory look at the same validation set, which also chose the checkpoints.
- **Not claimed:** anything from the test split. Every overnight result is train or validation only, and
  exploratory except where marked.
