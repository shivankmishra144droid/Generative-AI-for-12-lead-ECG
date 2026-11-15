# Study 4, D5 escalation: learned residual model. Rules fixed before training (2026-10-30)

Why: `NOTES_phase2_3b.md`. Stationary and envelope-modulated GP leftovers are detected at 0.998–0.999. Even the real
leftover, phase-randomised, scores 0.993. The giveaway is time-domain structure, so the plan's next rung (D5) is a
learned model.

## Design (changed from the plan's "R-aligned leftover segments")
- **Target:** the full residual r = highpass(x8) − place(template, R), as in the current generator's split. This
  covers per-beat variation and the leftover together.
- **Condition:** place(template, R) as 8 extra input channels, plus the class one-hot. This is exactly what exists
  at generation time (a generated template at sampled R-peaks).
- **Why the change:**
  - It removes the pairing confound of swapping only the deviations (the 0.77 floor in `NOTES_phase2_3b.md`), so
    E = 0.5 is again the reference.
  - It drops the Gaussian V2 variation (0.94).
  - It follows the plan's own premise that fixes 1 and 2 are one problem.
- **Wander stays separate:** own wander, or PCA wander (W1), which is the one component that works.

## Model and training (`synthetic_generation/realism/residual_diffusion.py`, `_study4_residual.py`)
- **Backbone:** the frozen `SSSD_ECG` class (not edited), built with `in_channels = 16` and `out_channels = 8`, 8 ×
  64, S4 lmax 1000, 1.53 M parameters.
- **Diffusion:** T = 200 linear schedule (frozen `calc_diffusion_hyperparams`), v-prediction, ancestral DDPM
  sampling, EMA 0.999 weights used for sampling.
- **Data:** fit part only; 5,510 of 5,545 recordings have a template.
  - Patient-level folds 0/1, seed 4005. Fold 0: SR 2,343 / AFIB 227 / STACH 107 / SBRAD 76. Fold 1: 2,365 / 181 /
    118 / 93.
  - 5 % of each fold's patients are held out to track the denoising loss only. **No early stopping and no checkpoint
    choice.**
- **Training:** **20,000 steps fixed**, batch 16, Adam 2e-4, 500 warm-up steps then cosine decay to 0.1×, gradient
  clip 1.0, class-weighted batches (1 / √count).
  - Scale: one global constant per fold for the residual and one for the condition, from that fold's training rows.
  - Feasibility (measured): 0.23 s/step, 1.6 GB peak, so about 1.3 h per fold.
  - **One training run per fold.** A crash resumes from its checkpoint; that is not a new attempt.
- **Cross-fitting:** each semi-synthetic source (the ~1,600 fit-part V2 sources of `_study4_fixes.py`) gets its
  residual from the model of the **other** fold.

## Evaluation order (fixed)
1. **Calibration before any detector.** Learned vs the sources' real residual, per class:
   - RMS quantiles and log-RMS SD;
   - kurtosis, all and without the edge seconds;
   - edge-to-middle power;
   - QRS-to-rest power;
   - power shares in 0–0.5 / 0.5–2 / 2–5 / 5–15 / 15–25 / 25–50 Hz.
2. **Memorisation:**
   - nearest correlation of each generated residual to the generating model's own training residuals; > 0.98 is a
     copy, and a copy fraction above 1 % fails;
   - also reported: correlation with the source's own true residual. This measures prediction from the template, not
     copying.
3. **Detector protocol v2:** same real side, sources and seeds as Phase 2–3.
   - LR = own template at own R + learned residual + own wander: full, between and < 0.5 Hz bands.
   - LRW = the same with PCA wander: full band.
   - Then, only if LR moves: oracle (real templates, sampled rhythm) and gen (generated templates).

## Success criteria (stated now)
- **"Moved":** LR full-band AUC with a 95 % CI entirely below 0.99. The references are ref_C 0.999, R1 0.998 and
  P_rand 0.993.
- **D1 met:** AUC ≤ 0.70.
- **Failure:** a CI that reaches 0.99. This is recorded as such. The model is not retrained with new settings
  against the detector.

## Log of deviations (kept in order)
1. **2026-10-30 ~03:00:** attempt 1 of fold 0 was ended at step 1,000. This was not a crash: it was stopped deliberately
   during a code review. `run_retry.sh` resumed it from the step-1,000 checkpoint.
2. **2026-10-30 ~03:35: schedule bug, training restarted from scratch.**
   - The first build used β 1e-4 → 0.02 (copied from `synthetic_generation/config.json`). That schedule leaves
     √ᾱ_T = 0.36 of the signal at t = T, while sampling starts from pure N(0, 1). This is the train/sample mismatch
     REPORT §5.2 had already found. Every other model in the project uses 5e-4 → 0.1 (√ᾱ_T = 0.006).
   - Found by a code review at step 7,000 of fold 0, **before any sample or detector run**. Fold 0 was stopped,
     the schedule fixed, and both folds retrained from step 0 with every other setting unchanged.
   - The partial run (checkpoint and log) is kept in `generator_comparison/study4/residual_attempt_bad_schedule/`.
   - This is a bug fix, not a tuning choice: no output of the old model was looked at.
   - Budget check from REPORT §5.2: full-length diffusion at 8 × 64 needed about 320 k ECGs seen to get the
     amplitude right. 20,000 steps × batch 16 = 320 k per fold, so the budget is unchanged.

## Results (2026-10-30 morning): the learned residual fails
**Training.** Both folds ran 20,000 steps, one attempt each after the schedule fix.

| fold | train loss | held-out loss |
|---|---|---|
| 0 | 0.45 | 0.76 (flat since about step 10 k) |
| 1 | 0.42 | 0.47 |

Logged only, as fixed in advance.

**Memorisation: pass.**

| comparison | value |
|---|---|
| nearest correlation to the generating model's training residuals (q50 / q99 / max) | 0.23–0.27 / 0.66–0.67 / 0.78–0.80 |
| copies (> 0.98) | 0 |
| correlation with the source's own true residual (median) | 0.13 |

**Calibration** (`residual/calibration.json`, real → learned):

| class | kurtosis | > 25 Hz power share | RMS q10 | QRS-to-rest power | edge-to-middle power |
|---|---|---|---|---|---|
| SR | 33.6 → 8.9 | 0.04 → 0.19 | 0.24 → 0.40 | 8.6 → 3.6 | 2.6 → 1.6 |
| AFIB | 26.3 → 9.0 | 0.03 → 0.14 | 0.27 → 0.40 | 4.0 → 2.6 | 1.9 → 1.3 |
| STACH | 23.4 → 8.1 | 0.03 → 0.15 | 0.28 → 0.42 | 4.1 → 2.7 | 1.8 → 1.3 |
| SBRAD | 39.4 → 4.2 | 0.04 → 0.25 | 0.19 → 0.37 | 13.7 → 3.3 | 2.8 → 1.4 |

The learned residual has a high-frequency noise floor. It is not spiky enough, and only partly locked to the beats.

**Diagnosis (held-out residuals, no detector).**
- Re-noising a real held-out residual to step t and denoising it back:

  | start step | > 25 Hz share (real 0.03–0.04) | correlation with the original |
  |---|---|---|
  | t = 5 | 0.04–0.05 | 0.99 |
  | t = 20 | 0.07 | 0.95 |
  | t = 60 | 0.14 | 0.79 |
  | t = 199 | 0.16–0.18 | 0.25 |

- **Deviation 3, logged:** a deterministic DDIM sampler (η = 0 and 0.5) was tried on held-out data, judged by
  physiology only. It gave the same > 25 Hz share (0.17–0.19) and kurtosis (7–13), so it was not adopted and the code
  was removed.
- The weakness is the model's denoising at intermediate noise levels: too little capacity or data for the fine
  structure. It is not the sampler.

**Detector** (pre-set success criterion: CI entirely below 0.99):

| variant | band | AUC [95 % CI] |
|---|---|---|
| LR (learned residual + own wander) | full | **1.000** [0.999, 1.001]: **FAIL** |
| LR | between beats | 0.991 [0.987, 0.994] |
| LR | < 0.5 Hz | 0.832 [0.812, 0.853] (C: 0.948; the learned residual leaks less low-frequency power than the GP) |
| LRW (with PCA wander) | full | 0.999 [0.997, 1.001] |

**R-peak detection** (project detector; truth = the real source's R-peaks):

| set | missed | spurious | ECGs with any error | SDNN, true → measured |
|---|---|---|---|---|
| LR | 7.6 % | 2.4 % | 39 % | 45 → 87 ms |
| C (current GP) | 11.2 % | 20.1 % | 49 % | 45 → 147 ms |

The learned residual corrupts detection about 3× less (10 % vs 31 % of beats), but still clearly.

**Verdict.**
- The learned residual is **not** a usable fix in this form: the detector is at 1.00 and it fails kurtosis
  calibration.
- The oracle / gen pool stages are not run (pre-set: only if LR moved).
- The usefulness gate in `PLAN_usefulness.md` fails on kurtosis (learned 4–9, against C's about 9 and real about 30),
  so U as planned is not run.
- **Detector tally:** 4 more evaluations, about 66 in total.
