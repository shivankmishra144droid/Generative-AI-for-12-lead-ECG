# Study 4, question W: can whole-signal generation make synthetic ECGs look real? (plan, 2026-10-30)

**Status: plan, written before any download, any code, or any look at the published data.** Exploratory realism work
inside study 4. It asks about realism only. Classifier usefulness is out of scope for this plan; that thread was
dropped on 2026-10-30.

## No plan can guarantee realism. What this one guarantees
- **Every phase ends in a reportable result.** If a gate fails, that failure is the finding, and we stop or switch
  branch.
- **Nothing expensive runs before a cheap gate passes.** No GPU training starts until a no-GPU check (Phase 3) shows
  that whole-signal generation at paper scale is realistic by our own measures.
- **Gates are fixed below, before any data is seen.**
- **Every number is checked against a control** that proves the pipeline itself cannot produce it.

## Why this question, and why now
- **Every fix to the hybrid generator (template + rhythm model + noise) failed.** That includes envelope noise,
  modelled variation and a learned diffusion residual (NOTES_phase2_3b.md, NOTES_residual.md). Detector AUC stayed
  at 0.94–1.00. Each hand-modelled part adds its own failure: the noise breaks R-peak detection, and beat
  variation is wrong (NOTES_features.md).
- **The alternative is to generate the whole 10 s ECG with one model,** so beats, variation, noise and rhythm come
  out together.
- **Week 4 tried this and failed,** but only at 8×64 and 16×128, which a 4 GB GPU allows (REPORT.md §5.2). Those
  models produced no regular heartbeat. The published SSSD-ECG uses **36×256** (config_SSSD_ECG.json) and was never
  tried here.
- **The authors published both things we need,** each under an MIT licence (figshare, checked 2026-10-30):
  - **the synthetic PTB-XL dataset** (`Dataset.zip`, 929.87 MB; https://figshare.com/s/43df16e4a50e4dd0a0c5);
  - **the trained weights at 100,000 iterations** (611.77 MB; https://figshare.com/s/81834b24a4711c2a5c55).
- **So we can test paper-scale whole-signal realism without any GPU,** by judging their synthetic ECGs with our
  study 4 suite.

## Facts this plan relies on (checked 2026-10-30)
- **Published model** (arXiv 2301.08227; repo config):
  - format: 100 Hz, 1,000 samples, 8 leads (I, aVF, V1–V6), conditioned on a 71-dim multi-label statement vector;
  - architecture: 36 residual layers, 256 channels, S4 with `s4_lmax` 1000;
  - training: T = 200, β 1e-4 → 0.02 linear, 100k iterations, batch 8, learning rate 2e-4.
  - The generator was trained on "the eight PTB-XL training folds", so folds 9 and 10 were held out.
- **Our splits are patient-level random, not by PTB-XL `strat_fold`.** The share of each split inside folds 1–8:
  train 79.6 %, val 81.8 %, test 80.2 %.
  - **The published model saw about 80 % of our validation and test patients.** Neither its data nor its weights
    may ever be used for a classifier result; this plan has none.
  - For realism the overlap is handled by the comparison set below.
- **Our lead conventions:**
  - our independent leads are I, II, V1–V6 (`synthetic_generation/leads.py`), so II = aVF + I/2 in mV;
  - `reconstruct_12` derives the other four limb leads;
  - normalisation uses train-split per-lead mean and SD (`normalization_stats.json`).
- **Week 4 noise-schedule finding:** β 1e-4 → 0.02 at T = 200 leaves about 36 % of the signal at t = T
  (REPORT.md §5.2). The published config uses exactly this schedule. We note it, and we reproduce it as published:
  we do not "fix" it in Phase 5.

## Fixed rules (restated)
- **Test split:** never loaded.
- **Study 4 judge part:** used once, in the final registered test (Phase 7) only.
- **Frozen code:** `check_frozen()` runs at the start and end of every script.
- **Code location:** new code only, in `_study4_full_*.py` and `synthetic_generation/realism/`.
- **Study 5:** none of its files are touched (`_study5_*`, `PREREGISTRATION_5.md`, `generator_comparison/study5/`).
- **Downloads:**
  - each needs explicit approval (file, source, size);
  - each goes into its own new, empty folder (`external/sssd_ecg/<item>/`, git-ignored);
  - Python that reads it runs with `-I`;
  - no code from the download is executed. We read their arrays only. The network code for Phase 4 is written or
    ported by us after reading theirs.
- GPU use is coordinated with other running jobs before any GPU phase.
- **Audit:** every phase ends with a written audit of its own code before its numbers are used.

---

## Phase 1: download and inspect the published synthetic dataset (CPU, ~1 h)
**Needs:** approval to download `Dataset.zip` (929.87 MB) from figshare.
1. Unzip into `external/sssd_ecg/dataset/`. List every file with its shape, dtype, min/max and NaN count.
2. **Identify the format from the files, not the paper:**
   - array layout (N, 8, 1000) or (N, 1000, 8);
   - which axis is leads;
   - the label array (N, 71);
   - whether data and labels are aligned 1:1 with PTB-XL records, and whether there is a record index.
3. **Lead-order check:**
   - Compute each synthetic lead's median-beat shape and per-lead amplitude percentiles.
   - Do the same for real ECGs converted to I, aVF, V1–V6 (step 4 of Phase 2).
   - Assign leads by best correlation of mean beat shape. **Pass if the assignment is the identity permutation**
     for the paper's stated order, with known signatures: V1 has a mostly negative QRS, and lead I a positive one.
4. **Units check:** the median per-lead peak-to-peak amplitude must be within a factor of 1.5 of real in mV.
   - If not, look for a stated scaling in the repo's data code (read only).
   - If none explains it, **stop and report**; do not guess a rescale.
5. **Label mapping:**
   - Identify the 71-statement order from the repo's PTB-XL preprocessing code (read only), or from the data if it
     ships a label list.
   - Map labels to our 4 classes with our own rule: exactly one of SR / AFIB / STACH / SBRAD present.
   - Report per-class counts.
   - **Sanity check (must pass):** median heart rate ordered SBRAD < SR < STACH, and AFIB has the highest RR
     irregularity. Both are measured with our R-peak detectors. If this fails, the mapping is wrong: stop.

**Exit:** a short `NOTES_fullsignal.md` §1 with formats, the lead permutation, units, counts and the sanity check.

## Phase 2: conversion control (CPU, ~1 h); no realism number is trusted before this passes
The risk is that a unit, lead or filter mismatch alone produces AUC 1.00, and we wrongly conclude "scale does not
help". Two controls:
1. **Identity check:**
   - Load raw PTB-XL `records100` for 200 train-split recordings, build I, aVF, V1–V6 in mV (their format), then
     run them through our conversion path (→ I, II, V1–V6 → `reconstruct_12` → normalise).
   - Compare with the same rows of `X_train.npy`.
   - **Pass if max |difference| < 1e-3** in z units on every lead.
2. **Detector control:**
   - Run detector v2 (`synthetic_generation/realism/detect.py`, unchanged) on two disjoint patient sets of real
     ECGs, one taken through the conversion round trip.
   - **Pass if the AUC is within the real-vs-real band of NOTES_phase0_1.md**, around 0.5, with the per-class
     t-interval containing the study 4 real-vs-real value.

**If either fails:** fix the conversion and rerun the controls. No Phase 3 number is computed until both pass.

## Phase 3: is the published paper-scale generator realistic by our measures? (CPU, ~3–5 h): **the decisive gate**
### Comparison sets, fixed now
- **Real side:** the study 4 **dev** part ∩ PTB-XL folds 9–10. These patients were seen by neither the published
  generator nor any study 4 model; about 800 recordings.
  - If a class has fewer than 30 there, the per-class result is reported as "too few" and excluded from per-class
    gates.
- **Synthetic side:** published synthetic ECGs with exactly one target rhythm label.
  - Per class, sample the same count as the real side (seed 7000).
  - Group by sample index: each is independent.
- **Secondary real side:** the same dev part ∩ folds 1–8. These patients **were** seen by the generator. If the AUC
  is much lower here than on folds 9–10, the generator is reproducing training patients, not generalising.

### Measures (all existing study 4 code, unchanged, imported)
1. **Detector v2:**
   - five seeds;
   - bands full, low, mid, beats and between;
   - per-class t-intervals.
2. **Feature judge:**
   - the 25 features (`synthetic_generation/realism/features.py`);
   - HistGradientBoosting, as in `_study4_features.judge`;
   - plus the class-stratified univariate AUCs, so any failure is named in clinical terms.
3. **Rhythm realism:**
   - per class: heart rate, SDNN and the fraction of ECGs as irregular as a typical AFIB recording, against the real
     side;
   - the project detector and wfdb `xqrs`, both as in `_study4_rpeak_check.py`.
   - There is no ground-truth R-peak for whole-signal samples, so this compares distributions, not errors.
4. **Periodicity** (REPORT.md §5.1 metric, `generator_comparison/eval_metrics.py`), against real. This shows whether a regular heartbeat exists at all.
5. **Memorisation:**
   - for each synthetic ECG, the maximum correlation to every train-split ECG;
   - the fraction above 0.98 counts as copies.
   - If copies are more than 5 %, the realism result is flagged as possibly memorisation-driven, and Phase 3 is
     rerun excluding them.
6. **Reference row:** the current hybrid generator's numbers on the same real side (detector 1.000, judge 0.97),
   recomputed with this exact sampling so the two rows are comparable.

### Gate G1 (fixed now)
- **PASS, scale fixes realism:** all three hold. Go to Phase 4.
  - detector v2 full-band mean AUC ≤ 0.70;
  - feature judge AUC ≤ 0.70;
  - the AFIB-irregular fraction in each regular class ≤ 2 × real.
- **FAIL, scale is not enough:** detector v2 full-band mean AUC ≥ 0.95.
  - Stop this branch. Write up the finding: a published, paper-scale ECG diffusion model is still trivially
    detectable by a small CNN and by 25 standard features, with the failing features named.
  - Then go to the **fallback (Phase F)**.
- **PARTIAL** (anything else):
  - Use the band results and univariate AUCs to name the failing property.
  - Proceed to Phase 4 only if both hold:
    - the feature judge AUC is ≤ 0.85;
    - heart rate and periodicity are realistic (the class HR ordering holds, and median periodicity is within 20 %
      of real).
  - Otherwise go to Phase F.
- Whatever the outcome, Phase 3 is written up and committed before Phase 4 starts.

## Phase 4: reproduce their sampler from their weights, then size the compute (GPU, ~½–1 day); only if G1 passes or is partial
**Needs:** approval to download the weights (611.77 MB).
**Why:** before training our own model, prove that our port of their network and sampler is correct. A silent bug
there would waste days of training.
1. **Port the network.**
   - Write our own `synthetic_generation/realism/sssd_full.py`, matching their 36×256 config. It may reuse our S4
     module if the shapes match; otherwise port theirs after reading it.
   - Load the published weights. **Pass if every tensor name and shape matches with no missing or unexpected keys.**
2. **Sample about 1,000 ECGs per class** with their conditioning (label vectors taken from real records of that class).
   - Use mixed precision and a batch that fits 4 GB. Record time per batch and peak memory.
3. **Reproduction check:**
   - run detector v2 on our samples vs their published samples (same classes);
   - **Pass if the AUC is ≤ 0.60** (our sampler draws from the same distribution);
   - also require the G1 measures on our samples to be within ±0.05 AUC of theirs.
   - If it fails, the port is wrong: fix and rerun; do not continue.
4. **Training-cost pilot on the laptop:**
   - 200 training steps of the 36×256 model on our data;
   - mixed precision, gradient checkpointing, batch 2 / 4 / 8;
   - record memory, steps per second and projected hours for 100k steps.
   - **Rule:** if the projection is ≤ 36 h at batch ≥ 4, train on the laptop (overnight runs, checkpoint and resume).
     Otherwise prepare a Kaggle notebook (16 GB GPU, about 30 h a week).
     - Kaggle runs are started manually from a personal Kaggle account.
     - The notebook is self-contained, with a file of the allowed `ecg_id`s and a checksum test.

## Phase 5: train our own, leak-free SSSD-ECG (GPU, ~1–3 days wall clock)
**Why our own:** the published model saw about 80 % of our val and test patients and the whole study 4 judge part.
A realism claim on the judge part needs a generator that never saw it.
- **Training data, fixed now:** study 4 **fit** part + **half of dev** (patient-grouped, seed 4040), about 7,600
  recordings.
  - The other half of dev (**dev-holdout**, about 2,000) is the real side for all development checks.
  - The judge part (4,175) is unseen until Phase 7.
  - Val and test are not used at all.
  - **Risk, stated now:** 7.6k recordings vs the paper's ~17k. Underfitting is possible, and Phase 6 measures it.
- **Change one thing at a time.** The architecture, noise schedule, iteration count and 71-label conditioning stay
  exactly as published. Labels come from `ptbxl_database.csv` `scp_codes` of our records, with the same 71-statement
  order identified in Phase 1. **The only change is the training data.**
  - Any later change (for example the noise schedule fixed to 5e-4 → 0.1, or 4-class conditioning) is a separate,
    logged run, made only if Phase 6 fails and only after diagnosis.
- **Monitoring:**
  - checkpoints every 4k iterations, with EMA;
  - at 20k, 50k and 100k, sample 200 ECGs and run the quick checks (periodicity, HR ordering, amplitude);
  - a run that has no regular heartbeat at 50k is reported (Week 4 failure mode) before it continues.
- **Memorisation guard:** the nearest training-ECG correlation is reported at every checkpoint check.

## Phase 6: realism of our own model on dev-holdout (CPU, ~4 h)
- **Measures:** the same Phase 3 suite, real side = dev-holdout.
- **Gate G2:**
  - **pass** if all G1 PASS criteria hold, and the detector AUC is within +0.10 of the published model's Phase 3
    value;
  - **fail** otherwise. Report the gap, which with the training-size risk is the likely cause.
  - Diagnose with the bands and univariates before any second run. A second run needs a written amendment first.

## Phase 7: registered realism test on the judge part (once)
- **Before running,** write a short registration (`generator_comparison/study4/REGISTER_W.md`) that fixes:
  - the generator checkpoint;
  - the sample count and seed;
  - the measures: detector v2 and the WORKFLOW_4 D3 judge detector (a 1-D ResNet with a different receptive field)
    plus the feature judge;
  - the primary outcome: our whole-signal model vs the current hybrid generator, judge-detector AUC, paired over
    detector seeds;
  - the decision rule.
- Commit it, then run once.
- This replaces nothing in WORKFLOW_4's Phase 6, which stays available for the hybrid generator.

## Phase F (fallback if G1 fails): real-noise-bank hybrid (CPU, ~½ day)
- Keep our generated beat templates and rhythm model.
- Replace the modelled noise and variation with **real** non-template activity borrowed from fit-part ECGs of the
  same class, aligned to the borrowed ECG's own R-peaks and re-placed.
- This measures how realistic the hybrid route can get at best. The earlier cross-patient real variation already
  improved the detector from 0.94 to 0.83.
- **It gets its own written plan** (`PLAN_noisebank.md`) before any code.

## Timeline (wall clock)
| phase | compute | estimate |
|---|---|---|
| 1–3 | CPU | 1 day, ending in G1, the main go/no-go |
| 4 | GPU | ½–1 day |
| 5 | GPU | 1–3 days, laptop or Kaggle |
| 6–7 | CPU | 1 day |

## What each outcome gives the paper
- **G1 FAIL:** "Even the published, paper-scale SSSD-ECG is detectable at AUC ≥ 0.95, by a CNN and by 25 standard
  features." The failing properties are named. This is a strong negative result about the state of the art.
- **G1 PASS, G2 PASS, Phase 7:** a leak-free whole-signal generator that is realistic on a held-out judge, compared
  head to head with the hybrid. This is the realism result study 4 set out to find.
- **G1 PASS, G2 FAIL:** realism needs data scale (~17k vs ~7.6k ECGs). That is informative on its own, and it says
  what a realistic generator costs.
