# Study 9 v2: whole-ECG latent diffusion: report

Written 2026-10-31 (IST). Phases B–H ran under [amendment 1 and 2](audits/v2/amendment1.md), after Phase A. v1 stays in
[REPORT.md](REPORT.md). Everything after Phase A is **under amendment 1**: the original Gate A failed
([phaseA_gate.json](results/v2/phaseA_gate.json)), and the gate was recalibrated after seeing control results.

## 1. Answer

**On validation, the v2 latent diffusion model (`latent_v2`) is not distinguishable from the hybrid on realism by the
fixed Phase F rule.**

- **Realism:**
  - The full clinical judge AUC is 0.961 ± 0.002 SD for `latent_v2` and 0.965 ± 0.005 for the hybrid. The intervals overlap.
  - Both are far from real data (real-vs-real floor 0.454) and from the autoencoder ceiling (0.627).
  - So no pool earns the word "realistic". All three v1 learned or simulated generators score 1.000.
- **Rhythm control (Gate D): negative under the fixed rule.**
  - The R-peak-map model follows the requested heart rate in its medians.
  - It misses the linear-fit slope gate in SR (0.87) and STACH (0.87).
  - Its morphology is worse than the unconditioned model's.
- **Utility (exploratory):**
  - Every augmentation arm raises untuned macro-F1. That includes the v2 generator, the hybrid and plain duplication of real minority ECGs.
  - None raises macro AUPRC or tuned macro-F1. This replicates studies 2–4: **the gain is threshold-only**.

## 2. What changed from v1 (D1–D5 of the instructions)

- **D1 (AE ceiling):** reconstructions are now judged. The v1 AE failed (0.718). Training the same architecture for 60
  epochs passed (0.625).
- **D2 (capacity and training):** 8 layers × width 64, class-balanced and HR-stratified sampling, v-prediction, EMA,
  60 k steps. That is about 164 epochs, against about 7 for v1.
- **D3 (rhythm):** an R-peak map (Gaussian bumps plus a validity flag) at latent resolution, with independent dropout of
  the map and the vector condition.
- **D4 (measurement):** the Phase A scorecard: floor, grouped clinical judges, patient-cluster bootstrap, two embeddings,
  and privacy controls.
- **D5 (utility):** the studies 2–3 recipe, 5 seeds, dev-selected and dev-tuned, scored on validation. It adds synthetic-SR
  shortcut arms.

## 3. Phase A: what the scorecard can and cannot separate

The floor clinical judge is AUC 0.443 ± 0.015 on dev and 0.454 ± 0.040 on validation (two disjoint patient halves).
Gaussian noise scores 1.000. ([phaseA_gate_amended.json](results/v2/phaseA_gate_amended.json),
[F_floor.json](results/v2/F_floor.json).)

- **What fails as a gate:**
  - Embedding precision and density accept a tight noise cluster: noise precision is 0.70–1.00.
  - The SSL embedding is noise-blind: it was trained with noise augmentation.
  - Both are descriptive only from now on.
- **What carries the claims:** fidelity rests on the clinical judges (the full judge and four feature groups). Diversity
  rests on coverage in the task embedding. Collapse and noise controls stay below max(0.1, 2(k+1)/n) there.
- **Resolution:** at n = 63–256 per class, judge AUCs above about 0.95 are hard to separate. The five-split SD is
  0.002–0.005, but differences between generators are of the same size.

## 4. Phase B: the AE ceiling

**The best a latent model on this AE can do is a judge AUC of about 0.62–0.63.** That is 0.625 on dev and 0.627 on
validation. ([ae_ladder.json](results/v2/ae_ladder.json), [ae_ceiling_ae_r1.json](results/v2/ae_ceiling_ae_r1.json),
[F_ae_recon.json](results/v2/F_ae_recon.json).)

| AE | full judge | noise_wander group | missed R | Gate B1 (≤ 0.65 / groups ≤ 0.75 / ≤ 3 %) |
|---|---|---|---|---|
| v1 AE | 0.718 ± 0.025 | 0.724 | 1.5 % | fail |
| r1 (60 epochs) | 0.625 ± 0.025 | 0.621 | 0.9 % | **pass** |

What still gives reconstructions away is smoothed baseline wander (`wander_edge_inc_ratio`, `wander_rms`). Details are
in [audits/v2/phaseB.md](audits/v2/phaseB.md). The main generator is at 0.961 against this 0.627 ceiling, so **the
remaining gap is in the diffusion prior, not the AE.**

## 5. Phase C: diffusion model and Gate C (dev)

Only one capacity rung (C-S, 8×64) was trained, because of the one-night budget. The step budget of 60 k steps (about
131 min on an RTX 3050 4 GB, 560 MiB peak) was written before training in [phaseC.md](audits/v2/phaseC.md).

**Gate C** (dev, full dev counts 256/170/82/63, CFG 2, DDIM 50): the rule is a full judge below the hybrid's, **or**
HR-Wasserstein ≤ the hybrid's in ≥ 3 classes. ([gate_CD.json](results/v2/gate_CD.json).)

| dev pool (n = 571) | full judge | rhythm judge | mean HR-W (bpm) | mean emb_task coverage | rhythm failure |
|---|---|---|---|---|---|
| hybrid | 0.970 | 0.720 | 6.81 | 0.758 | 0.041 |
| latent v1 | 1.000 | 0.982 | 23.3 | 0.518 | 0.617 |
| **C-S (`cs`)** | **0.967** | 0.841 | 6.22 | 0.962 | 0.119 |
| D (`ds`, R-map) | 0.983 | 0.624 | 4.33 | 0.872 | 0.047 |

- **Result:** Gate C passed on the point estimate. The margin is 0.003, smaller than the judge SD. The Phase F rule
  with SDs is the real test.
- **v1 versus v2:** v1 → v2 is a large improvement on every metric (judge 1.000 → 0.967, HR-W 23 → 6 bpm, coverage
  0.52 → 0.96).
- **Still training at 60 k.** Judge on 63 per class across snapshots
  ([snapshot_curves.json](results/v2/snapshot_curves.json)):

| step | cs full judge | cs rhythm failure | ds full judge | ds rhythm judge |
|---|---|---|---|---|
| 15 k | 0.996 | 0.341 | 0.994 | 0.809 |
| 30 k | 0.989 | 0.202 | 0.982 | 0.677 |
| 45 k | 0.969 | 0.119 | 0.966 | 0.617 |
| 60 k | 0.947 | 0.095 | 0.958 | 0.576 |

Dev loss was still falling at 60 k: cs 0.4117, ds 0.3662
([diffusion_training_cs.json](results/v2/diffusion_training_cs.json), [_ds](results/v2/diffusion_training_ds.json)).
These snapshot judges are at n = 252 and are not comparable to the n = 571 gate numbers. Figure: [training_curves.png](results/v2/training_curves.png).

## 6. Phase D: R-peak-map conditioning: negative under the fixed rule

**Gate D:** with `rr_bank` maps and CFG 2, all four conditions must hold:
- missed beats ≤ 5 % and extra beats ≤ 5 %;
- sweep slope 0.9–1.1 with R² ≥ 0.9 in SR, STACH and SBRAD;
- rhythm judge ≤ 0.70;
- full judge ≤ C + 0.02.

([rhythm_control.json](results/v2/rhythm_control.json).)

| condition | value | pass |
|---|---|---|
| missed R-peaks (50 ms) | 4.5 % | yes |
| extra peaks per beat | 0.19 % | yes |
| sweep slope / R², SR | 0.868 / 0.908 | **no** |
| sweep slope / R², STACH | 0.873 / 0.894 | **no** |
| sweep slope / R², SBRAD | 0.975 / 0.989 | yes |
| rhythm judge | 0.624 | yes |
| full judge vs cs + 0.02 | 0.983 ≤ 0.987 | yes |

**Gate D failed.** The Gate C model (`cs`) therefore stays the main generator, as the protocol requires.

- **Sweep, descriptive:**
  - Per-HR medians of the R-map model equal the requested HR from 40 to 110 bpm in SR, and from 40 to 150 bpm in STACH
    and SBRAD.
  - SR medians fall short at 130–150 bpm (124, 140), where missed beats rise (14.6 % at 150).
  - In STACH the slope is pulled below 0.9 by individual samples with dropped beats, not by the medians.
  - Figure: [rhythm_sweep.png](results/v2/rhythm_sweep.png).
  - The model without a map barely follows the HR condition: slope SR 0.53, STACH 0.64, SBRAD 0.15.
  - The v1 model was not included in the sweep.
- **What the R-map costs:** morphology and global judges get worse (validation 0.966 / 0.918 vs 0.888 / 0.774 for
  `cs`). The QRS template amplitude is too large: `tpl_p2p` median 8.1 vs real 5.2 in SR.
- **Not done:** the `paired_dev` R-map source (D3 #1) and the HR-error median and P90 (D4) were not run.

## 7. Phase E: sampler (dev, 63 per class, `cs`)

The grid was CFG {1, 2, 3} × DDIM {50, 100} with η = 0. The rule is the mean rank over the Gate C metrics, among
settings with coverage ≥ 0.8 × best. ([sampler_sweep.json](results/v2/sampler_sweep.json),
[sampling.json](artifacts/v2/sampling.json).)

| setting | full judge | emb_task coverage | rhythm judge | HR-W | rank sum |
|---|---|---|---|---|---|
| CFG 1, 50 | 0.959 | 0.964 | 0.816 | 7.98 | 19 |
| CFG 1, 100 | 0.960 | 0.964 | 0.801 | 7.61 | 20 |
| **CFG 2, 50** | 0.936 | **0.984** | **0.760** | 7.50 | **4** |
| CFG 2, 100 | 0.938 | 0.968 | 0.772 | 7.60 | 12 |
| CFG 3, 50 | 0.929 | 0.964 | 0.765 | 7.47 | 13 |
| CFG 3, 100 | 0.928 | 0.976 | 0.757 | 7.60 | 7 |

**Chosen: CFG 2, DDIM 50, η 0, EMA 0.999, last checkpoint.** Figure: [sampler_heatmap.png](results/v2/sampler_heatmap.png). All settings are eligible, so diversity did not bind.
CFG ≥ 2 beats CFG 1, while 100 steps add nothing.

## 8. Phase F: validation benchmark

**Reference:** 256 SR plus all validation AFIB/STACH/SBRAD (224/125/96). The floor is two disjoint validation patient
halves at 125/99/55/47. Judge values are mean ± SD over five grouped splits ("split range" = ± 2 SD, not a CI).
([benchmark_val.json](results/v2/benchmark_val.json), [benchmark_val.csv](results/v2/benchmark_val.csv),
[BENCHMARK_V2.html](BENCHMARK_V2.html).)

| pool | full judge | rhythm | morphology | noise/wander | global | mean HR-W | emb_task cov | rhythm failure |
|---|---|---|---|---|---|---|---|---|
| real floor | 0.454 ± 0.040 | 0.495 | 0.522 | 0.448 | 0.484 | 2.70 | 0.952 | 0.015 |
| AE recon (ceiling) | 0.627 ± 0.018 | 0.502 | 0.518 | 0.624 | 0.518 | 0.61 | 1.000 | 0.017 |
| hybrid | 0.965 ± 0.005 | 0.735 | 0.882 | 0.959 | 0.766 | 8.35 | 0.750 | 0.022 |
| latent v1 | 1.000 | 0.979 | 0.983 | 0.993 | 0.962 | 24.3 | 0.520 | 0.684 |
| VAE v1 | 1.000 | 0.993 | 0.997 | 0.999 | 0.999 | 19.3 | 0.026 | 0.329 |
| ECGSYN | 1.000 | 0.973 | 1.000 | 0.999 | 1.000 | 27.7 | 0.146 | 0.003 |
| **latent_v2 (cs)** | **0.961 ± 0.002** | 0.834 | 0.888 | 0.893 | 0.774 | 5.07 | **0.943** | 0.117 |
| latent_v2 R-map, rr_bank | 0.980 ± 0.005 | 0.645 | 0.966 | 0.935 | 0.918 | 2.87 | 0.819 | 0.033 |
| latent_v2 R-map, hybrid rhythm model | 0.979 ± 0.005 | 0.653 | 0.964 | 0.932 | 0.922 | 2.97 | 0.836 | 0.029 |

R-map pools: missed R-peaks are 4.8 % (rr_bank) and 4.3 % (rhythm model).

**Headline rule.** "Better" needs both of these:
- the judge split range entirely below the hybrid's;
- not worse on emb_task coverage and on HR-W in ≥ 3 of 4 classes.

`latent_v2` is not worse on coverage in 4 of 4 classes and on HR-W in 3 of 4. But its judge range [0.958, 0.965]
overlaps the hybrid's [0.955, 0.976]. **Verdict: not distinguishable.** Three sampling seeds of the main generator give
0.959–0.962 ([benchmark_val.json](results/v2/benchmark_val.json) `main_sampling_seeds`), so sampling variance is smaller
than the gap to the hybrid's upper range.

**What still gives each one away** (top permutation importances,
[F_latent_v2.json](results/v2/F_latent_v2.json), [F_hybrid.json](results/v2/F_hybrid.json)):

- **`latent_v2`:**
  - `beat_corr_median` (AUC drop 0.12): beats are less alike than real beats (SR median 0.91 vs 0.97).
  - Baseline wander that is too small (`wander_rms` 0.21 vs 0.25).
  - 11 % of its SR ECGs have an RR gap over 2 s, against under 1 % of real SR ECGs.
- **Hybrid:**
  - `res_kurtosis` (0.135): residual noise is too Gaussian (median 9.2 vs real 32.5).
  - Beats are too variable (0.82).
  - Wander is too large (0.40).
- **R-map model:** `tpl_p2p` (0.12), meaning QRS amplitude is too large.

The two generators fail in different places. Neither is close to the AE ceiling.

**Privacy** (limited nearest-neighbour attack, no guarantee; fit-patient reference):
- Exact copies are 0 for every pool.
- The membership AUC (are fit-reference ECGs nearer to synthetic samples than held-out validation ECGs are?) is
  0.46–0.55 for every pool and class. All 95 % intervals contain 0.5.
- In raw waveform space the membership AUC hardly varies between pools (SR 0.53–0.54 for every pool, including latent
  v1, which the judge separates perfectly), so that space does not discriminate. The "no member-specific signal" claim
  rests on `emb_task` only (0.46–0.55, every CI containing 0.5).
- For `cs`, the ratio of synthetic-to-train to real-to-train nearest distance is 0.86–0.97, below 1. With membership at
  chance, this is consistent with samples concentrating in dense, typical regions (CFG 2) rather than member-specific
  copying.
- Nothing here shows the model is private.

**Published SSSD-ECG:** not scored. `Dataset.zip` (929.87 MB, https://figshare.com/s/43df16e4a50e4dd0a0c5, MIT) needs a
manual browser download into `external/sssd_ecg/dataset/`. No download was attempted.

## 9. Phase G: utility (exploratory; validation used once)

**Recipe:** fit patients for training, dev for checkpoint selection and threshold offsets, 6,250 steps, 5 seeds
(9100–9104). Minority classes are filled to 2× the fit count. `*_plus_sr` arms add synthetic SR equal to 20 % of real SR.
Values are mean ± SD over seeds. Differences vs `real` use a 2-level bootstrap (val patients, seeds; 300 replicates)
with 95 % CIs. All of it is **exploratory**. ([utility_G.json](results/v2/utility_G.json),
[utility_G_summary.json](results/v2/utility_G_summary.json), [utility_table.md](results/v2/utility_table.md).)

| arm | untuned macro-F1 | macro AUPRC | tuned macro-F1 |
|---|---|---|---|
| real | 0.786 ± 0.006 | 0.883 ± 0.005 | 0.845 ± 0.009 |
| oversample | 0.809 ± 0.008 | 0.881 ± 0.008 | 0.838 ± 0.005 |
| hybrid | 0.817 ± 0.012 | 0.879 ± 0.005 | 0.836 ± 0.007 |
| latent_v2 | 0.803 ± 0.011 | 0.877 ± 0.008 | 0.832 ± 0.007 |
| hybrid_plus_sr | 0.792 ± 0.010 | 0.888 ± 0.009 | 0.848 ± 0.013 |
| latent_v2_plus_sr | 0.792 ± 0.020 | 0.892 ± 0.006 | 0.847 ± 0.010 |
| *latent_v2_rhythm_model (out of protocol)* | 0.805 ± 0.006 | 0.880 ± 0.004 | 0.828 ± 0.006 |

| contrast (exploratory) | untuned F1 | macro AUPRC | tuned F1 |
|---|---|---|---|
| oversample − real | +0.023 [+0.009, +0.037] | −0.002 [−0.014, +0.013] | −0.007 [−0.020, +0.007] |
| hybrid − real | +0.031 [+0.017, +0.044] | −0.004 [−0.016, +0.009] | −0.009 [−0.025, +0.007] |
| latent_v2 − real | +0.017 [+0.001, +0.034] | −0.006 [−0.019, +0.007] | −0.013 [−0.028, +0.004] |
| latent_v2 − hybrid | −0.013 [−0.033, +0.006] | −0.003 [−0.019, +0.012] | −0.003 [−0.025, +0.018] |
| hybrid_plus_sr − hybrid | −0.025 [−0.044, −0.009] | +0.009 [−0.002, +0.020] | +0.012 [−0.003, +0.028] |
| latent_v2_plus_sr − latent_v2 | −0.012 [−0.027, +0.004] | +0.015 [+0.003, +0.026] | +0.014 [+0.000, +0.031] |
| hybrid_plus_sr − real | +0.006 [−0.009, +0.019] | +0.006 [−0.009, +0.021] | +0.003 [−0.019, +0.024] |
| latent_v2_plus_sr − real | +0.006 [−0.012, +0.024] | +0.009 [−0.003, +0.023] | +0.002 [−0.016, +0.022] |

**Answers to G5, in the order the instructions set:**

1. **Does v2 give at least the hybrid's untuned gain?**
   - The v2 gain is +0.017 [+0.001, +0.034], the hybrid's +0.031 [+0.017, +0.044].
   - Their difference, −0.013 [−0.033, +0.006], is not distinguishable from zero.
   - Plain oversampling gives +0.023, in the same range.
2. **Is either gain present in AUPRC or tuned macro-F1?**
   - No. For oversample, hybrid and latent_v2, every AUPRC and tuned-F1 difference vs `real` has a CI that includes 0,
     with slightly negative point estimates.
   - The `*_plus_sr` arms are slightly positive and also null.
   - The only CI that excludes 0 is a **loss**: the out-of-protocol `latent_v2_rhythm_model` arm, tuned F1 −0.017
     [−0.033, −0.000].
   - This replicates studies 2–4: the gain is a threshold or prior shift, not better ranking.
3. **Does more realism translate into more utility?**
   - Realism of `latent_v2` and the hybrid is not distinguishable (Phase F), and neither is their utility. Neither
     beats duplication.
   - The F hybrid pool (`tstr/pool_beatgen.npz`) and the G hybrid pool (`pools2/pool_1.npz`) are different files from
     the same frozen generator.
   - **No evidence either way at this resolution.**
4. **Does adding synthetic SR remove part of the gain?**
   - Untuned F1 falls when synthetic SR is added (hybrid −0.025, CI excludes 0).
   - AUPRC and tuned F1 move up, and no `*_plus_sr − real` contrast differs from 0.
   - Adding SR also changes the class prior the untuned argmax sees. A prior shift and removal of a "synthetic ⇒
     minority" shortcut predict the same pattern, so this design cannot tell them apart.
   - The +0.000 lower bound on `latent_v2_plus_sr − latent_v2` tuned F1 is one of 27 uncorrected exploratory contrasts
     and is not claimed.

**Out of protocol:** the `latent_v2_rhythm_model` arm should only exist if Gate D passed. The script included it because
the checkpoint existed. It is reported in italics and kept out of the four answers.

## 10. Limitations

- **Seeds and capacity:**
  - One training seed per model; the second training seed was dropped (amendment 2).
  - Only one capacity rung; C-M and C-L were not trained.
  - Both models were still improving at 60 k steps, so the result is a lower bound for this architecture.
- **Data reuse:**
  - Validation has been reused across studies 1–5 and v1.
  - The hybrid was trained on all train patients, including dev, so its dev Gate C numbers are slightly
    optimistic for it.
- **Measurement:**
  - HR and R-peaks come from a heuristic detector, and the R-map training targets come from that same detector (98.7 % plausible).
  - QT at 100 Hz is coarse.
  - The SSL encoder is small and noise-blind.
  - Judge AUCs above 0.95 are close to saturation at these sample sizes.
- **Not done:**
  - Published SSSD-ECG was not scored.
  - No clinical reader study.
  - Phase D `paired_dev` source and HR-error percentiles were not run.
  - The v1 model is not in the rhythm sweep.
- **Changed after seeing results:** Gate A was recalibrated after its controls were seen (amendment 1). Every later
  gate was fixed before its own run.

## 11. Commands, seeds, compute

All commands are run from the repository root with Python 3.14.2, torch 2.11.0+cu128, on an RTX 3050 Laptop (4 GB).

| step | command | seed(s) | wall time | peak GPU |
|---|---|---|---|---|
| Gate A (amended) | `py -3.14 -m latent_diffusion.gate_a_amended_v2` | — | CPU, < 1 min | — |
| B1 v1 AE / ladder r1 | `py -3.14 -m latent_diffusion.phase_b_v2`, `py -3.14 -m latent_diffusion.ae_ladder_v2` | 9200 | r1 training 224 s | 92 MiB |
| C (cs) | `py -3.14 -m latent_diffusion.train_diffusion_v2 --name cs --steps 60000` | 9200 | 131 min | 560 MiB |
| D (ds) | `py -3.14 -m latent_diffusion.train_diffusion_v2 --name ds --steps 60000 --rhythm` | 9200 | 132 min | 562 MiB |
| Gates C/D, sweep, E grid | `py -3.14 -m latent_diffusion.dev_eval_v2 gates`, `… sweep`, `… grid` | sampling seeds fixed in `dev_eval_v2.py` (SEED 9200 + offsets) | 2 / 4 / 3 min | — |
| F | `py -3.14 -m latent_diffusion.phase_f_v2` | main seeds 9630–9632 | 9 min | — |
| G pools / train | `py -3.14 -m latent_diffusion.phase_g_v2 pools`, `… train` | pools 9800/9810/9820; classifiers 9100–9104 | 9 / 30 min | — |
| snapshots | `py -3.14 -m latent_diffusion.dev_eval_v2 snapshots` | — | 3 min | — |

`bash latent_diffusion/run_rest_v2.sh` runs everything after training, logging to `artifacts/v2/logs/`. GPU leases and
`nvidia-smi` snapshots are in `audits/v2/phase*_nvidia_smi.txt`. Registrations (protocol written before each run) are in
`audits/v2/*_registration.json`.

**Prior work:** none of this is novel. Whole-ECG diffusion exists (SSSD-ECG, DiffuSETS, SE-Diff), and Pöhl et al. 2025
(arXiv 2505.18189) combined diffusion with rhythm assembly.
