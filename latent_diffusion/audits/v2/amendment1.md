# Amendment 1: Gate A recalibration and continuation of the run (2026-10-31 ~02:10 IST)

Written after Phase A was run under `STUDY9_V2_INSTRUCTIONS.md` and the
original Gate A failed. The remaining phases were then completed overnight. **The original gate
result is not changed or hidden** (`results/v2/phaseA_gate.json`, `passed: false`). This amendment is written after
seeing the control results, so everything downstream is labelled "under amendment 1".

## Code reuse
Phases B-H reuse the Phase A code
(`v2_common`, `gen_eval_v2`, `phase_a_v2`, `encoder_v2`, `baselines_v2`) unchanged, including its own
`gpu_lease` (desktop/browser processes exempt, stale lock recovery), which replaces the patch in `NOTE_V2_GPU_LEASE.md`.

## What failed and why it is a gate-design error, not a measurement bug
- **Noise precision < 0.1.** Gaussian noise gets embedding precision 0.70-1.00 in most classes. Precision only asks
  whether a synthetic point falls inside *some* real k-NN ball; a tight noise cluster landing inside a few large
  balls passes. This is the known weakness of k-NN precision (Naeem et al. 2020). Density is not robust here either
  (noise density up to 2.62). The gate demanded something these metrics cannot deliver.
- **Collapse coverage < 0.1 at SBRAD n = 63.** A single repeated ECG can cover about (k+1)/n of the reals by
  construction (6/63 ≈ 0.095, observed 0.127 / 0.111). A fixed 0.1 threshold is impossible at small n.
- **emb_ssl is noise-blind.** It was trained with Gaussian-noise augmentation (as flagged before running), and
  real STACH ECGs are 29 % "covered" by pure noise in that space.

## Amended Gate A (applied to the existing numbers, no recomputation): `gate_a_amended_v2.py`
1. Floor clinical judge AUC 0.40-0.60, SD ≤ 0.06 (unchanged).
2. Noise clinical judge AUC > 0.95 (unchanged).
3. Floor emb_task recall and coverage ≥ 0.85 per class.
4. Noise and collapse **recall and coverage** in emb_task < max(0.1, 2(k+1)/n) per class (k = 5).
5. Copy controls and bootstrap sanity unchanged.
Embedding precision and density, and everything in emb_ssl, are **descriptive only** from now on.

## Consequences for later gates (fixed now, before any B-G result)
- Gate C metric 4 "coverage in emb_ssl" → **coverage in emb_task**. emb_ssl coverage is still reported.
- Gate E diversity rule → computed on **emb_task coverage**.
- Phase F headline rule → "not worse on coverage" uses **emb_task coverage**.
- Fidelity claims rest on the clinical feature judges (full and the four groups) and per-feature Wasserstein.
- Percentile PRDC intervals sometimes exclude their own point (8/32 floor intervals). Superiority
  decisions in F therefore use the clinical-judge rule plus non-overlap of feature-Wasserstein intervals, not PRDC
  intervals.
- Judge "CI" for Phase F (fixed now): mean ± 2 SD over the 5 grouped splits, labelled "split range", not a CI.

## Other fixed decisions for B-G (recorded before running them)
- **Scope under one night of compute.** Priority order from the instructions: A → B1 → C-S → D on C-S → E (reduced)
  → F → G. C-M is trained only if time remains after D; B2 only if B1 fails.
- **Step budget.** C-S and the D model get the same budget, decided from the measured step time so that C-S + D fit in
  ≤ 7 h of GPU in total; at least 40 k steps each. The actual number is written into `audits/v2/phaseC.md` before training.
- **Phase E reduced grid** (fixed now): CFG {1, 2, 3} × DDIM steps {50, 100}, η = 0 only, on the main generator, 63/class.
- **Missing HR during training sampling:** rows with missing HR (0.6 %) get the median HR-bin weight.
- **R-map dropout (Phase D):** independent: map dropped p = 0.1, vector dropped p = 0.1; additionally with p = 0.05 both
  dropped. Guidance uses both-present vs both-absent.

# Amendment 2: Phase F/G scope (written 2026-10-31 ~03:00 IST, before C/D training finished and before any F/G run)
- **F reference:** all validation AFIB/STACH/SBRAD (224/125/96) + 256 SR (instructions said SR cap 512). Every
  frozen v1 pool has 256 per class, so 256 keeps all pools at matched counts. Floor = two disjoint validation patient
  halves at feasible matched counts.
- **F second training seed:** not run (time). Three sampling seeds of the main generator are scored with the judge
  only and reported as sampling variance.
- **F pools:** hybrid = `generator_comparison/tstr/pool_beatgen.npz` (440/class); hybrid privacy is valid on validation
  (val patients were never in its training), so it is included in F.
- **F `latent_v2_rhythm_model`** uses the hybrid's own fitted RhythmModel (`beat_templates/assembly_models.pkl`,
  fitted on the full train split) with `MEAN_RR_BOUNDS`; the condition HR is set from the requested rhythm.
- **G hybrid pools:** minorities from study 2 `pools2/pool_1.npz` (bounded RR, same frozen generator); hybrid SR
  generated fresh with the frozen `_beat_gen_pool.py` (seed 9800) into Study 9 artifacts, after GPU training ends.
- **G tuned F1:** offsets fitted on the dev split's softmax with the copied `_threshold_explore.tune`; checkpoint
  selection also on dev (plain-argmax macro-F1, as `train_steps` does). Validation is scored once per run.

# Amendment 3: longer training (written 2026-10-31 14:53 IST, before any training past 60 k)
Longer training was requested after seeing the 60 k results. **The 60 k results stay the primary, pre-registered
result.** Everything below is a **second look**: dev was seen at 60 k, and validation was used once in F and once in G.
- **Training:** resume `cs` and `ds` from their 60 k checkpoints to **120 k** steps, with everything else
  unchanged (constant LR 2e-4, same sampler, same RNG stream from the checkpoint). Snapshots at 90 k and 120 k.
  Registrations: `phaseCD_{cs,ds}_ext120000`. The 60 k results are kept as `results/v2/*_60k.json`, and the
  checkpoints as `step60000.pt`.
- **Dev:**
  - Gate C metrics at full dev counts, sampled at CFG 2 / DDIM 50 (no new sampler grid).
  - Rhythm sweep and Gate D re-evaluated exactly as written, and labelled a second attempt.
- **Validation second look:**
  - Score the 120 k `cs` pool and the 120 k `ds` rr_bank pool against the same reference, floor and hybrid at the
    same counts, with the same Phase F rule.
  - Utility: rerun only the `latent_v2` and `latent_v2_plus_sr` arms with fresh 120 k pools, using the same seeds,
    pool seeds and recipe. Compare them with the existing `real` and `hybrid` runs, which are deterministic.
- **Reporting:** any "better" verdict at 120 k is labelled "second look, not pre-registered at 60 k".

# Amendment 4: C-M capacity rung on Google Colab (written 2026-10-31 15:28 IST, before any C-M training)
A larger model trained on Colab was approved to improve realism. C-M is the instructions' own second rung (§C2), deferred in
amendment 1 for time.
- **Model:** `cm` = 16 layers × width 128, S4 state 64 (5.87 M params), no R-map. Everything else is identical to
  `cs`: same AE r1 latents, sampler, loss, LR, EMA, seed 9200, batch 32, and dev diagnostics.
- **Steps:** 100 k (§C2), or as many as fit in 12 h of GPU if the measured step time says 100 k would exceed that.
  The measured time is recorded in `audits/v2/phaseC_cm.md` before the main run.
- **Hardware:** Colab GPU (type recorded from `nvidia-smi`). The S4 Cauchy kernel may use pykeops there, which
  differs from the laptop at float precision. Bundle: `colab/ecg_colab_bundle.zip` (train split only).
- **Evaluation (on return):** dev Gate C metrics for `cm` vs `cs` at the same counts and sampler (CFG 2 / DDIM 50).
  C-M "beats" C-S only if its full judge is lower by more than the larger of the two SDs (§C2 "by more than the CI").
  If it does, it is also scored on validation as a second look under amendment 3's rules.
- **Code change:** `gpu_lease` gained a Linux branch (`ps` instead of `tasklist`). Windows behaviour is unchanged.

# Amendment 5: 120 k dev evaluation mechanics + ds guidance check (written 2026-10-31 20:35 IST, before any 120 k sample)
- **Cache isolation:** `dev_eval_v2` caches results by tag (`results/v2/dev_<tag>.json` and
  `artifacts/metric_cache/dev_<tag>.npz`), so re-running it would silently return the 60 k numbers. The 120 k
  evaluation therefore runs from a new module, `eval_120k_v2.py`, with `_120k` tags and outputs in
  `results/v2/ext120k/`.
  - Same requests, seeds, counts, CFG 2 / DDIM 50 and replicate counts as at 60 k, so the comparison is paired.
  - The sweep and Gate D logic are reused unchanged.
- **Snapshots:** `dev_eval_v2 snapshots` adds the 90 k and 120 k checkpoints. Earlier snapshots come from the
  unchanged `step*.pt` files and reuse their cached scores.
- **ds guidance check (descriptive):** for `ds` at 120 k, CFG {1.0, 1.5, 2.0} × DDIM 50, 63 per class, seed 9400 (the
  Phase E seed).
  - Reported: full / rhythm / morphology / global judges, QRS template amplitude ratio (`tpl_p2p` median, synthetic
    / real), missed R-peaks, and HR-W.
  - It is **not** used to change Gate D, which is evaluated at CFG 2 exactly as written.
  - It informs `GENERATOR_CHOICE.json` (`rmap_guidance`) for the controllable-generation project, whose own Phase 3
    re-tunes on its own dev grid.

# Amendment 6: S4 kernel mismatch for Colab-trained C-M (written 2026-10-31 22:31 IST, after the first, invalid C-M dev scoring)
- **Finding:** `synthetic_generation/s4.py` (frozen) has three Cauchy-kernel paths.
  - The CUDA extension (`symmetric=True`) and pykeops (`cauchy_conj`) sum over conjugate pairs: factor 2 and
    `Conj(w)`.
  - The fallback `cauchy_slow`, used on the laptop because neither is installed, sums only the stored half.
  - The same weights therefore give different networks on different machines.
- **Evidence** (dev diagnostic loss recomputed locally, identical cohort and noise):

  | checkpoint | fallback | conjugate-pair (fallback patched in memory) | logged in training |
  |---|---|---|---|
  | `cm` 25 k (trained with pykeops) | 0.662 | **0.4193** | 0.4193 |
  | `cs` 120 k (trained with the fallback) | **0.3938** | 1.497 | 0.3938 |
- **Consequences:**
  - All laptop-trained models (`cs`, `ds`, earlier SSSD runs) are internally consistent: trained and evaluated
    with the fallback. Their results stand. They are, however, a non-standard S4 variant (half conjugate sum), which
    is a limitation to report.
  - The first C-M dev scoring (full judge 0.988) used the wrong kernel and is **invalid**. Its files are renamed
    `INVALID_halfkernel_*` (results and metric cache) and are kept, not deleted.
  - C-M is re-scored with the fallback patched **in memory** to the conjugate-pair sum, via
    `s4_kernel.conj_kernel()`. `s4.py` is not edited.
  - The overfitting diagnosis for C-M stands: its train and dev losses were computed in training with one consistent
    kernel (pykeops).
- **Guard:** any future run must use the kernel its checkpoint was trained with. Laptop checkpoints must not be
  evaluated where pykeops or the CUDA extension is installed, and Colab/pykeops checkpoints need the conjugate
  kernel locally.

# Amendment 7: D-M rung (C-M size + R-peak map) on Kaggle (written 2026-11-01 13:58 IST, before any D-M training)
- **Model:** `dm` = 16 layers × width 128, S4 state 64, **with** R-map input (`--rhythm`). Everything else is identical
  to `ds` (R-map dropout, sampler, LR, EMA, seed 9200, batch 32). The conjugate-pair kernel (pykeops) is used on Kaggle.
- **Steps:** 120 k, the same as `ds`'s final budget.
- **Evaluation** (dev, before any validation use):
  - snapshot protocol (63 per class, seed 9500) at every saved checkpoint, with the conjugate kernel;
  - the ds guidance check at CFG {1.0, 1.5, 2.0};
  - the HR sweep and Gate D as written.
- **Selection:** `dm` replaces `ds` as the R-map model for the controllable-ECG project if its best dev checkpoint and
  guidance has a full judge lower than `ds` 120 k at CFG 1.0 (0.861) by more than the larger SD. The QRS amplitude
  ratio must also stay within 0.8–1.25, and missed R-peaks must be ≤ 5 %.
- **Lesson from C-M:** checkpoints are chosen by dev **realism**, never by dev loss.

# Amendment 8: validation second look for the 120 k / C-M generators (written 2026-11-14 17:21 IST, before any second-look sample exists)
This implements the validation second look that amendments 3 and 4 pre-specified. It is run **once**, so that the
submitted report's headline numbers are on validation.
- **Status label:** "second look". Validation was used once before, in Phase F/G. The checkpoints were selected on dev:
  - `cs` and `ds` at 120 k (amendment 3);
  - `cm` at step 100 k, the best dev checkpoint out of 25k/50k/54k/75k/100k (amendment 4; `results/v2/ext120k/cm_checkpoints.json`).
- **Reference, floor and fit reference:** **identical** to Phase F: the cached `val_real` and `val_fit_reference`, and
  the same floor halves. The counts are 256/224/125/96, and the hybrid result is `F_hybrid.json`, unchanged.
- **Pools:** the same requests as Phase F (`fit_requests(VAL_COUNTS, 9600)`), the Phase E sampler (CFG 2, DDIM 50,
  η 0), seed 9620, EMA 0.999.
  - `F2_cs120k`: `cs` at 120 k (fallback kernel).
  - `F2_ds120k_bank`: `ds` at 120 k with the request's rr_bank R-peaks (fallback kernel).
  - `F2_cm100k`: `cm` step 100000 (**conjugate kernel**, amendment 6).
  - Three extra sampling seeds for `cm` (9630–9632), judge only, as in Phase F.
- **Decision rule:** Phase F's `decide()`, unchanged, applied to each pool against the hybrid:
  - **better** iff judge (mean + 2 SD) < hybrid (mean − 2 SD), **and** emb_task coverage is not worse in ≥ 3/4
    classes, **and** HR-W is not worse in ≥ 3/4 classes;
  - **worse** iff hybrid (mean + 2 SD) < pool (mean − 2 SD);
  - otherwise **not distinguishable**.

  Privacy is computed as in Phase F.
- **Primary result for the report:** `F2_cm100k` vs the hybrid. The other pools are secondary.
- **Outputs:** `results/v2/F2_*.json` and `results/v2/benchmark_val_second_look.json`. Nothing earlier is overwritten,
  and no test-split file is used.
