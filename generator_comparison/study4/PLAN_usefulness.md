# Study 4, question U: does a more realistic residual make better training data? (plan, 2026-10-30)

**Status: plan, written before the learned residual model finished training and before any of its outputs were
seen.** Exploratory: validation split only, no test split, not pre-registered. Its job is to decide whether
realism is worth pursuing for *usefulness*, the question that gives study 4 its point.

## Why
Studies 2–3 found that synthetic data helps mostly by moving the decision threshold (THRESHOLD_EXPLORE.md). Study 4
found the current noise model is unrealistic in specific ways. It is spiky-less, too strong and beat-unlocked. It
also breaks R-peak detection: 11 % of beats missed and 20 % spurious, so 30–51 % of regular-rhythm synthetic ECGs
look as irregular as AFIB (NOTES_features.md). If a realistic residual gives synthetic data that **separates classes
better** (higher threshold-free AUPRC), realism matters for usefulness. If not, realism and usefulness are decoupled,
and that is a finding too.

## Gate before running (decided now)
Run U only if the learned residual passes the physiology checks. Each is measured with the same code as
NOTES_features.md, on the LR semi-synthetic set:
- the R-peak detection error rate (missed + spurious) is less than half of C's (31 %);
- residual kurtosis is closer to real (about 30) than C's (about 9).

If the gate fails, U is not run and the failure is reported. The detector AUC is **not** part of the gate:
usefulness can change even if the CNN still separates the two at 1.0.

## Design: two arms that differ only in the residual
- **Templates:** the study 1/2 generated templates (`pools2/pool_1.npz` `templates`). This is the same generator as
  study 2's random pool.
- **Shared parts:** the same template, the same sampled R-peaks (class rhythm models, `fixes/models.pkl`) and the
  same GP wander sample in both arms.
  - **Arm A (current):** + GP residual (`gp_residual`, fit part; the current recipe).
  - **Arm B (learned):** + learned residual (the two fold models alternate; condition = place(template, R)).
- **Counts:** the study 2 synthetic counts at ratio 1.0 (+1,056 AFIB / +568 STACH / +446 SBRAD). Each pool holds
  exactly those, index-aligned across the two arms, so every seed adds the same templates in A and B.
- **Classifier:** `generator_comparison.augment_experiment`, study 2 recipe. That is `--steps 6250 --eval-every 250
  --deterministic --val-only`, condition `random`, 10 seeds (seed-start 600). The `real` condition is the reference.
  - 3 conditions × 10 seeds = 30 runs, about 40 min of GPU.
  - Building pool B: about 2,070 samples, about 30 min of GPU.

## Outcomes (validation only; computed once after all runs)
- **Primary:** macro AUPRC, B − A, paired by seed. Wilcoxon over 10 seeds, with a bootstrap CI. Threshold-free, so it
  measures separation, not operating point.
- **Secondary:**
  - plain macro-F1 B − A;
  - tuned macro-F1, using per-class offsets on patient-split validation halves (`_threshold_explore.py`'s method,
    imported, not re-implemented);
  - per-class AUPRC, especially SBRAD.
- **Pool descriptives:** feature-judge AUC and R-peak error rate of A and B, so usefulness can be read against
  realism.
- **Interpretation, decided now:**
  - B − A AUPRC CI entirely > 0: realism helps separation.
  - CI within ±0.005: no practical effect.
  - Otherwise: inconclusive at this size.

## Leakage and fairness
- The learned residual models saw only fit-part residuals. The generated templates come from the study 1 generator,
  as in study 2.
- The validation split was never used by any generator or noise model; it is used here for checkpoint choice and
  outcome, as in studies 2–3. That shared use is a limitation shared with THRESHOLD_EXPLORE.md, and it is stated.
- Arms A and B differ only in the residual, so any difference is attributable to it.

---

# Amendment U2 (2026-10-30 06:50), written after the U gate failed and before any classifier run
**Gate outcome.** The learned residual failed the U gate: kurtosis 4–9, against C's about 9 and real about 30.
It also failed the detector (1.000). So U as planned (learned vs GP, motivated as "more realistic residual") is
**not** run.

**New question, from tonight's mechanism finding (NOTES_features.md).** Does the noise model matter for usefulness
at all? The current GP residual corrupts R-peak detection: 31 % of beats are missed or spurious, and 30–51 % of
regular-rhythm synthetic ECGs look as irregular as AFIB. The learned residual corrupts it about 3× less (10 %). A
residual-free ECG (template + wander) corrupts nothing but has no beat-to-beat variation at all. Three arms then
form a "detection-corruption" axis with otherwise identical content.

## Arms (same templates, same R-peaks, same wander; only the residual differs; pools built by `_study4_useful.py build`)
| arm | residual | R-peak corruption |
|---|---|---|
| `N` | none: `useful_pool_clean.npz` | none by construction |
| `B` | learned residual | about 10 % of beats in the semi-synthetic test |
| `A` | GP residual (current recipe) | about 31 % of beats |

Plus `real` (no synthetic data). 10 seeds (600–609), study 2 recipe, validation only. That is 40 runs, about 50 min,
plus the pool build, about 35 min.

## Outcomes (as in U; computed once after all runs)
- **Primary:** macro AUPRC for N − A and B − A, paired by seed. Wilcoxon over 10 seeds, with a bootstrap CI.
- **Secondary:**
  - plain and tuned macro-F1;
  - SBRAD and STACH AUPRC;
  - each arm − real.
- **Realism descriptives:** the R-peak error rate of each pool against the residual-free truth, and the
  feature-judge AUC.
- **Interpretation, fixed now:**
  - If N ≈ B ≈ A (CI within ±0.005 AUPRC), the noise model does not matter for usefulness, consistent with the
    threshold account.
  - If A < N or A < B with the CI excluding 0, the noise model **affects usefulness**: it hurts separation. The arms
    differ in more than R-peak corruption (residual energy, beat-to-beat variation, spectrum), so this alone does not
    show *why*.
  - **Mechanism readout, fixed now.** From the saved validation probabilities (plain argmax), count the confusions
    between AFIB and the regular-rhythm classes: AFIB predicted as SR / STACH / SBRAD, and SR / STACH / SBRAD
    predicted as AFIB. Report AFIB precision and recall too.
    - **The detection-corruption mechanism is supported only if** these confusions are ordered A > B > N (means over
      seeds).
    - Otherwise it stays a hypothesis, even if A < N.
  - If A > N, the noise acts as useful augmentation despite being unrealistic.
- Two primary comparisons; report both with unadjusted p and say so. Exploratory, validation only.
- **CI scope:** one pool, one set of templates and one rhythm draw. The bootstrap resamples training seeds only, so
  the CI covers classifier-seed noise for this pool, not pool-to-pool variation.
- **Axis check before training:** run `realism` on A, B and N, and record whether R-peak corruption on these
  generated-template pools is ordered A > B > N = 0. The arms are not changed whatever it shows. If B is not between
  A and N, the write-up drops the "axis" framing.
- The analysis runs once (guard in `stage_analyse`). This is one more exploratory use of the validation split,
  after THRESHOLD_EXPLORE.md.

## Code change needed (small, audited before running)
- `_study4_useful.py`: add condition `N` (pool = `useful_pool_clean.npz`) to `CONDS`, `stage_train` and
  `stage_analyse`.
- Add the paired comparisons N − A, N − B and N − real.
- The realism stage already evaluates A and B; add N.
