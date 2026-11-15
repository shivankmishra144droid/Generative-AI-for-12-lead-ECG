# Workflow for study 3 (data-scarcity curve)

**Question.** How much does synthetic minority-class data help when real minority data is scarce? At 10 %, 25 %, 50 %
and 100 % of the real AFIB / STACH / SBRAD training ECGs, compare real-only, oversampling and synthetic augmentation.

Same layout as `WORKFLOW_2.md`: **who does what, in which order, and what has to be true before moving on.** The design
is fixed in `report/PREREGISTRATION_3.md` (written in Phase 0, registered in Phase 4).

## The one thing that makes or breaks this study: no leakage from the removed data

The frozen generator (`tpl_8x64_last.pt`), the template set and `assembly_models.pkl` were fitted on **100 %** of the
real train data. Used at the 10 % level, they would generate ECGs that carry information from the 90 % we claim to have
removed. "Synthetic beats real-only at 10 %" would then partly measure that leak.

So **at every level below 100 %, the template generator is retrained and the assembly models refitted on that subset
only.** Only the 100 % level may use the frozen generator. That makes this study about a day of GPU time, not the 3–4 h
first estimated:

| step | count | each | total |
|---|---|---|---|
| template generator (15k steps) | 3 levels × 3 draws = 9 | ~30 min | ~4.5 h |
| assembly models + pool | 9 + 3 (100 % level) | ~8 min | ~1.5 h |
| classifier runs (6,250 steps) | 4 levels × 3 conditions × 15 units = 180 | ~57 s | ~3 h |

Ground rules (unchanged from study 2):
- **The test split is opened only in Phase 5 and 6.** Any script that computes a test metric before then is a bug.
- Each phase ends with a commit and an intense audit. A phase is done only when its exit check passes.
- If a gate fails, stop and write it down. No tweak-and-retry inside this study.
- **The study 1/2 artifacts are read-only.** Every study 3 output goes to `generator_comparison/study3/`.

```
Phase 0  Decide and draft the pre-registration    ~1 h      no GPU
Phase 1  Code + tests                             ~3-4 h    no GPU, no test
Phase 2  Subsets, 9 generators, 12 pools          ~6 h      GPU, no test     (overnight)
Phase 3  Validation gate                          ~1 h      GPU, val only    <- GO / NO-GO
Phase 4  Register                                 ~0.5 h    no GPU
Phase 5  Run the study (180 runs)                 ~3 h      GPU, test saved but not looked at
Phase 6  Analyse (once)                           ~0.3 h
Phase 7  Report                                   ~2 h
```

---

## Phase 0: Decide and draft (needs your OK on each decision)

Recommended defaults; each one changes what the result means, so they are fixed before any code runs.

| # | Decision | Recommended | Alternative |
|---|---|---|---|
| D1 | **How much synthetic data** | Fill each minority class up to a fixed target of **2 × its full real count** (AFIB 2,112 / STACH 1,136 / SBRAD 892). At 100 % that is exactly study 2 (ratio 1.0); at 10 % it is ~10 % real + ~90 % synthetic. The training-set size is the same at every level and condition, so only the real share changes. | Ratio 1.0 of the subset (at 10 %: 106 real + 106 synthetic AFIB). Smaller gain, harder to see. *(Plain "top up to the full count" is not recommended: at 100 % it adds nothing.)* |
| D2 | **Oversample control** | Duplicate the real subset up to the same target as D1. | — |
| D3 | **Subsets** | Patient-level, stratified by class, **nested** (10 ⊂ 25 ⊂ 50 ⊂ 100) within a draw. 3 independent draws. SR is never subsampled. Index files saved and hashed. | 2 draws (cheaper, ~3 h less GPU) |
| D4 | **Units** | (draw, seed): 3 draws × 5 seeds = 15 paired units per level. At 100 % the 3 "draws" are 3 pools from the frozen generator. | 3 × 10 = 30 units (+3 h) |
| D5 | **Primary test** | One test: **synthetic − real at 10 %**, macro-F1, two-sided Wilcoxon, α 0.05. | Trend of the gain across levels as primary |
| D6 | **Secondary family (Holm)** | synthetic − oversample at each of the 4 levels; trend test (gain decreases with level: Spearman between level and per-unit gain, or Page's test). | — |
| D7 | **Threshold-free metric** | Also report macro and SBRAD **AUPRC** (descriptive). Study 2 showed the F1 gain was mostly an operating-point shift. | — |
| D8 | **Validation split** | Kept full for checkpoint selection (isolates the training-data effect); disclosed. Validation probabilities saved, so a threshold-tuned real-only control *can* be added (optional, see the end). | Subsample validation too (more realistic, noisier) |
| D9 | **Test split** | Reuse PTB-XL test, disclosed as the 4th analysis on it. A "fresh holdout" from train is not possible: the frozen 100 % generator saw every train patient. | External test set (Chapman-Shaoxing or Georgia): stronger, needs a download and label mapping |
| D10 | **Synthetic selection** | Random from the pool (no quality filter; study 2 showed it adds nothing). | — |

**Seeds (new, no overlap with 7, 11–41, 0–239):** subset draws 1001 / 1002 / 1003; template generator per (level, draw)
3010 + 10·draw + level index; pools 3110 + 10·draw + level index (spaced so no diffusion seed is shared); classifier seeds 300–314.

**Exit check:** decisions D1–D10 written into a draft `report/PREREGISTRATION_3.md`, with power numbers taken from
study 2 (the gain at 100 % was +0.036, dz 2.6, so 15 units are plenty at 100 %; the 10 % effect size is unknown and
stated as such).

## Phase 1: Code + tests (no GPU, no test)

| # | Change | File |
|---|---|---|
| 1.1 | `_make_subsets3.py`: nested, patient-level, stratified subsets per draw from `y_train` + `train_metadata` (no X needed). Writes `study3/subsets/draw{d}.npz` (train indices per level) and SHA-256. | new |
| 1.2 | Template generator takes `--templates-idx` (keep only rows whose `src_idx` is in the subset; `templates_train.npz` already has `src_idx`), `--out` and `--seed`. Defaults unchanged. | `_run_template_gen.py` |
| 1.3 | Assembly fit takes a subset index and `--out`; it must **not** write `beat_templates/assembly_models.pkl`. | `_beat_oracle.py` or new `_fit_assembly3.py` |
| 1.4 | Pool builder takes `--models` (assembly pickle path) instead of the hard-coded one. Default unchanged. | `_beat_gen_pool.py` |
| 1.5 | Experiment: `--train-idx file:level` (subset the real minority rows), `--target-per-class` (D1/D2 fill rule), `--save-val-prob`. **Defaults unchanged**, so study 2 reproduces and its 30 tests still pass. | `augment_experiment.py` |
| 1.6 | Guard: every study 3 script refuses to start if the frozen-artifact hashes (b2b133a6…, 19a8e764…, pools2/*) differ from the study 2 registration, and checks again at the end. | `_study3_common.py` |
| 1.7 | Launchers `_build_study3.sh` (Phase 2) and `_run_study3.py` (Phase 5), resumable, for `run_retry.sh`. | new |
| 1.8 | Frozen analysis `_study3_analysis.py`: D5 primary, D6 Holm family, bootstrap over patients × draws × seeds, draw-level sensitivity (one-sample t on 3 draw means), AUPRC table. Refuses unless all 180 runs exist; refuses a second run. | new |
| 1.9 | Tests (fake data only): subsets nested and patient-disjoint from val/test; per-class counts correct; fill rule gives identical totals across conditions and levels; at 100 % the run is identical to study 2's recipe; `--train-idx` never loads test; generator subset contains **no** template from outside the subset (leak test); the analysis gives the expected decisions on fake results. | `tests/test_study3.py` |

**Exit check:** all tests pass (old 30 + new); a 2-step `--val-only` smoke run creates no `test_prob_*`; `git diff`
shows no change to the frozen study 2 files.

## Phase 2: Build the data (GPU, no test; run overnight)

```bash
py -3.14 _make_subsets3.py
bash run_retry.sh _build_study3.sh generator_comparison/study3/build.log
py -3.14 _check_study3.py
```
For each draw and level 10/25/50 %: refit assembly models on the subset, train the template generator on the subset
templates, build the pool. For 100 %: 3 pools from the frozen generator (new pool seeds).

**Checks (`_check_study3.py` → `study3/checks.json`):**
- **Leak check:** the training templates and assembly fit of each generator come only from its subset (index sets compared).
- **Memorisation:** copy fraction (corr > 0.98) of generated templates against **the subset's own templates**. Reference
  for real-to-real is 0.5–3 %. At 10 % SBRAD has only ~35 clean templates, so this is the main risk.
- Study 2 pool checks: RR inside class bounds ≤ 2 % out, replay exact, no NaN/inf, |z| > 20 share.
- Frozen study 1/2 hashes unchanged.

**Exit check:** all pools PASS; SHA-256 of subsets, generators, assembly models and pools in `checks.json`.

## Phase 3: Validation gate (val only) — GO / NO-GO

`py -3.14 _gate_study3.py` → `study3/gate.json`:
- **G1 (usable generators):** train-on-synthetic balanced accuracy on real validation is above chance (> 0.25 + 0.10)
  for every generator. A level where a generator fails is reported as "generator could not be trained at this level";
  it is not retuned.
- **G2 (memorisation):** copy fraction ≤ 0.10 for every generator. A level that fails is still run but flagged; its
  result is reported with "the generator memorised its training set".
- **G3 (determinism):** the study recipe run twice (10 %, draw 1, seed 300, synthetic, `--val-only`) gives identical
  histories.

**If G1 fails at 10 %:** the primary test cannot run; move the primary to the lowest level that passes and record why
*before* registration. **If G3 fails:** fix the code, redo Phase 3.

## Phase 4: Register
Fill the hash table in `PREREGISTRATION_3.md`, tick the preconditions, remove "DRAFT", and commit the prereg, the
launcher and the frozen analysis script together. From here on, none of them is edited.

## Phase 5: Run the study
```bash
bash run_retry.sh _run_study3.py generator_comparison/experiment_study3_run.log
```
180 runs, about 3 h. The log prints validation metrics only. A crash is rerun with the same seed; no run is excluded.

**Exit check:** 180 entries with test results; the intense audit (as in study 2): hashes match the registration, every
run trained 6,250 steps, the saved probabilities reproduce the recorded F1, no duplicate prediction files.

## Phase 6: Analyse (once)
`py -3.14 _study3_analysis.py`, exactly once, then commit the result files immediately.

## Phase 7: Report
New REPORT section "Study 3": the leakage design, the gate results, the curve (gain vs real share, with CIs), the
decisions. Anything looked at after Phase 6 is labelled **post hoc**. Update the abstract, README and project memory.

---

## Decision map
```
G1 fails at a level        ──► that level is "generator not trainable"; primary moves only if decided before Phase 4
G2 fails at a level        ──► run, but flag "memorised"
Primary significant, synthetic > real at 10 %   ──► synthetic data helps under scarcity
Primary not significant                         ──► no detectable gain even at 10 %; report the CI
Trend significant                               ──► the gain grows as real data shrinks (the curve)
```

## Optional add-on (not part of the plan unless chosen in Phase 0)
**Threshold-tuned real-only control.** For each real-only run, pick per-class decision thresholds on the saved
validation probabilities and apply them to test. It asks whether synthetic data still wins once the real-only model's
operating point is tuned. It costs no extra training, but it is one more registered comparison.
