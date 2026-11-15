# Pre-registration 3: synthetic augmentation under real-data scarcity

**Status: REGISTERED** 2026-10-29 (this commit), after the Phase 3 validation gate (GO) and before any study 3
classifier run touched the test split. Written as a draft before any gate result; Section 8 filled at registration.

Decisions D1–D10 are the recommended defaults in `WORKFLOW_3.md`; they were approved on 2026-10-29
("start phase 1 to 7, do everything"). Changes made after the design review are listed in Section 9.

## 1. Question
How much does synthetic minority-class data help a 4-class rhythm classifier (SR / AFIB / STACH / SBRAD, PTB-XL)
when real minority data is scarce, and does synthetic *content* matter more than plain duplication as real data shrinks?

## 2. Data levels and leakage control
- **Levels:** 10 %, 25 %, 50 %, 100 % of each real minority class (AFIB 1,056 / STACH 568 / SBRAD 446). SR (11,741)
  is never subsampled.
- **Subsets:** 3 independent draws (seeds 1001–1003). Patient-level (a patient's recordings of a class are kept or
  dropped together), nested within a draw (10 ⊂ 25 ⊂ 50 ⊂ 100). Files `generator_comparison/study3/subsets/draw<d>.npz`
  (`_make_subsets3.py`). Real counts at 10 %: AFIB 106, STACH 57–58, SBRAD 45.
- **No leakage:** for every level below 100 %, the template generator is retrained and the assembly (rhythm, residual,
  wander) models refitted on that draw's subset only (`_study3_build.py`), with the frozen recipe of
  `_run_template_gen.py` (15k steps, v-prediction, 8×64, EMA) and generator seed 3000 + 100·draw + 10·level-index.
  The 100 % level uses the frozen study 1/2 generator and assembly models (3 pools, one per "draw").
  `_check_study3.py` verifies that every training template and every assembly template comes from the subset.
- **Pools:** per (level, draw), 440 SR + 2,112 AFIB + 1,136 STACH + 892 SBRAD, pool seed 4000 + 100·draw + 10·level-index.

## 3. Conditions (D1, D2, D10)
Per level, with the real subset as the base:
- `real`: SR + the real minority subset.
- `oversample`: each minority class duplicated (cycled, deterministic) up to **2 × its full real count**
  (AFIB 2,112 / STACH 1,136 / SBRAD 892).
- `synthetic`: each minority class filled up to the same 2 × target with ECGs drawn at random (seed 1000 + training
  seed) from that (level, draw)'s own pool. No quality filter.

At 100 % this is exactly study 2's `real` / `oversample` / `random` recipe.

## 4. Training (unchanged from study 2)
CNN-LSTM, Adam 1e-3 / wd 1e-4, batch 64, **6,250 fixed steps** per run, validation every 250 steps, LR halved after
3 evaluations without improvement, best-validation state kept, class-weighted cross-entropy from the condition's own
labels, deterministic CUDA. Validation split kept full (D8). Training seeds: draw 1 → 300–304, draw 2 → 305–309,
draw 3 → 310–314. 4 levels × 3 draws × 5 seeds × 3 conditions = **180 runs** (`_run_study3.py`).

## 5. Hypotheses and tests (`_study3_analysis.py`, frozen at registration, run once)
Unit: (draw, seed), 15 per level, paired across conditions and matched across levels. Metric: test macro-F1.
- **H1 (primary):** synthetic − real at 10 % ≠ 0. Two-sided Wilcoxon signed-rank, α = 0.05, no adjustment.
- **Secondary family, Holm (family-wise α 0.05):**
  - H2-10 / H2-25 / H2-50 / H2-100: synthetic − oversample at each level, two-sided Wilcoxon.
  - H3: the synthetic − oversample gain increases as the real share falls. Page's L test, levels ordered 100 < 50 < 25 < 10.
  - H4: the same trend for synthetic − real. *Expected to hold whatever synthetic content does,* because at 10 %
    the real-only run weights SBRAD about 66× (vs ~4.5× in the filled conditions); reported, but H3 is the
    informative trend.
- **95 % CIs:** three-level bootstrap (test patients × draws × seeds within draw), 2,000 resamples, seed 0.
- **Descriptive only:** every cell's mean and SD; per-class F1 and recall; per-class and macro AUPRC; oversample − real;
  per-draw mean differences with a one-sample t-test on the 3 draw means; figures (curves, gains, recall).
- **Consistency check:** the 100 % cells should land near study 2's means (0.803 / 0.820 / 0.839). A large gap
  means a bug in the new script, not a finding.

Decision wording: adjusted p < 0.05 and positive mean → "SUPPORTED"; otherwise "NOT SUPPORTED".

## 6. Validation gate (`_gate_study3.py`, validation data only)
- **G1:** train-on-synthetic balanced accuracy on real validation > 0.35 for every pool. A level below 100 % that
  fails is reported as "generator not trainable at this level"; nothing is retuned.
- **G2 (flag, not a stop):** memorisation copy fraction (corr > 0.98 to a template of the same subset) ≤ 0.10. Pools
  failing `_check_study3.py` flag their level the same way.
- **G3:** the study recipe run twice gives identical validation histories.
- **GO** needs G1 and the checks to pass at 10 %, and G3. If 10 % fails, the primary moves to the lowest passing level,
  decided and recorded here **before** registration.

## 7. Known limitations (stated in advance)
- The PTB-XL test split is reused (4th analysis on it). A fresh holdout from train is not possible: the frozen 100 %
  generator saw every train patient.
- Seeds within a draw share the subset and (below 100 %) the generator, so the 15-unit Wilcoxon is optimistic; the
  bootstrap (which resamples draws) and the draw-level t-test (df = 2) carry the honest uncertainty and are reported
  next to H1.
- At 100 % one generator serves all 3 draws; the lower levels have 3 generators each, so generator-training variance
  is missing from the 100 % cells.
- Per-lead z-score statistics come from the full training set; SR recordings of patients whose minority recordings
  were dropped stay in; validation is kept full.
- At 10 % the SBRAD generator learns from ~45 templates; memorisation is checked (G2) but cannot be ruled out.
- One classifier, one generator family, one dataset; synthetic-to-real targets fixed at 2× the full count.

## 8. Preconditions and hashes (filled at registration)
- [x] P1 tests pass (`py -3.14 -m pytest tests -q`: 51 passed, 2026-10-29)
- [x] P2 all 12 pools built (one attempt, no resume; all 9 generators at 15,000 steps); `_check_study3.py`: ALL PASS
  (leak checks, copy fraction <= 0.002, out-of-class 0, |z| > 20 <= 0.4 %, counts 440 / 2,112 / 1,136 / 892)
- [x] P3 gate (`generator_comparison/study3/gate.json`): **GO**
  - G1 train-on-synthetic balanced accuracy (> 0.35): L10 0.593 / 0.571 / 0.463, L25 0.674 / 0.617 / 0.695,
    L50 0.714 / 0.679 / 0.739, L100 0.736 / 0.730 / 0.748 (draws 1 / 2 / 3); every level passes, so H1 stays at 10 %
  - G2 max copy fraction 0.0018 (L10 d1); nothing flagged
  - G3 two `--val-only` runs (L10 d1 seed 300 synthetic): identical validation histories (val macro-F1 0.737)
- [x] P4 hashes below
- [x] P5 frozen study 1/2 artifacts unchanged (`_study3_common.check_frozen`)

| file | sha256 |
|---|---|
| `subsets draw1.npz` | `c25b1858797d607edd5c4bbc6016658224ec19e6d692512b635f40420e22452f` |
| `subsets draw2.npz` | `12f41b02d56f921765ec416c7d6b16bd822d157481a61aeaf869b9e885c9a772` |
| `subsets draw3.npz` | `7602b53e5dbb04b4bc5bed4a5a9923fe9f7eee728765787d1dba2f6a87846151` |
| `L10_d1 pool` | `45a4b40d4e1a1931eb9b53ba641904dff237f57357abedf219d326ce177335c3` |
| `L10_d1 generator tpl_last.pt` | `6384142bc92358d323293fa385bbef825b1f44938cab762dd422175688a73509` |
| `L10_d1 assembly_models.pkl` | `85da4124c4db0d20dc335df6e20fa05d60bf6180c25a13bc94dd2271f7d5145a` |
| `L10_d2 pool` | `05749860143fa727cf3bbbcb6eadd442ff84f548ce42fda82a756a27ec898448` |
| `L10_d2 generator tpl_last.pt` | `2679cba07c603b5635e33ba536ec75203c90ea7bc92273628e1b0a1892ff0539` |
| `L10_d2 assembly_models.pkl` | `b875e18d6d0d1d90971d33e98a5ac82e4a1118e290b325101252d0ec97c5a89a` |
| `L10_d3 pool` | `be2e15faf2faf9254883adebc3c66a5f01f9d1dcbee455a34c2df7673c809dcb` |
| `L10_d3 generator tpl_last.pt` | `2de1d0e3deef1b1c24219ae18d6be94f77e7d21105b451153e7d4e0ac563b0cb` |
| `L10_d3 assembly_models.pkl` | `8568fbc691953ecf098d95525c7d64d2469098d68431b5100e93f1f131097436` |
| `L25_d1 pool` | `bfb70f9ee39c253ed085b026f3e3d93df48e1576487765f5e49295eb9589984f` |
| `L25_d1 generator tpl_last.pt` | `50f69f4a812303a6456219bcf5a5f8737cd3893010658800d8abea06f4a432ec` |
| `L25_d1 assembly_models.pkl` | `4117d0bb0e98ae1287a04f8b9f3b6d0c93f81f22612f495a996c68e2e4219d4c` |
| `L25_d2 pool` | `74e8208651d0bcd0967fe4f9beff1c8d2cbab73bce2f8300c05e4e46fb9ecb46` |
| `L25_d2 generator tpl_last.pt` | `9f7e736585f6881130f46ff50033055d183f0c0d4855936e8383aaba3e6ba966` |
| `L25_d2 assembly_models.pkl` | `e264538867187c627cfeafa345c0470faf4fbf902d7c531151196e2a72c87ec4` |
| `L25_d3 pool` | `68545a3f42ed91a7f126cf84751d6ee104bdf9a9befd206ac164350df4122862` |
| `L25_d3 generator tpl_last.pt` | `38d01ac3b8fa95f6464e96c4ad275d98c857ba3d54201bf4e64be7d4d95d9308` |
| `L25_d3 assembly_models.pkl` | `514251494260b4d42372895b2e4d6db2589b6c3566d9e70e6e10ebd8187b829f` |
| `L50_d1 pool` | `3a39e9e4d8674ca5108a0847137b8dd0c1b218e79749b83601bb153a65f6eec0` |
| `L50_d1 generator tpl_last.pt` | `7ccfa4a5a3775dc9ea492d373a6c068ddc47c9b1ef4e23a9e0f35de0d0f81d51` |
| `L50_d1 assembly_models.pkl` | `8a578cc739283eaeb1a62023cac27bb8b3408b4acf249d560d2096695c9d8d09` |
| `L50_d2 pool` | `d3ec8dfc932d332f941fa93df6cdc9fe7e7e44ba8e2f2ee11a47d7868a0e1517` |
| `L50_d2 generator tpl_last.pt` | `de3325cd39a60d8edf6e6879d85bf5c36d7f38d37bc851c9852785c81284238a` |
| `L50_d2 assembly_models.pkl` | `f74a7f269f9db5d5de3fe1217fcff15d13be93add43278f9170262180c646c5b` |
| `L50_d3 pool` | `9f28a8ae89122a407b2b02a97600367b969b2ed4280ff65237f20c2db273ed78` |
| `L50_d3 generator tpl_last.pt` | `4c42fc0cb6e96f8f430891cf7591e82084f2ceeb9f92459e7e71b936c2d5c4d2` |
| `L50_d3 assembly_models.pkl` | `f46354ec3310f60e3669e08964a859d39b737c5fa26f5d54fad8f30bbdd2284d` |
| `L100_d1 pool` | `1088440df6a3b94f277b43721778ca6ed874286641bbdb852237f61e77047768` |
| `L100_d2 pool` | `68a5db9e8a21b4df5106c8e0353cacc4858a38bbdbb3f9c3a6d76a735646be2e` |
| `L100_d3 pool` | `5282d9fefdf2653367e9e97fa06aead6812950bf2e45d01e9f6c0b6fcb0d0fa6` |
| `_study3_common.py` | `946e9af89c60527bb393cca5b093912b2a0d03bba4254cd3d775811839c54144` |
| `_study3_experiment.py` | `9b7773b164fc3b354da206f479cd686bd260ea45566e3cc98f20dff5734c2b80` |
| `_run_study3.py` | `013ecb700d346c3fcdb58ea3b3626a7378148b3adb5b1a1e4bb6e7fd0010d76e` |
| `_study3_analysis.py` | `93279a2d94bece0cc644249d24132e3bf192e29551de4392c01f1425d80f92c0` |
| `generator_comparison/augment_experiment.py` | `8bc2f83c214f98e14c268d872afb3d273041fd374c1ad234ddf79c4f325d49a5` |
| `generator_comparison/study3/checks.json` | `8eb38d6895a1b61fa2834fa21bbadf6259c1583c4ffbd67f15cc4d08330544b4` |
| `generator_comparison/study3/gate.json` | `14f578ceadbd6d20643fd94aecde90e3cea660364dc7722ed03854c282bc23b2` |

## 9. Changes after the design review (before any gate result)
- H3 moved from the synthetic − real trend to the synthetic − oversample trend; the synthetic − real trend kept as H4.
- Failure rule: a level whose pools fail the checks is flagged, not a stop; only a failure at 10 % stops the study.
- The 100 % consistency check against study 2 was added.
