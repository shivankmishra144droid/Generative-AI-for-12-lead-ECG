# Is synthetic augmentation's gain a threshold effect? (exploratory)

**Status: exploratory and post hoc. Not pre-registered. No multiplicity control.** This analysis only decides
whether a pre-registered study 5 is worth running. Script: `_threshold_explore.py`. Raw numbers:
`generator_comparison/threshold_explore/results.json`. Figures: `generator_comparison/threshold_explore/`.

## In plain language

In study 3, classifiers trained with synthetic minority ECGs beat real-only classifiers on macro-F1 at 25-100 %
real data. Their macro AUPRC, which ignores the decision threshold, was about the same. That suggests synthetic
data mainly moves *where* the classifier draws the line between classes, not *how well* it separates them.

To test this, we took every one of the 180 trained classifiers and let it adjust its decision rule. Each class
gets an offset added to its log-probability, and the offsets are picked to maximise macro-F1 on one half of the
validation patients. The tuned rule is then scored on the other half. No model was retrained, and nothing
outside the validation split was used.

**Result: once each classifier's decision rule is tuned, most of synthetic's advantage over real-only disappears.**
At 25 %, 50 % and 100 % real data, tuning closes 69.9 %, 76 % and 85 % of the synthetic − real gap. The
confidence intervals are wide, and a true share of about 55-70 % is still plausible at every level. What is left is
+0.005 to +0.011 macro-F1. The main driver is SBRAD: the class-weighted real-only model over-predicts SBRAD
(precision 0.33-0.43), and a negative SBRAD offset fixes most of that. At 100 % real data, a tuned real-only
model is better than the untuned synthetic model that study 3 actually trained (−0.021, 2 of 15 runs favour
synthetic). At 10 % there was no gap to explain (−0.001 before tuning).

## Method

- **Runs.** All 180 study 3 runs: 4 levels × 3 draws × 5 seeds × {real, oversample, synthetic}, using the
  saved `val_prob_*.npy` (softmax, n_val = 2,960).
- **Decision rule.** Predict argmax_c (log p_c + b_c), with b_SR = 0 and b_AFIB, b_STACH, b_SBRAD tuned. This
  is a per-class prior shift, and it still gives exactly one label per ECG. The search is coordinate search on
  the grid b ∈ [−4, 4] with step 0.05, up to 4 sweeps, maximising macro-F1 on the tuning half, starting from
  b = 0. When several grid values tie for best, the one closest to the current value is kept. Probabilities
  are cast to float64 and clipped at the float32 `tiny` before taking the log.
- **Splits.** Validation is split into two halves by patient (2,598 patients). The split is stratified by each
  patient's rarest class, so a patient with any SBRAD record falls in the SBRAD stratum. We tune on A and
  score on B, then the reverse, over R = 50 splits with seeds 0-49. Every run uses the same 50 splits, so
  comparisons between conditions stay paired.
- **Scoring.** Untuned and tuned macro-F1 are both scored on the same held-out halves. Each run's score is the
  mean of 100 held-out half-scores (50 splits × 2 directions).
- **Units.** Per level, there are 15 units (3 draws × 5 seeds), paired across conditions.
  - The Wilcoxon signed-rank test is two-sided, unadjusted, and exploratory.
  - The 95 % bootstrap CI resamples the 3 draws, then 5 seeds within each drawn draw, with 10,000 replicates.
    **This CI is coarse:** 3 draws give only 10 distinct draw multisets. Treat the Wilcoxon over 15 units as
    the main evidence and the bootstrap as a sensitivity check.
  - Share closed = 1 − (tuned gap / untuned gap), computed from mean gaps. Its CI uses a ratio of means within
    each bootstrap replicate.

## 1. Macro-F1, untuned vs tuned (held-out validation halves)

Mean ± SD over the 15 runs. Macro AUPRC is on the full validation set and is the threshold-free reference.
**Tuning cannot change AUPRC**, because a per-class offset does not change how ECGs are ranked within a class.

| Level | Condition | Untuned macro-F1 | Tuned macro-F1 | Δ (tuned − untuned) | Macro AUPRC |
|---|---|---|---|---|---|
| 10 % | real | 0.7356 ± 0.0183 | 0.7291 ± 0.0209 | −0.0065 | 0.765 ± 0.027 |
| 10 % | oversample | 0.7115 ± 0.0217 | 0.7126 ± 0.0185 | +0.0011 | 0.730 ± 0.023 |
| 10 % | synthetic | 0.7346 ± 0.0154 | 0.7370 ± 0.0145 | +0.0024 | 0.768 ± 0.021 |
| 25 % | real | 0.7905 ± 0.0107 | 0.8046 ± 0.0131 | +0.0141 | 0.850 ± 0.008 |
| 25 % | oversample | 0.8040 ± 0.0074 | 0.7997 ± 0.0084 | −0.0043 | 0.831 ± 0.008 |
| 25 % | synthetic | 0.8169 ± 0.0111 | 0.8126 ± 0.0117 | −0.0043 | 0.850 ± 0.013 |
| 50 % | real | 0.7891 ± 0.0109 | 0.8228 ± 0.0123 | +0.0337 | 0.869 ± 0.009 |
| 50 % | oversample | 0.8159 ± 0.0077 | 0.8215 ± 0.0100 | +0.0056 | 0.861 ± 0.008 |
| 50 % | synthetic | 0.8325 ± 0.0099 | 0.8334 ± 0.0099 | +0.0009 | 0.874 ± 0.008 |
| 100 % | real | 0.7837 ± 0.0059 | 0.8399 ± 0.0119 | +0.0562 | 0.885 ± 0.008 |
| 100 % | oversample | 0.7996 ± 0.0091 | 0.8384 ± 0.0076 | +0.0388 | 0.881 ± 0.008 |
| 100 % | synthetic | 0.8188 ± 0.0088 | 0.8452 ± 0.0086 | +0.0264 | 0.888 ± 0.005 |

Tuning helps a condition when its operating point is off and the half-set offset estimate is stable enough.
It helps real most, and more so as real data grows. At 10 %, and for synthetic or oversample at 25 %, it gives
nothing or slightly hurts: the offsets fitted on about 48 SBRAD / 60 STACH ECGs are noise there.

![macro-F1](../generator_comparison/threshold_explore/fig_macro_f1.png)

## 2. Gaps and share closed

Held-out validation macro-F1. Each gap is the mean over 15 paired units, with its 95 % bootstrap CI (draws,
then seeds). p is the exploratory, unadjusted Wilcoxon p-value. "k/15" counts the units where the gap is > 0.

### synthetic − real (the gap in question)

| Level | Untuned gap [CI] | Untuned p | Tuned gap [CI] | Tuned p (k/15) | Share closed [CI] |
|---|---|---|---|---|---|
| 10 % | −0.0010 [−0.020, +0.015] | 1.00 | +0.0079 [−0.012, +0.025] | 0.25 (10/15) | not meaningful (untuned gap ≈ 0; nominal 8.8, CI −14.8 to +14.7) |
| 25 % | +0.0264 [+0.016, +0.036] | 0.0002 | +0.0079 [−0.013, +0.025] | 0.23 (10/15) | **0.699** [0.25, 1.61] |
| 50 % | +0.0434 [+0.035, +0.052] | 0.0001 | +0.0106 [+0.002, +0.019] | 0.041 (11/15) | **0.76** [0.57, 0.94] |
| 100 % | +0.0350 [+0.029, +0.043] | 0.0001 | +0.0053 [−0.002, +0.014] | 0.30 (9/15) | **0.85** [0.56, 1.05] |

Per-unit SD of the tuned synthetic − real gap (n = 15) is 0.026 / 0.022 / 0.017 / 0.015 at 10 / 25 / 50 /
100 %. The draw means of the tuned gap are:

| Level | Draw 1 | Draw 2 | Draw 3 |
|---|---|---|---|
| 10 % | −0.012 | +0.020 | +0.016 |
| 25 % | +0.019 | −0.013 | +0.018 |
| 50 % | +0.012 | +0.010 | +0.010 |
| 100 % | +0.005 | +0.003 | +0.008 |

![gap](../generator_comparison/threshold_explore/fig_gap.png)

### Other contrasts

| Level | Contrast | Untuned gap [CI] | Tuned gap [CI] | Tuned p (k/15) | Share closed |
|---|---|---|---|---|---|
| 10 % | synthetic − oversample | +0.0231 [+0.006, +0.040] | +0.0244 [+0.012, +0.036] | 0.0001 (14/15) | −0.06 |
| 10 % | oversample − real | −0.0241 [−0.037, −0.012] | −0.0165 [−0.029, −0.003] | 0.010 (3/15) | 0.32 |
| 25 % | synthetic − oversample | +0.0129 [+0.007, +0.019] | +0.0129 [+0.006, +0.019] | 0.0009 (13/15) | 0.00 |
| 25 % | oversample − real | +0.0135 [+0.006, +0.020] | −0.0049 [−0.019, +0.007] | 0.25 (7/15) | 1.36 |
| 50 % | synthetic − oversample | +0.0166 [+0.008, +0.026] | +0.0118 [+0.003, +0.019] | 0.015 (12/15) | 0.29 |
| 50 % | oversample − real | +0.0267 [+0.020, +0.034] | −0.0012 [−0.008, +0.004] | 0.80 (9/15) | 1.05 |
| 100 % | synthetic − oversample | +0.0191 [+0.013, +0.027] | +0.0068 [+0.000, +0.013] | 0.041 (12/15) | 0.65 |
| 100 % | oversample − real | +0.0159 [+0.012, +0.020] | −0.0015 [−0.009, +0.008] | 0.45 (5/15) | 1.09 |

**Untuned synthetic − tuned real.** This compares study 3's synthetic recipe, as trained, against a real-only
model with a tuned decision rule.

| Level | Gap [CI] | p (k/15) |
|---|---|---|
| 10 % | +0.0055 [−0.016, +0.024] | 0.49 (8/15) |
| 25 % | +0.0123 [−0.004, +0.027] | 0.035 (10/15) |
| 50 % | +0.0097 [+0.003, +0.017] | 0.035 (10/15) |
| 100 % | **−0.0211** [−0.029, −0.013] | 0.0003 (2/15) |

The oversample − real gap closes completely at 25-100 %. Duplicating minority ECGs is purely a threshold shift,
as expected. Synthetic keeps a small edge over oversample after tuning at every level, about +0.007 to +0.024.

### Spread over the 50 patient splits

This is the level-mean gap for each split, averaging the two directions, shown as p5 / p50 / p95 across
splits. It shows how much a single split would move the answer. It is not a CI over runs.

| Level | Untuned syn − real | Tuned syn − real |
|---|---|---|
| 10 % | −0.0014 / −0.0010 / −0.0007 | +0.0034 / +0.0081 / +0.0122 |
| 25 % | +0.0261 / +0.0264 / +0.0268 | +0.0040 / +0.0077 / +0.0119 |
| 50 % | +0.0430 / +0.0434 / +0.0437 | +0.0065 / +0.0106 / +0.0146 |
| 100 % | +0.0347 / +0.0351 / +0.0353 | +0.0028 / +0.0053 / +0.0087 |

## 3. Per-class precision and recall, untuned → tuned

Held-out halves, mean over the 15 runs.

| Level | Condition | AFIB P | AFIB R | STACH P | STACH R | SBRAD P | SBRAD R | SR R |
|---|---|---|---|---|---|---|---|---|
| 10 % | real | 0.761→0.746 | 0.705→0.712 | 0.730→0.729 | 0.746→0.727 | 0.516→0.543 | 0.562→0.515 | 0.954→0.956 |
| 10 % | oversample | 0.750→0.723 | 0.671→0.684 | 0.758→0.737 | 0.660→0.685 | 0.591→0.538 | 0.423→0.480 | 0.966→0.956 |
| 10 % | synthetic | 0.811→0.752 | 0.644→0.711 | 0.800→0.769 | 0.687→0.725 | 0.618→0.561 | 0.481→0.522 | 0.973→0.959 |
| 25 % | real | 0.821→0.827 | 0.865→0.851 | 0.796→0.808 | 0.841→0.820 | **0.435→0.651** | 0.756→0.584 | 0.939→0.966 |
| 25 % | oversample | 0.862→0.845 | 0.812→0.824 | 0.842→0.825 | 0.789→0.794 | 0.632→0.634 | 0.580→0.570 | 0.973→0.970 |
| 25 % | synthetic | 0.853→0.836 | 0.839→0.852 | 0.814→0.809 | 0.814→0.809 | 0.663→0.692 | 0.630→0.599 | 0.971→0.970 |
| 50 % | real | 0.828→0.839 | 0.903→0.887 | 0.792→0.818 | 0.874→0.842 | **0.379→0.712** | 0.796→0.581 | 0.925→0.970 |
| 50 % | oversample | 0.843→0.850 | 0.895→0.883 | 0.817→0.823 | 0.840→0.836 | 0.537→0.690 | 0.700→0.587 | 0.956→0.970 |
| 50 % | synthetic | 0.843→0.843 | 0.895→0.889 | 0.832→0.836 | 0.849→0.841 | 0.615→0.725 | 0.710→0.621 | 0.963→0.972 |
| 100 % | real | 0.823→0.856 | 0.933→0.907 | 0.799→0.821 | 0.896→0.868 | **0.331→0.743** | 0.860→0.615 | 0.906→0.971 |
| 100 % | oversample | 0.834→0.868 | 0.935→0.909 | 0.826→0.832 | 0.881→0.863 | 0.375→0.720 | 0.816→0.603 | 0.925→0.972 |
| 100 % | synthetic | 0.843→0.869 | 0.936→0.912 | 0.806→0.836 | 0.903→0.872 | 0.462→0.750 | 0.784→0.616 | 0.941→0.974 |

Per-class F1 (untuned → tuned), SBRAD only, since that is where the movement is:

| | real | oversample | synthetic |
|---|---|---|---|
| 10 % | 0.526→0.516 | 0.488→0.495 | 0.532→0.525 |
| 25 % | 0.548→0.604 | 0.599→0.593 | 0.641→0.633 |
| 50 % | 0.511→0.633 | 0.605→0.625 | 0.655→0.661 |
| 100 % | 0.475→0.666 | 0.513→0.650 | 0.577→0.667 |

The pattern is the same throughout. At 25-100 % every model over-calls SBRAD, and the class-weighted real model
does so most (SBRAD weight about 66× at full data). Untuned, it trades very low SBRAD precision (0.33-0.43) for
high SBRAD recall and lower SR recall. Tuning moves all conditions to about the same SBRAD operating point:
precision 0.69-0.75, recall 0.58-0.62. AFIB and STACH barely change. After tuning, per-class F1 is nearly
identical across conditions at 50-100 %.

## 4. Fitted offsets (median [IQR] over 15 runs × 100 fits)

The offset is added to log p, so a negative value means the class is called less often.

| Level | Condition | b_AFIB | b_STACH | b_SBRAD |
|---|---|---|---|---|
| 10 % | real | +0.30 [−0.40, +0.85] | 0.00 [−0.65, +0.31] | −0.55 [−1.10, 0.00] |
| 10 % | oversample | +0.65 [0.00, +1.35] | +0.85 [0.00, +1.95] | +0.90 [+0.25, +1.91] |
| 10 % | synthetic | +1.30 [+0.75, +1.80] | +0.95 [0.00, +1.90] | +0.15 [−0.20, +1.10] |
| 25 % | real | −0.55 [−1.30, +0.45] | −0.25 [−1.25, 0.00] | −1.85 [−2.35, −1.45] |
| 25 % | oversample | +0.75 [−0.15, +2.05] | +0.25 [0.00, +1.80] | 0.00 [−1.10, +0.75] |
| 25 % | synthetic | +0.40 [−0.25, +1.10] | +0.15 [−0.25, +1.20] | −0.25 [−0.70, 0.00] |
| 50 % | real | −0.65 [−1.85, +0.05] | −1.00 [−2.45, 0.00] | −2.25 [−2.95, −1.90] |
| 50 % | oversample | −0.38 [−2.00, +0.50] | 0.00 [−1.35, +0.81] | −2.45 [−3.05, −1.05] |
| 50 % | synthetic | −0.15 [−0.95, +0.75] | 0.00 [−0.85, +0.35] | −0.80 [−1.30, −0.20] |
| 100 % | real | −1.30 [−2.45, −0.60] | −1.00 [−1.95, 0.00] | −2.30 [−3.15, −1.90] |
| 100 % | oversample | −1.70 [−2.95, −0.75] | −0.80 [−2.15, 0.00] | −2.90 [−3.45, −2.35] |
| 100 % | synthetic | −1.25 [−2.35, −0.25] | −1.43 [−2.40, −0.20] | −1.50 [−2.10, −1.05] |

The IQRs are wide, often 1-2 log units. Only the real-condition SBRAD offset at 25-100 % is consistently far
from 0. Only 0.46 % of fitted offsets hit the ±4 grid edge (AFIB 0.13 %, STACH 0.35 %, SBRAD 0.91 %), so the
grid did not limit the result and was not widened.

## 5. Audit checks (outputs as printed by the script)

```
validation class counts {'SR': 2515, 'AFIB': 224, 'STACH': 125, 'SBRAD': 96} patients 2598
[audit] loading: 180 runs, max |argmax mF1 - best_val_macro_f1| sklearn 0.00e+00, fast scorer 0.00e+00
[audit] leak: OK. per-half class counts over 50 splits: min {'SR': 1239, 'AFIB': 100, 'STACH': 59, 'SBRAD': 47} max {'SR': 1276, 'AFIB': 124, 'STACH': 66, 'SBRAD': 49}
[audit] null: OK
[audit] search == direct (max diff 0.0e+00); in-sample tuned >= untuned on every fit; optimism: in-sample 0.8213 >= held-out 0.7997; offsets on grid edge 0.0046
```

- **Loading check.** Plain argmax on the full validation set reproduces every run's `best_val_macro_f1` exactly
  (max abs diff 0), both with `sklearn.metrics.f1_score(average="macro")` and with the script's numpy scorer.
- **Leak check.** No patient appears in both halves of any of the 50 splits, which is asserted. Per-half class
  counts range over SR 1,239-1,276, AFIB 100-124, STACH 59-66 and SBRAD 47-49. Split 0: A = [1259, 106, 62,
  48], B = [1256, 118, 63, 48].
- **Optimism check.** Mean tuned-on-A-scored-on-A is 0.8213, against 0.7997 for tuned-on-A-scored-on-B, which
  passes. Also asserted for every one of the 18,000 fits: in-sample tuned ≥ in-sample untuned.
- **Null check.** argmax(log p + 0) equals argmax(p) on every run and half. The incremental search's in-sample
  score also equals a direct argmax(log p + b) rescoring on every fit (max diff 0).
- **No data outside validation.** Every file load goes through an allowlist guard that accepts only `y_val.npy`,
  `val_metadata.csv`, `results.json` and `val_prob_*.npy`. 194 files were opened. Each `results.json` was opened
  only to read `best_val_macro_f1`, and the rest of it was discarded unused. Paths are built explicitly, not
  globbed. `grep -n -i "test" _threshold_explore.py` returns nothing:
  ```
  grep:
  (end grep)
  ```

## 6. Caveats

- **Checkpoint selection used validation.** Each checkpoint was chosen on full-validation *plain-argmax*
  macro-F1, using both halves, so every validation score here is optimistically biased. The bias favours the
  untuned rule specifically, since that is what was maximised, and it may not be equal across conditions.
  The held-out macro-F1 used here is 0.026-0.043 for the untuned synthetic − real gap, versus 0.036-0.048 on the
  registered test split. So the gap being explained is somewhat smaller on validation than on test.
- **Small minority counts.** Each half holds only about 48 SBRAD, 62 STACH and 112 AFIB ECGs. The offsets are
  noisy (wide IQRs, section 4), and tuning on half-sets sometimes hurts (real at 10 %, synthetic at 25 %). A
  real deployment would tune on the full validation set, which halves the variance of the offsets (their SD
  falls by about √2).
- **Coarse bootstrap.** Only 3 draws give 10 distinct draw multisets. The 15-unit Wilcoxon is the better guide,
  and it is unadjusted.
- **Unstable share closed.** The share-closed ratio is unstable wherever the untuned gap is small. At 10 % it is
  meaningless, and at 25 % its CI runs from 0.25 to 1.61.
- **Exploratory, post hoc, no multiplicity control.** This was run after seeing study 3's results. It decides
  only whether study 5 is worth doing.

## 7. Verdict

**Mostly closes at 50 % and 100 %, borderline at 25 %.** This meets the "mostly closes" criterion (> 70 % at
every level from 25 to 100 %) only by point estimate, and only at 50 % and 100 %. The CIs leave a share of about
55-70 % plausible at every level. On held-out validation halves, tuning per-class log-offsets on each run closes
**69.9 % (25 %, CI 0.25-1.61), 76 % (50 %, CI 0.57-0.94) and 85 % (100 %, CI 0.56-1.05)** of the
synthetic − real macro-F1 gap. The untuned gaps are +0.026,
+0.043 and +0.035, and they fall to **+0.008, +0.011 and +0.005** after tuning. At 25 % and 100 % the residual
is not distinguishable from 0 (Wilcoxon p = 0.23 and 0.30). At 50 % a small residual remains (p = 0.041,
unadjusted; CI +0.002 to +0.019). At 100 % real data, a tuned real-only model beats the untuned synthetic
model as trained (−0.021, 13/15 runs). At 10 % there was no gap to begin with. This matches the flat macro
AUPRC from study 3: most of synthetic's macro-F1 advantage comes from where the SBRAD decision line sits,
which the class-weighted real model places badly. It does not come from better class separation. A small
synthetic edge of about 0.005-0.01 survives tuning, and it is the only part that might be a real
representation benefit. That edge is smaller than this design can confirm.

## 8. Is a study 5 justified, and what should it register?

**Yes.** Study 3's headline effect, synthetic > real, has a plausible and cheap alternative explanation that
works in practice: tune the real-only model's per-class decision offsets. Any claim that synthetic ECGs help
should be judged against that stronger baseline.

**Hypothesis.** Against a real-only CNN-LSTM whose per-class log-offsets are tuned on held-out real data, the
synthetic condition (with its own offsets tuned the same way) does not improve macro-F1 by a margin that
matters.

**Primary test.** The remaining effect is small (about +0.005 to +0.011), so the primary test should be
equivalence rather than superiority.
- Run TOST on tuned synthetic − tuned real macro-F1 at 100 % real data, with margin ±0.015, α = 0.05.
- From the per-unit SD seen here (about 0.015), a true difference of 0.005 needs about 20-25 paired units for
  80 % power. Use at least 5 draws × 5 seeds, because draws are the honest unit and 3 is too few.
- Secondary: the same comparison at 25 % and 50 %; tuned real vs untuned synthetic; macro AUPRC (expected
  equal); SBRAD precision and recall.
- The tuning procedure is fixed in advance: this script's coordinate search, grid [−4, 4] step 0.05,
  4 sweeps, b_SR = 0.

**Data to hold out.** Offsets must be tuned on data that is not scored, and ideally not the checkpoint-selection
data either. In this exploration they overlap: checkpoints were picked on full validation, and the offsets were
tuned on half of it. Two options:
- **Cheap, no training.** Tune offsets on the full validation set for the saved study 3 runs, then score the
  registered study 3 test probabilities once. This is allowed only as a registered analysis.
  - It is not fresh. Study 3's test macro-F1 and AUPRC are already known, which hints at the answer.
  - It also inherits the validation overlap between checkpoint selection and tuning.
- **Clean, retraining required.** Carve a patient-disjoint calibration split, about 15 % of train patients,
  out of training.
  - Train on the rest, select checkpoints on validation, tune offsets on the calibration split, and judge once
    on test.
  - The PTB-XL test split has been looked at in aggregate, so for a truly fresh judge, add an external rhythm
    dataset. That needs a download, which is left for later.
  - Note that the study 4 patient split partitions the train split, which study 3 trained on. It contains no
    unseen patients that could serve as a holdout here.
