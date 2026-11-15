# V2 pre-run discrepancies and interpretations

Recorded 31 October 2026, before V2 metric computation or training. Original instructions are preserved; their hash and gate text are in `gate_registry.json`.

## Confirmed against files

- Fit counts [9964,886,486,383], dev counts [1777,170,82,63], and fit/dev totals match the instructions. There are zero overlapping fit/dev patients.
- V1 VAE, diffusion and ECGSYN pools each contain only 64 per supported class. ECGSYN contains no AFIB. These are insufficient for the larger A/F comparisons.
- V1 code matches its final audit. No V1 code or result was changed during planning.

## Pre-run protocol amendments

- **A3 feasibility:** two disjoint development patient halves cannot each contain the complete minority development reference. The fixed seed-9200 partition and actual feasible counts are in `preflight.json`. Use those matched counts for floor-calibrated comparisons; retain full-reference metrics separately. Preserve A8 thresholds and label the original full-n floor requirement infeasible. No fit patients are substituted.
- **A/F baseline size:** expand baseline pools with frozen V1 models/settings into V2 output paths. Original saved-pool rescoring remains a separate 64/class analysis. Do not manufacture independent samples by duplicating saved waveforms.
- **C fixed dev loss cohort:** 63 unique SBRAD recordings prevent a 64-per-class cohort. Use 63/class, 252 total, with fixed noise. This changes a diagnostic sample count, not Gate C.
- **D1 validation timing:** delay validation R-map computation until F to honor the higher-priority hard rule against pre-F validation use.
- **Final audit preservation:** the existing `final_audit.py` writes V1 outputs. Reuse its checks through a V2-output wrapper instead of overwriting the preserved V1 audit.

## Interpretation cautions, not changes to numerical gates

- Reconstruction judge AUC is not a mathematical bound for all latent priors.
- Percentile CI point inclusion and shrinking width are not universal statistical guarantees. Keep truthful intervals and fixed-seed tests; do not force the desired result.
- Five repeated-split judge scores yield mean, SD and range, not automatically a 95% confidence interval. Any F decision using a CI needs its construction explicitly fixed before F; no conversion of a range into a confidence interval.
- Contrastive augmentations include noise and lead dropout, so noise sensitivity of the second embedding must be demonstrated by controls.
- D's channels/map resolution must follow B's chosen latent dimensions.
- The dropout probabilities in D need an explicit joint distribution before training (independent 10% drops plus a 5% joint override is one interpretation); gate results must state the interpretation used.
- Synthetic-SR augmentation also changes training counts and class weights; it is not a uniquely identifying shortcut intervention.

## Preflight self-review

- Loaded training arrays and metadata only; no validation waveform and no prohibited final-test array loaded.
- Used global patient grouping for the real-floor split, not separate record-level splits per class.
- Carried recording indices unchanged and retained all source IDs in the artifact partition.
- Did not read any Study 5 outputs or write outside the Study 9 folder.
- Checked GPU status without launching a GPU job. Availability must be rechecked before training.
- Stored gates before any V2 outcome, and did not mark A or later phases complete.
