# Phase A — code review and empirical acceptance audit

The implementation follows the original gate registry and the preregistered feasible-floor amendment. The original full-n floor is impossible with two patient-disjoint halves. Floor-calibrated comparisons use 256 SR, 79 AFIB, 38 STACH and 29 SBRAD per side. The separate primary reference uses 256, 170, 82 and 63 respectively.

## Review checks

- Re-read the patient selection code. It keeps every minority development recording in the primary reference, preserves repeated-patient groups, and uses globally disjoint patient halves for the floor. No fit patients fill missing floor rows.
- Re-read the embedding paths. Task and SSL encoders are frozen; one fixed fit-only reference determines their scaling. The SSL encoder saw only fit patients during optimization. Its development loss selected the checkpoint. Historical selection of the task encoder on validation remains a limitation.
- Re-read PRDC against the frozen implementation and checked its point estimates with an independent test call. Noise samples can all be accepted by a few large real-neighbour balls. This is a failure of the requested acceptance criterion, not evidence that noise is a realistic ECG. The original failures remain in `phaseA_gate.json`.
- Checked cluster resampling, independent synthetic resampling and paired-source resampling. Within each sampled patient all recordings travel together. Five grouped judge splits report mean, SD and range; they are not relabelled as a 95% CI.
- Checked all 25 feature names, the 5/8/8/4 group partition, missing-value denominators, 100 Hz timing, strict >2-second gaps and the development AFIB median used for irregularity. A missing HR is reported separately from an observed out-of-range HR.
- Checked the copy control, class-specific collapse control, Gaussian per-lead SD control and fixed-seed bootstrap sanity simulation. The AE fallback was evaluated descriptively after the preferred embedding failed; it did not replace the original result or thresholds.
- Detected a provenance problem in the initial scoring path: historical hybrid training includes development patients, so development cannot supply nonmembers for its privacy attack. The corrected code excludes that attack. Any provisional hybrid privacy output is archived and explicitly invalidated; the unrelated metrics remain valid descriptive measurements.
- Added five-seed secondary CNN uncertainty through the frozen detector function. The first seed is retained, extra seeds are saved individually, and these detector scores never affect Gate A. A seed t interval is distinguished from a patient-bootstrap interval.
- Checked every result/checkpoint path remains under Study 9 V2, the GPU guard logs and waits for other workloads, and STOP is checked between scorecards and generation batches. Encoder resume restores optimizer and RNG state; completed scorecards and pools are reused on restart. Pool generation interrupted mid-pool must restart that pool.
- Found that ordinary bootstrap duplicates alter k-nearest-neighbour radii. Several PRDC percentile intervals exclude their point estimate. Those intervals are retained exactly as calculated, labelled as unvalidated for superiority claims, and not widened to manufacture agreement.

## Gate result and consequence

**Gate A failed.** The clinical-feature real-real floor passes, as do real-real embedding precision/coverage, the exact-copy privacy controls and the fixed normal bootstrap simulation. Gaussian-noise precision and two SBRAD collapse-coverage checks fail. The four-group judge fallback does not repair an embedding failure. The permitted AE-summary fallback also fails controls.

The instruction says: **“Do not continue to B until A passes.”** Accordingly, no B–G training, tuning or validation evaluation has been launched. The final report must not describe the whole V2 study as completed.

Machine-readable evidence: `results/v2/phaseA_gate.json`, `controls_A8.json`, `scorecard_floor.json`, `measurement_diagnosis.json`, and `final_audit.json`. `phaseA.json` records completion of the requested rescoring, not acceptance of the gate.
