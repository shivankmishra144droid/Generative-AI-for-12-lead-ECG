# Phase B audit: autoencoder reconstruction ceiling (under amendment 1)

Gate B1 (fixed in the instructions): paired reconstructions vs real_full have a full clinical judge AUC ≤ 0.65, every
feature-group judge ≤ 0.75, and an R-peak miss rate ≤ 3 %.

| AE | full judge (SD over 5 grouped splits) | rhythm | morphology | noise_wander | global | missed R | B1 |
|---|---|---|---|---|---|---|---|
| v1 AE (frozen Study 9 v1) | 0.718 (0.025) | 0.508 | — | 0.724 | — | 1.5 % | FAIL (full) |
| r1 (`ae_ladder_v2`, 60 epochs, cosine LR 1e-3 → 1e-5) | 0.625 (0.025) | 0.517 | 0.524 | 0.621 | 0.526 | 0.9 % | PASS |

- **v1 failure cause:** the top permutation importances are `wander_edge_inc_ratio` (AUC drop 0.134) and `wander_rms`
  (0.093). The AE smooths baseline wander, so wander RMS comes out about 20 % low.
- **r1:** same architecture and the same parameter count as v1 (tested in `test_models_v2.py`), trained longer.
  Dev MSE 0.0183; 224 s on GPU. The same wander features still lead, but their AUC drops are roughly halved
  (0.075 and 0.059).
- **Chosen AE:** r1, sha256 `0cf75ef3…` (`artifacts/v2/chosen_ae.json`). The rule is the first passing rung. Rungs
  r2–r4 and B2 were not run because r1 passed.
- **Reading:** 0.625 is the **ceiling** for anything decoded by this AE, not a generator score. Generated samples can
  only score higher (worse) than this.

Files: `results/v2/ae_ceiling_v1ae.json`, `results/v2/ae_ceiling_ae_r1.json`, `results/v2/ae_ladder.json`.
