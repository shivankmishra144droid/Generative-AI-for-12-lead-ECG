# Phase C/D: budget decision (written before the main training runs, 2026-10-31)

- Chosen AE: rung r1 (60 epochs, cosine LR), Gate B1 passed: judge 0.625, groups ≤ 0.62, missed R-peaks 0.9 %
  (`results/v2/ae_ladder.json`). Latent 16×125.
- Measured on the RTX 3050 (profile runs `diffusion_profile_cs`, `diffusion_profile_ds`, not used for results):
  0.124 s/step for 8×64 (state 32) at batch 32, with and without the R-map, including diagnostics and checkpoints.
- **Budget: 60,000 steps for C-S (`cs`) and 60,000 for the R-map model (`ds`)**, about 2.1 h each, 4.1 h total.
  This keeps C+D under the 7 h cap in amendment 1 and leaves time for E-G. 60 k steps × batch 32 = 1.92 M
  samples, about 164 passes over the 11,719 fit ECGs (v1: 10 k × 8 = 80 k samples, about 7 passes).
- C-M (16×128) is not trained tonight. The capacity ladder is therefore a single rung; Gate C reduces to "does C-S
  beat the hybrid / v1 by the registered metrics", reported as such.
- Snapshots at 15 k, 30 k, 45 k, 60 k steps; the sample-quality check per snapshot runs after training, on dev,
  because the GPU lease serialises jobs.
- Evaluation checkpoint for C and D: EMA 0.999 at 60 k (fixed now; the 0.995 EMA and earlier snapshots are reported
  as descriptive curves only).
