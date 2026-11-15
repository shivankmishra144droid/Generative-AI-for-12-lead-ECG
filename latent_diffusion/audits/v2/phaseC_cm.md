# Phase C, C-M rung: step-time measurement (amendment 4), recorded before the main run

- **Hardware:** Google Colab, Tesla T4 (free tier), Python 3.13. The S4 Cauchy kernel uses pykeops (`pykeops OK`;
  no "slow Cauchy kernel" fallback).
- **Profile run:** `cm_profile`, 16 × 128, state 64, batch 32, 1,000 steps. Time **5.1 min**, including the one-time
  pykeops compile and console output from the deprecation warnings. Dev loss at step 1,000 was 0.9855 (`cs` at step
  1,000: 0.998).
- **Projection:** 100 k steps ≈ 510 min ≈ 8.5 h, under the 12 h cap in §C2. **Main run: 100,000 steps.**
- **Colab-specific code paths:**
  - `V2_SINGLE_GPU_HOST=1` skips the foreign-process check in `gpu_lease`. The Colab VM listed a non-job PID; the
    lock file is still used.
  - The AE fingerprint is checked on `ae_r1/best.pt` instead of the stored Windows path.
- Free-tier disconnects are expected. Training resumes from `last.pt` on Drive, which is saved every 1,000 steps.
