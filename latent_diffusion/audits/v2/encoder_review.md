# Phase A encoder review

The preferred SSL encoder completed 20 epochs on internal fit patients only. Its checkpoint was selected using fixed-augmentation development contrastive loss. This is an embedding training run, not evidence that ECG generation has improved.

- Checked channels-first inputs (12 × 1000), four residual blocks, 64 channels, and a 128-dimensional embedding. The contrastive projection head is discarded at inference.
- Verified the two augmented views use independent crops, scaling, Gaussian noise and per-lead masks. The positive target links each recording only to its other view; self-similarity is excluded from the denominator.
- Confirmed patient-disjoint fit/development indices and no validation array loading.
- Confirmed best-checkpoint selection uses development contrastive loss only. RNG states, optimizer and epoch are saved for resumption. Resume now rejects a changed protocol.
- The first launch stopped before training because Windows lists browser/desktop graphics in the compute-process query. Corrected the V2 guard to exempt only named UI processes, wait on other workloads, and log each check. Original failed-launch logs remain.
- Confirmed the completed run's log, checkpoint and memory evidence live under V2. No V1 encoder or checkpoint was overwritten.
- Noise invariance is an intended augmentation, so this embedding must still pass the preregistered noise control before it can support later gates.

Evidence: `results/v2/encoder_ssl_training.json`, `audits/v2/phaseA_encoder_registration.json`, and `artifacts/v2/logs/encoder_run2.out.log`.
