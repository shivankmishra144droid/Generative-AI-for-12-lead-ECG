# Phase D audit: R-peak-map conditioning (dev, under amendment 1)

- **Model:** `ds` = same 8×64 latent SSSD as `cs`, plus 2 input channels (R-map, validity flag) at latent resolution.
  Map dropout 0.1, vector dropout 0.1, both dropped 0.05 (amendment 1). 60 k steps, 132 min, 562 MiB.
- **Training targets:** R-peaks from the heuristic detector on train ECGs, 98.7 % plausible (`artifacts/v2/rpeaks_train.npz`).
- **Gate D** (fixed in the instructions, evaluated as written in `dev_eval_v2.sweep`): **FAILED**.
  - SR sweep slope 0.868, R² 0.908. STACH slope 0.873, R² 0.894. Both are outside 0.9–1.1 / ≥ 0.9.
  - Every other condition passed: missed 4.5 %, extra 0.19 %/beat, rhythm judge 0.624, full judge 0.983 ≤ 0.987.
- **Consequence:** `cs` stays the main generator (`artifacts/v2/sampling.json`, `main: cs`).
- **Descriptive:**
  - Medians match the request (SR to 110 bpm; STACH and SBRAD to 150).
  - The R-map raises QRS amplitude (`tpl_p2p` 8.1 vs real 5.2) and worsens the morphology and global judges.
- **Not done:**
  - The `paired_dev` source (D3 #1).
  - HR-error median and P90 (D4).
  - v1 in the sweep.
  - The instruction's file name `latent/latent_diffusion_rhythm.py` maps to `latent/latent_diffusion_v2.py` (`rhythm=True`).
- **Files:** `results/v2/rhythm_control.json`, `results/v2/gate_CD.json`, `results/v2/rhythm_sweep.png`.
