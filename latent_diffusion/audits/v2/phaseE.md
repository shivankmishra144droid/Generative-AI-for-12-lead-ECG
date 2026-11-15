# Phase E audit: sampler (dev, cs, 63/class)

- **Grid:** reduced, as fixed in amendment 1: CFG {1, 2, 3} × DDIM {50, 100}, η 0.
- **Rule (written in `dev_eval_v2.grid` before running):** mean rank over the Gate C metrics; settings are eligible if
  emb_task coverage ≥ 0.8 × best. All 6 were eligible.
- **Chosen:** CFG 2 / DDIM 50 (rank sum 4). Frozen in `artifacts/v2/sampling.json` at 07:15 IST, before Phase F ran.
- **Files:** `results/v2/sampler_sweep.json`, `results/v2/sampler_heatmap.png`.
