# Phase G audit: utility (exploratory)

- **Split use:** fit patients train, dev selects the checkpoint and fits the offsets, validation is scored once per run.
- **Seeds:** 9100–9104, paired across arms.
- **Pools:**
  - `G_latent_v2.npz` (cs, seed 9820, rr_bank peaks 9800).
  - `G_latent_v2_rhythm_model.npz` (ds, hybrid RhythmModel, 9810).
  - Hybrid minority from `generator_comparison/pools2/pool_1.npz`.
  - Hybrid SR `G_hybrid_sr.npz`, generated with the frozen `_beat_gen_pool.py`, seed 9800 (1,993 SR).
- **Protocol deviation:** the `latent_v2_rhythm_model` arm ran although Gate D failed (the script checked that the
  checkpoint existed, not the gate). It is reported as out of protocol and kept out of the G5 answers.
- **File names:** the instructions name `utility_v2.py` / `utility.json`; the actual files are `phase_g_v2.py` /
  `results/v2/utility_G.json` and `utility_G_summary.json`.
- **Wall time:** 9 min for pools, 30 min for 35 classifier runs.
