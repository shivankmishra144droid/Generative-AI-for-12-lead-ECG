# Phase F audit: validation benchmark (run once)

- **Counts:** reference 256/224/125/96; floor halves 125/99/55/47. All pools are at matched counts.
- **Pool settings:**
  - `latent_v2` (cs) uses the frozen Phase E setting.
  - The R-map pools use CFG 2 / DDIM 50.
  - Main sampling seeds are 9630–9632 (judge only).
- **Decision (`phase_f_v2.decide`, amended headline rule, emb_task coverage): not distinguishable.**
  - Main judge 0.961 ± 0.002 vs hybrid 0.965 ± 0.005.
  - Coverage is not worse in 4/4 classes; HR-W is not worse in 3/4.
- **Post-run fix (presentation only):** `missed_rpeaks` for `latent_v2` (cs) showed 0.851. It compared detected peaks
  with peaks the unconditioned model never received. It is now blanked in `benchmark_val.csv/.json/.html`, and
  `phase_f_v2.py` only computes adherence for R-map pools. No scored metric changed.
- **Privacy:** emb_task membership AUC is 0.46–0.55 for all pools, all CIs include 0.5, and there are 0 exact copies. Raw-space membership AUC is pool-blind (SR 0.53–0.54 for every pool). The cs
  distance ratio below 1 is explained by typicality, not membership (see REPORT_V2 §8).
- **Published SSSD-ECG:** not available, and not downloaded.
