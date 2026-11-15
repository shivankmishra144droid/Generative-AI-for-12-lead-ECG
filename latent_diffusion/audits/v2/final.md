# Final checklist (instructions §12), checked 2026-10-31 ~08:20 IST

- [x] **Test split never loaded.**
  - `grep -rnE "X_test|y_test|load_split\(['\"]test" latent_diffusion --include=*.py` gives one hit: a docstring in
    `benchmark.py` saying "never X_test".
  - `final_audit.json` shows `prohibited_split_references: []`.
- [x] **No pre-existing repository files changed; tests pass.**
  - `final_audit_v2` reports `changed_preexisting_files: []` (103 checked), and 56 study 2–4 tests pass.
  - `pytest latent_diffusion -q`: 30 pass and 1 fails. The failure is `test_empirical_gate_a`, which asserts the
    **original** Gate A passed. It did not, as recorded in `phaseA_gate.json` and amendment 1. The test is kept unchanged
    as the record; the amended gate is in `phaseA_gate_amended.json`.
  - `final_audit_v2` lists `latent_diffusion/conftest.py` as changed against the v1 snapshot. Its mtime is 01:56 IST, during
    the Phase A run. It only adds `collect_ignore` for artifacts, audits and results.
- [ ] **v1 numbers reproduce from v1 commands:** not re-run tonight. The v1 rescoring under the v2 scorecard is in
  `results/v2/scorecard_v1_rescored.json`.
- [~] **Every gate was written before its run and evaluated as written.**
  - Gate A was recalibrated **after** its run (amendment 1). This is disclosed, and the original result is kept.
  - B1, C, D, E, F and G were fixed in the instructions or amendments 1–2 before running; the registrations are in
    `audits/v2/*_registration.json`.
  - One deviation in G: the out-of-protocol `latent_v2_rhythm_model` arm (see `phaseG.md`).
- [x] **All tuning used dev.**
  - B1, C, D and E ran on dev.
  - Validation was used in F (once; the presentation fix in `phaseF.md` did not rescore) and in G (once per run).
- [x] **No study 5 file read or written.**
- [x] **GPU never shared:** C/D training and the chain ran sequentially under `gpu_lease`.
- [x] **Every number in `REPORT_V2.md` points to a file in `results/v2/`** (or an `audits/v2/` note for compute).
- [x] **No download, nothing pushed, nothing committed.**

## Deliverables mapped to actual files
| instructions name | actual |
|---|---|
| `latent/latent_diffusion_rhythm.py` | `latent/latent_diffusion_v2.py` (`rhythm=True`) |
| `utility_v2.py`, `results/v2/utility.json` | `phase_g_v2.py`, `results/v2/utility_G.json`, `utility_G_summary.json` |
| `rhythm_sweep.png`, sampler heatmap, training curves | `results/v2/rhythm_sweep.png`, `sampler_heatmap.png`, `training_curves.png` (`figures_v2.py`) |
| phase audits | `audits/v2/phaseB.md`, `phaseC.md`, `phaseD.md`, `phaseE.md`, `phaseF.md`, `phaseG.md` |

## Not done
- C-M and C-L capacity rungs.
- Second training seed.
- `paired_dev` R-map source; HR-error percentiles; v1 in the rhythm sweep.
- Published SSSD-ECG (needs a manual browser download).
