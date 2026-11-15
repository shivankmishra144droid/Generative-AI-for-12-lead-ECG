# Study 4, Phases 0–1: results and audit notes (2026-10-29)

Plan: `report/WORKFLOW_4.md`. Literature gate: `report/LITERATURE_4.md` (GO, reframed as an extension of
Pöhl et al. 2025). Everything here uses the train split only. Fitting data comes from the `fit` part and the
detector's real side from the `dev` part; the `judge` part is untouched.

## Split (`_study4_split.py`, `split_summary.json`)
- Patient-level split, seed 4004, 40 / 30 / 30 % of patients.
- Recordings per part (SR / AFIB / STACH / SBRAD):

| part | SR | AFIB | STACH | SBRAD |
|---|---|---|---|---|
| fit | 4,734 | 413 | 229 | 169 |
| dev | 3,455 | 336 | 164 | 136 |
| judge | 3,552 | 307 | 175 | 141 |

- No patient appears in two parts; this is tested in `tests/test_study4.py`.

## Audit finding 1: the detector split must be grouped by patient
- Under protocol v2 as first written, **real vs real** (fit-part ECGs as they are, against dev-part ECGs) gave
  AUC 0.529, and **AFIB 0.578**.
- E, which rebuilds real ECGs from their own parts and so should be 0.5, gave AFIB 0.606.
- **Cause:** the detector's 70/30 split was by recording. 30 % of AFIB recordings belong to patients with more than one
  recording, so the same patient sat on both sides and the detector learned patients, not realism.
- **Fix:** `detect.group_split`. Every patient is kept on one side; synthetic ECGs are grouped by their source
  recording's patient. After the fix, real vs real gave 0.500 and E gave 0.499.
- The study 1 detector (`realism_detector.py`, frozen) has the same flaw. Its ablation AUCs may be slightly high; study
  1–2 conclusions do not depend on them.

## Phase 0 baselines (`baselines/baselines.md`; 5 seeds, mean with 95 % CI)

| variant | full | < 0.5 Hz | 0.5–40 Hz | beats only | between beats |
|---|---|---|---|---|---|
| E: real, rebuilt from its own parts | 0.499 | 0.503 | 0.495 | 0.507 | 0.490 |
| C: Gaussian-process residual | 1.000 | 0.948 | 1.000 | 1.000 | 0.981 |
| D: Gaussian-process wander | 0.888 | 0.977 | 0.743 | 0.808 | 0.807 |
| oracle: real templates + assembly | 1.000 | 0.984 | 1.000 | 1.000 | 0.988 |
| gen: generated templates + assembly | 1.000 | 0.983 | 1.000 | 1.000 | 0.988 |

**Exit checks:**
- E is about 0.5 in every band, so the protocol is sound.
- C reproduces study 1 (1.00).
- D is 0.89, against 0.95 in study 1. The difference is expected from the patient-grouping fix.
- The split is patient-disjoint.

**Reading:**
- The residual is the dominant giveaway: C is at 1.00 in every band except the slowest.
- Wander alone is detectable, almost perfectly in its own band (0.98).
- oracle ≈ gen everywhere. With the current assembly, generated templates add no detectability of their own.

## Phase 1 diagnostics (`diagnostics/diagnostics.json`, figures)

**Rank K** (smallest rank explaining ≥ 80 % of the beat-deviation variance, a rule fixed before any detector run): SR 11,
AFIB 24, STACH 17, SBRAD 8.

**Where the residual lives** (`fig_residual_phase.png`):
- The real residual (d_k = 0) is concentrated at the QRS: a sharp double peak at R ± 30 ms, with power 20–25 against
  about 1 elsewhere.
- The Gaussian-process residual is flat at about 2–4. It puts too little energy on the QRS and too much between beats.
- After the per-beat fit, the QRS peak is gone. The real leftover is about 0.5–1.

**Leftover energy share** (median leftover energy / high-passed energy, sanity check on 40 recordings per class):

| class | d_k = 0 | K = 8 |
|---|---|---|
| SR | 0.20 | 0.08 |
| AFIB | 0.33 | 0.13 |
| STACH | 0.39 | 0.13 |
| SBRAD | 0.23 | 0.11 |

**Beat-to-beat statistics:**
- Lag-1 correlation of the per-beat coefficients across beats: −0.23 to 0.00. Beat variation is **independent from beat
  to beat**, not a slow drift.
- About 74 % of the coefficient variance is within a recording.
- The first component correlates with the preceding RR: SR 0.26, STACH 0.38, AFIB 0.07, SBRAD −0.16.

**Wander, real vs current Gaussian process** (median over classes):
- RMS is similar: 0.24–0.32 against 0.22–0.29.
- Kurtosis of the first difference: **10–15 against 3.0**. Real wander has jumps and bursts.
- SD of the log RMS ratio between the two halves of a recording: **0.66–0.91 against 0.12–0.15**. Real wander is
  non-stationary.
- Cross-lead correlation: about 0.19–0.21 against 0.08–0.12.
- Wander RMS correlates with residual RMS across recordings (0.24–0.37); in the current generator the two are sampled
  independently.

## Audit finding 2: edge beats in the per-beat fit
- The first version fitted coefficients for beats clipped by the recording edge. Their windows are only partly
  observed, so the fit was ill-posed: |a| up to about 120, and a spurious leftover bump appeared.
- **Fix:** clipped beats keep a_k = 0, the ridge was raised from 1e-3 to 0.1, and the first and last 0.5 s are left out
  of the phase profiles.
- A remaining leftover bump at +0.6 to +1.0 s from the preceding R (SR, STACH) holds **0.9 %** of the leftover energy.
  It comes from very few samples (9 at +0.83 s, against 2,113 at 0 s) from recordings with missed R-peaks.
- Figures show it, but it carries no weight. The real leftover is otherwise close to flat outside the QRS (flatness
  values in `diagnostics.json` are dominated by this sparse tail and should not be read alone).

## Targets for Phases 2–3 (frozen in `diagnostics.json`, block `targets`)
- **Per-beat variation:** independent across beats; per-class coefficient variance; coupled to the preceding RR.
- **Wander:** heavy-tailed increments, non-stationary amplitude, cross-lead correlation about 0.2, and coupled to
  residual scale.
- **Leftover:** low and roughly flat; a stationary process is probably enough for it.
