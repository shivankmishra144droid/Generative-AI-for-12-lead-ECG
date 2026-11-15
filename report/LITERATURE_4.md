# Study 4, Phase 0: literature check (2026-10-29)

**Gate question (WORKFLOW_4.md, Phase 0):** has a published method already done "diffusion beats + modelled per-beat
variation + learned/realistic residual", and evaluated it with a real-vs-synthetic detector?

**Answer: no, but the base generator is not new.** We reframe study 4 as an extension of published beat-assembly
generators, not as a new generator.

## Closest prior work

| work | what it does | per-beat variation | noise / wander / TP segment | realism test |
|---|---|---|---|---|
| **Pöhl et al. 2025, "Generating Realistic Multi-Beat ECG Signals"** ([arXiv 2505.18189](https://arxiv.org/abs/2505.18189)) | diffusion model for single beats (MIT-BIH SVDB, single lead), a separate time-series model for inter-beat features (RR, P/Q/T intervals and amplitudes), beats picked from a store by feature matching and stitched with cross-fading | only through which stored beat is picked; beats are sampled independently, no drift or conditioning on the previous beat | **none**: no residual, wander or noise model; the TP stretch is only cross-faded | no detector; beat- and feature-level distribution distances plus train-on-synthetic accuracy |
| **SE-Diff 2025** ([arXiv 2511.09895](https://arxiv.org/abs/2511.09895)) | latent diffusion over whole 10 s 12-lead ECGs from text; an ECGSYN-style ODE simulator and a beat decoder used **only as training regularisers** | not modelled explicitly | simulator baseline term used only in training | relative FID and downstream metrics, no detector |
| MIDT-ECG 2025 ([arXiv 2510.05492](https://arxiv.org/abs/2510.05492)), DiffuSETS 2025 ([arXiv 2501.05932](https://arxiv.org/abs/2501.05932)), SSSD-ECG 2023 | whole-ECG diffusion | implicit | implicit | FID-style, downstream, privacy (MIR/NNAA) |
| ECGSYN, McSharry et al. 2003 ([PhysioNet](https://physionet.org/content/ecgsyn/1.0.0/)) | ODE model with parametric beat-to-beat variation in morphology and timing (respiratory sinus arrhythmia, QT–rate dependence, R-amplitude modulation) | parametric, not learned | parametric noise | none |
| MultiODE-GAN 2024 ([arXiv 2409.17833](https://arxiv.org/abs/2409.17833)) | GAN for 12-lead ECG constrained by heart-simulator ODEs | via the ODE | — | downstream |
| Adib et al. 2022 ([arXiv 2202.00569](https://arxiv.org/abs/2202.00569)) | AC-WGAN-GP single beats for MIT-BIH imbalance | single beats only | — | downstream |

Related evaluation practice:
- **Human Turing tests:** GenECG ([PMC12142132](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12142132/)).
- **Noise ablations with AUC:** synthetic-ECG training ([arXiv 2111.06175](https://arxiv.org/abs/2111.06175)).
- **MMD fidelity with added respiratory wander, EMG and mains noise:** [arXiv 2506.23259](https://arxiv.org/abs/2506.23259).
- **Cyclostationary ECG analysis:** used for denoising only, not for generation ([CinC 2007](https://www.cinc.org/old/Proceedings/2007/pdf/0093.pdf)).

## What this means for study 4

- **Not novel:** "diffusion beats + separate rhythm model, assembled" (Pöhl et al. 2025 did this first, May 2025, single
  lead). The report must cite it and describe our generator as a 12-lead variant of that idea.
- **Open, and what study 4 does:**
  1. *Diagnosing* why beat-assembly ECGs are detectable. Our ablations show the residual is beat-locked, which nobody
     has reported.
  2. Modelling **correlated beat-to-beat variation** and a **beat-locked (cyclostationary) residual plus wander** for
     12-lead assembly.
  3. Evaluating with a **patient-grouped real-vs-synthetic detector** and a held-out judge detector, plus the 2 × 2
     ablation.
  4. Testing whether realism gains translate into usefulness.
- **Decision: GO**, framed as "closing the realism gap of beat-assembly ECG generators".

## Caveats on this search

- Searches were web searches on 2026-10-29, not a systematic review.
- The two closest papers were read in their HTML versions: Pöhl et al. up to Section IV-E plus the appendix, SE-Diff up
  to Section 3.5. Their later tables were not checked verbatim.
- A systematic search (Google Scholar and arXiv on "beat-to-beat variability" + "synthetic ECG" + "diffusion/GAN") is
  still needed before any paper is written.
