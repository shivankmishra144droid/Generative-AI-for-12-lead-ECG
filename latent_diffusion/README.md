# latent_diffusion

New-file-only implementation of whole-ECG latent diffusion and a common generative benchmark. Start with [REPORT.md](REPORT.md), [PLAN.md](PLAN.md), and the [comparison table](BENCHMARK.html).

All Study 9 code, experiments, checkpoints, audits and reports are in this folder. Models are in `latent/`; the reusable scorecard is `gen_eval.py`. The Markdown report is `REPORT.md`.

Consolidated on 31 October 2026. Imports and reproduction commands use the `latent_diffusion` package. The existing repository's SSSD backbone, real data and baseline weights remain read-only dependencies. The original audit is preserved in `audits/history/`; the final audit checks the consolidated layout.

## Run

Use the existing CUDA-enabled Python 3.14, from the repository root:

```powershell
py -3.14 -m latent_diffusion.run_all --resume
```

The full runner executes preparation, tests, AE/VAE training, diffusion, sampling, controls, ECGSYN, scorecard, fresh utility tests, final audit and report. It attempts the upstream SSSD download; 403 or empty responses become an explicit unavailable row. External archives are never executed or automatically unpacked. Existing hybrid pools are read only.

Individual phases:

```powershell
py -3.14 -m latent_diffusion.data
py -3.14 -m latent_diffusion.train ae --epochs 15 --batch 32
py -3.14 -m latent_diffusion.reconstruction
py -3.14 -m latent_diffusion.train diffusion --steps 10000 --batch 8 --resume
py -3.14 -m latent_diffusion.train ae --name vae --beta .01 --epochs 15 --batch 32
py -3.14 -m latent_diffusion.generate latent_diffusion --n 64
py -3.14 -m latent_diffusion.generate vae --n 64
py -3.14 -m pip install --target latent_diffusion/artifacts/deps --no-deps neurokit2==0.2.13 PyWavelets==1.10.0
py -3.14 -m latent_diffusion.generate ecgsyn --n 64
py -3.14 -m latent_diffusion.generate control --n 16
py -3.14 -m latent_diffusion.published download
py -3.14 -m latent_diffusion.benchmark --n 64 --qt
py -3.14 -m latent_diffusion.utility --n 64 --updates 200 --device cuda
py -3.14 -m latent_diffusion.final_audit
py -3.14 -m latent_diffusion.report
```

Explicit conditions write `artifacts/pools/latent_diffusion_custom.npz` so benchmark pools are preserved:

```powershell
py -3.14 -m latent_diffusion.generate latent_diffusion --label SBRAD --hr 45 --age 60 --sex 0 --n 16
```

Accepting conditions does not establish calibrated control: consult the measured HR sweep in the report. Sex uses the source metadata's 0/1 encoding. ECGSYN is evaluated for sinus rhythms only; it does not provide AFIB, age or sex control.

## Artifacts and review

- `artifacts/`: local, regenerable checkpoints, waveform pools, dependency wheels and caches; git-ignored.
- `results/`: durable measured JSON, CSV and plots. `BENCHMARK.html` renders the comparison table.
- `audits/`: phase evidence and pre-existing file hash comparisons.
- `initial_hashes.json`: pre-work snapshot of existing source, labels and metadata.
- `test_models.py`, `test_metrics.py`, `test_integration.py`: new meaningful model and metric controls.

The test split is guarded out of the Study 9 loader. The validation set has historical selection exposure and the current results are exploratory. A distance-based membership attack is not a privacy guarantee. The current pilot does not prove superiority over the hybrid, demographic fidelity, convergence, or clinical suitability. A published 36×256 model and larger-cloud-GPU training remain alternative experiments.
