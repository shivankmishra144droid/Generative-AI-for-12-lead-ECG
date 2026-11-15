# Synthetic 12-Lead ECG Generation for Rhythm Classification (PTB-XL)

Generative models for 10-second, 12-lead ECGs of four rhythm classes (sinus rhythm, AFIB, sinus tachycardia, sinus
bradycardia), and a study of whether synthetic ECGs help a CNN-LSTM rhythm classifier.

- Full report: [`report/report.pdf`](report/report.pdf) · one-page summary: [`report/summary.pdf`](report/summary.pdf)
- Demo, no dataset needed: [`report/06_Demo_Generate_ECGs.ipynb`](report/06_Demo_Generate_ECGs.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivankmishra144droid/Quality-Aware-Synthetic-Data-Augmentation-for-ECG-Classification/blob/main/report/06_Demo_Generate_ECGs.ipynb)

## What has been done

- **Generator 1, beat stitching.** Learned beat templates are assembled into rhythms. A quality score
  ranks the synthetic ECGs: classifiers trained on the top-scored ECGs reach 0.743 balanced accuracy on real ECGs,
  random ones 0.672 and bottom-scored ones 0.582.
- **Augmentation studies.**
  - Selecting synthetic ECGs by quality does not beat random selection.
  - Synthetic data raises untuned macro-F1, but the gain is a decision-threshold effect: tuned thresholds match it,
    and AUPRC does not improve.
- **Realism analysis.** A feature-based real-vs-synthetic judge shows what gives synthetic ECGs away (residual noise,
  beat-to-beat variation, baseline wander).
- **Generator 2, latent diffusion.** An autoencoder plus an S4 diffusion model, scaled up and trained on Colab and
  Kaggle. On held-out validation patients, the real-vs-synthetic judge AUC is **0.796** for the latent diffusion
  model and **0.965** for beat stitching; 0.5 means indistinguishable from real. No training ECGs were copied.

## What more needs to be done

- A second training seed for the diffusion model, and a second dataset.
- A head-to-head comparison with published ECG generators (for example SSSD-ECG).
- Better synthesis of AFIB's fibrillatory baseline.

## How to download and run

```bash
git clone https://github.com/shivankmishra144droid/Quality-Aware-Synthetic-Data-Augmentation-for-ECG-Classification.git
cd Quality-Aware-Synthetic-Data-Augmentation-for-ECG-Classification
pip install -r requirements.txt          # install a CUDA build of torch first (see the file)
```

1. Download PTB-XL 1.0.1 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/). Unzip it into
   `archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/`.
2. Run `notebooks/02_PTBXL_Exploration.ipynb` to build `processed_data/X_*.npy`. Labels and splits are already
   included.
3. Generate ECGs with the trained latent diffusion model. The checkpoints are already in the repository. The scripts
   behind Studies 1–4 are in [`studies/`](studies/README.md).

```python
from latent_diffusion.sample_v2 import Generator, fit_requests
from latent_diffusion.s4_kernel import conj_kernel
import torch
labels, conditions, rpeaks, _ = fit_requests([8, 8, 8, 8], seed=0)   # 8 ECGs per class
with conj_kernel():                                                  # required for this checkpoint
    gen = Generator('cm', torch.device('cuda'), 'step100000.pt')
    ecgs = gen(labels, conditions, rpeaks)                           # (32, 1000, 12) at 100 Hz
```

Author: Shivank Mishra
