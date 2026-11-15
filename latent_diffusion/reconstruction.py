"""Clinical compression audit on internal development patients, before generative claims."""
import numpy as np
import torch

from latent_diffusion.gen_eval import clinical_features, feature_distribution, paired_rpeak_check
from .common import ART, SEED, audit, balanced_rows, load_split, seed_all, write_json
from .generate import load_ae, project_leads
from .train import decoder_condition


@torch.no_grad()
def run(n=32):
    seed_all()
    x, y = load_split('train')
    dev = np.load(ART / 'partition.npz')['dev']
    c = np.load(ART / 'train_conditions.npz')['values']
    rows = dev[balanced_rows(y[dev], n, np.random.default_rng(SEED))]
    source = np.asarray(x[rows], np.float32)
    model = load_ae('ae')
    raw = model(torch.from_numpy(source).transpose(1, 2),
                decoder_condition(torch.as_tensor(y[rows]), torch.as_tensor(c[rows])), sample=False)[0].transpose(1, 2).numpy()
    projected = project_leads(raw)
    fr, fs = clinical_features(source), clinical_features(projected)
    result = dict(n=len(rows), source='patient-disjoint internal development, not validation',
                  latent_shape=[16, 125], compression_ratio=6.,
                  raw_mse=float(np.mean((raw-source)**2)), projected_mse=float(np.mean((projected-source)**2)),
                  raw_slope_mae=float(np.mean(np.abs(np.diff(raw, axis=1)-np.diff(source, axis=1)))),
                  projection_mse=float(np.mean((raw-projected)**2)),
                  rpeak=paired_rpeak_check(source, projected),
                  by_class={str(c):feature_distribution(fr[y[rows] == c], fs[y[rows] == c]) for c in range(4)},
                  note='Mean-code reconstruction with true condition; reconstruction fidelity does not establish prior-sample quality')
    np.savez_compressed(ART / 'reconstruction_examples.npz', source=source[:8], raw=raw[:8], projected=projected[:8])
    write_json(ART / 'reconstruction.json', result)
    audit('phase2_clinical_reconstruction', result)
    print({k:v for k,v in result.items() if k != 'by_class'})


if __name__ == '__main__':
    run()
