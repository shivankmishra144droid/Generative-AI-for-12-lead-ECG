"""Generate Study 9 VAE/diffusion/ECGSYN pools using training-only conditions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from latent_diffusion.latent.vae import ECGVAE
from latent_diffusion.latent.latent_diffusion import LatentSSSD, LatentDiffusion
from synthetic_generation.leads import reconstruct_12, to_independent_8
from .common import ART, DATA, SEED, audit, digest, load_split, seed_all, write_json
from .data import condition_values
from .train import decoder_condition, device_for


def optional_dependencies():
    deps = ART / 'deps'
    if deps.exists() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))


def load_ae(name='ae', device='cpu'):
    path = ART / name / 'best.pt'
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ECGVAE(**ck['config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(ck['model'])
    return model


def training_conditions(n, seed):
    _, y = load_split('train')
    fit = np.load(ART / 'partition.npz')['fit']
    cond = np.load(ART / 'train_conditions.npz')['values']
    rng = np.random.default_rng(seed)
    rows = np.concatenate([rng.choice(fit[y[fit] == c], n, replace=True) for c in range(4)])
    return y[rows], cond[rows]


def project_leads(x):
    # Limb identities hold in mV, not standardized coordinates. Keep raw output too.
    return reconstruct_12(to_independent_8(np.asarray(x, np.float32)))


@torch.no_grad()
def neural_pool(kind, n=64, sampling_steps=50, guidance=2., seed=SEED, diffusion_dir='diffusion',
                label=None, hr=None, age=None, sex=None):
    seed_all(seed)
    device = device_for('auto')
    y, cond = training_conditions(n, seed)
    custom = label is not None or any(v is not None for v in (hr, age, sex))
    if label is not None:
        keep = y == label
        y, cond = y[keep], cond[keep]
    for column, value, center, scale, lo, hi in ((0, hr, 75., 40., 20., 250.),
                                                (1, age, 50., 25., 0., 100.),
                                                (2, sex, .5, .5, 0., 1.)):
        if value is not None:
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError('Requested condition is outside supported physical range')
            if column == 2 and value not in (0, 1):
                raise ValueError('Sex must use the source binary code 0 or 1')
            cond[:, column] = (value-center)/scale
            cond[:, column+3] = 1
    ae = load_ae('vae' if kind == 'vae' else 'ae', device)
    if kind == 'latent_diffusion':
        path = ART / diffusion_dir / 'last.pt'
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck['ae_sha256'] != digest(ART / 'ae' / 'best.pt'):
            raise ValueError('Autoencoder changed after diffusion training')
        diffusion = LatentDiffusion(LatentSSSD(**ck['config'])).to(device)
        diffusion.load_state_dict(ck['ema'])
        diffusion.eval()
    output = []
    for start in range(0, len(y), 8):
        labels, values = torch.as_tensor(y[start:start+8], device=device), torch.as_tensor(cond[start:start+8], device=device)
        if kind == 'latent_diffusion':
            z = diffusion.sample(labels, values, sampling_steps, guidance, seed+start)
            z = z*ck['std'].to(device) + ck['mean'].to(device)
        else:
            z = torch.randn((len(labels), 16, 125), device=device)
        output.append(ae.decode(z, decoder_condition(labels, values)).transpose(1, 2).cpu().numpy())
    raw = np.concatenate(output)
    if not np.isfinite(raw).all():
        raise FloatingPointError('Non-finite generated ECG')
    projected = project_leads(raw)
    result = dict(kind=kind, n_per_class=n, seed=seed, sampling_steps=sampling_steps if kind != 'vae' else None,
                  guidance=guidance if kind != 'vae' else None, projection_mse=float(np.mean((projected-raw)**2)),
                  requested=dict(label=label, hr=hr, age=age, sex=sex),
                  conditioning='empirical joint class/HR/age/sex from internal fit, with explicit user overrides; no validation conditions')
    if kind == 'latent_diffusion':
        result.update(checkpoint_step=ck['step'], checkpoint_config=ck['config'], ae_sha256=ck['ae_sha256'])
    else:
        result['vae_sha256'] = digest(ART / 'vae/best.pt')
    path = ART / 'pools' / f'{kind}{"_custom" if custom else ""}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=projected, raw=raw, y=y, conditions=cond)
    write_json(path.with_suffix('.json'), result)
    audit(f'phase5_generate_{kind}{"_custom" if custom else ""}', result)
    return path


def ecgsyn_pool(n=64, seed=SEED):
    optional_dependencies()
    import neurokit2 as nk
    from .ecgsyn_adapter import simulate
    import json
    stats = json.loads((DATA / 'normalization_stats.json').read_text())
    order = stats['lead_order']
    y, cond = training_conditions(n, seed)
    output, labels, values = [], [], []
    # ECGSYN is a sinus-rhythm simulator. Never label an irregular sinus trace as AFIB.
    for i, (label, cv) in enumerate(zip(y, cond)):
        if label == 1:
            continue
        hr = float(cv[0]*40+75) if cv[3] else {0:70., 2:110., 3:45.}[int(label)]
        data = simulate(duration=10, sampling_rate=100, heart_rate=float(np.clip(hr, 25, 200)),
                               heart_rate_std=1., noise=.01, method='multileads', random_state=seed+i)
        x = (data[order].to_numpy()[:1000]-stats['mean'])/stats['std']
        output.append(x); labels.append(label); values.append(cv)
        if len(output) % 16 == 0:
            print('ECGSYN records generated', len(output), flush=True)
    path = ART / 'pools/ecgsyn.npz'; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=project_leads(np.asarray(output, np.float32)), y=np.asarray(labels), conditions=np.asarray(values))
    info = dict(neurokit2=nk.__version__, supported_classes=['SR', 'STACH', 'SBRAD'],
                excluded_class='AFIB: ECGSYN is a sinus model', n_per_class=n, seed=seed,
                rr_interpolation='NeuroKit supported NumPy interpolation via local function clones; positive-RR guard',
                note='Multilead simulator output treated as mV, then train-normalized and limb-projected; no data-driven amplitude fitting')
    write_json(path.with_suffix('.json'), info)
    audit('phase5_generate_ecgsyn', info)
    return path


@torch.no_grad()
def control_sweep(n=16, sampling_steps=50):
    device = device_for('auto')
    ck = torch.load(ART / 'diffusion/last.pt', map_location=device, weights_only=False)
    model = LatentDiffusion(LatentSSSD(**ck['config'])).to(device)
    model.load_state_dict(ck['ema']); model.eval()
    ae = load_ae('ae', device)
    from .data import measure_hr
    rows = []
    for label in (0, 2, 3):
        for target in (45., 70., 110.):
            values = torch.as_tensor(condition_values([target]*n, [60.]*n, [0.]*n), device=device)
            labels = torch.full((n,), label, device=device, dtype=torch.long)
            z = model.sample(labels, values, sampling_steps, 2., SEED+100)
            x = ae.decode(z*ck['std'].to(device)+ck['mean'].to(device), decoder_condition(labels, values))
            x = project_leads(x.transpose(1, 2).cpu().numpy())
            hr = np.asarray([measure_hr(a) for a in x])
            rows.append(dict(label=int(label), requested_hr=target, measured_hr=hr,
                             measured_median=float(np.nanmedian(hr)), valid_fraction=float(np.isfinite(hr).mean())))
    result = dict(rows=rows, note='Matched seeds across HR settings; conflicting class/HR requests are stress tests; no success assumed')
    write_json(ART / 'control_sweep.json', result)
    audit('phase5_control_sweep', result)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('kind', choices=['vae', 'latent_diffusion', 'ecgsyn', 'control'])
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--sampling-steps', type=int, default=50)
    p.add_argument('--guidance', type=float, default=2.)
    p.add_argument('--label', choices=['SR', 'AFIB', 'STACH', 'SBRAD'])
    p.add_argument('--hr', type=float)
    p.add_argument('--age', type=float)
    p.add_argument('--sex', type=int, choices=[0, 1])
    args = p.parse_args()
    if args.n < 1:
        p.error('n must be positive')
    if args.kind == 'control':
        control_sweep(args.n, args.sampling_steps)
    elif args.kind == 'ecgsyn':
        ecgsyn_pool(args.n)
    else:
        neural_pool(args.kind, args.n, args.sampling_steps, args.guidance,
                    label=['SR','AFIB','STACH','SBRAD'].index(args.label) if args.label is not None else None,
                    hr=args.hr, age=args.age, sex=args.sex)


if __name__ == '__main__':
    main()
