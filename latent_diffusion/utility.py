"""Fresh, matched TSTR/TRTR and augmentation pilots; real validation only.

Every condition gets the same optimizer updates and batch size. Real and synthetic
sets are class-balanced with identical n/class. Augmentation uses 50:50 real/synthetic
sampling, so this tests balanced low-data utility, not the earlier scarcity studies.
Published SSSD is excluded because of overlapping model training provenance.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from generator_comparison.tstr import CNNLSTM
from .benchmark import patient_rows, registry
from .common import ART, SEED, audit, load_split, seed_all, write_json


def train_eval(x, y, xv, yv, seed, updates, device='cpu'):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = CNNLSTM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(np.asarray(x, np.float32).copy()).to(device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    model.train()
    for _ in range(updates):
        rows = rng.integers(len(x), size=64)
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(xt[rows]), yt[rows])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite utility loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(torch.as_tensor(np.asarray(xv[i:i+128]).copy(), device=device)).argmax(1).cpu().numpy()
                               for i in range(0, len(xv), 128)])
    return dict(seed=seed, macro_f1=float(f1_score(yv, pred, labels=range(4), average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(yv, pred)),
                per_class_f1=f1_score(yv, pred, labels=range(4), average=None, zero_division=0).tolist(),
                confusion=confusion_matrix(yv, pred, labels=range(4)).tolist())


def run(n=64, updates=200, seeds=(9009, 9010, 9011), device='cpu', resume=False):
    from pathlib import Path
    train, yt = load_split('train'); val, yv = load_split('val')
    meta = np.load(ART / 'train_conditions.npz')
    fit = np.load(ART / 'partition.npz')['fit']
    rows = patient_rows(yt, meta['patient'], fit, n, np.random.default_rng(SEED))
    if any((yt[rows] == c).sum() != n for c in range(4)):
        raise ValueError('Not enough patient-distinct real rows per class')
    real, yr = np.asarray(train[rows]), yt[rows]
    sources = registry()
    sets = {'TRTR': (real, yr)}
    exclusions = {}
    for name, cfg in sources.items():
        if name == 'real_control':
            continue
        if name == 'published_sssd' or cfg['supported'] != list(range(4)):
            exclusions[name] = 'Excluded: incompatible training provenance or unsupported AFIB'
            continue
        path = Path(cfg['path'])
        if not path.exists():
            exclusions[name] = 'Pool unavailable'
            continue
        d = np.load(path, allow_pickle=False)
        y = d['y'].astype(np.int64)
        rng = np.random.default_rng(SEED)
        if any((y == c).sum() < n for c in range(4)):
            exclusions[name] = 'Insufficient class counts'
            continue
        sr = np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False) for c in range(4)])
        sx, sy = d['X'][sr], y[sr]
        sets[f'TSTR_{name}'] = sx, sy
        sets[f'augment_{name}'] = np.r_[real, sx], np.r_[yr, sy]
    results = {}
    protocol = dict(n_per_class=n, updates=updates, batch=64, seeds=list(seeds), device=device,
                    selection='fixed last update, no validation selection', evaluation='all real validation ECGs; no test split',
                    augmentation='50:50 real/synthetic on balanced small real subset; TRTR is the matched resampling control',
                    budgets='identical optimizer updates for all conditions',
                    limitation='Exploratory small-data utility; generators have different training budgets/data sizes')
    if resume and (ART / 'utility.json').exists():
        import json
        old = json.loads((ART / 'utility.json').read_text())
        if old['protocol'] != protocol:
            raise ValueError('Utility resume protocol differs from saved experiment')
        results = old['results']
    for name, (x, y) in sets.items():
        if name in results:
            continue
        runs = []
        for seed in seeds:
            runs.append(train_eval(x, y, val, yv, seed, updates, device))
            print(name, seed, runs[-1]['macro_f1'], flush=True)
        results[name] = dict(runs=runs, macro_f1_mean=float(np.mean([r['macro_f1'] for r in runs])),
                             macro_f1_std=float(np.std([r['macro_f1'] for r in runs], ddof=1)) if len(runs)>1 else None)
        write_json(ART / 'utility.json', dict(protocol=protocol, results=results, exclusions=exclusions))
    audit('phase5_utility', dict(protocol=protocol, results=results, exclusions=exclusions))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.n, args.updates) < 1:
        p.error('n and updates must be positive')
    run(args.n, args.updates, device=args.device, resume=args.resume)
