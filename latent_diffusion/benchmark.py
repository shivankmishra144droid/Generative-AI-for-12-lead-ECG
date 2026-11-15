"""Matched generative scorecard on real validation patients, never X_test.

Outputs a JSON scorecard and a CSV summary (all raw feature distances retained).
python -m latent_diffusion.benchmark --n 64 --qt
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from latent_diffusion.gen_eval import ECGEncoder, score_pool
from .common import ART, HOME, ROOT, SEED, audit, digest, load_split, seed_all, write_json
from .generate import optional_dependencies


def patient_rows(y, patients, candidates, n, rng):
    result = []
    for c in range(4):
        eligible = rng.permutation(candidates[y[candidates] == c])
        seen, kept = set(), []
        for i in eligible:
            if patients[i] not in seen:
                kept.append(i); seen.add(patients[i])
            if len(kept) == n:
                break
        result.extend(kept)
    return np.asarray(result, dtype=np.int64)


def registry():
    published = ART / 'pools/published_sssd.npz'
    return {
        'real_control': dict(kind='control', supported=list(range(4)), privacy=False),
        'hybrid': dict(path=str(ROOT / 'generator_comparison/tstr/pool_beatgen.npz'), supported=list(range(4)), privacy=True),
        'vae': dict(path=str(ART / 'pools/vae.npz'), supported=list(range(4)), privacy=True),
        'latent_diffusion': dict(path=str(ART / 'pools/latent_diffusion.npz'), supported=list(range(4)), privacy=True),
        'ecgsyn': dict(path=str(ART / 'pools/ecgsyn.npz'), supported=[0, 2, 3], privacy=False),
        'published_sssd': dict(path=str(published), supported=list(range(4)), privacy=False,
                               exclusion='PTB-XL fold-trained model overlaps random train/val/test split; descriptive realism only')}


def benchmark(n=64, include_qt=False):
    seed_all()
    optional_dependencies()
    train, yt = load_split('train'); val, yv = load_split('val')
    tmeta = np.load(ART / 'train_conditions.npz'); vmeta = np.load(ART / 'val_conditions.npz')
    fit = np.load(ART / 'partition.npz')['fit']
    dev = np.load(ART / 'partition.npz')['dev']
    rng = np.random.default_rng(SEED)
    tr = patient_rows(yt, tmeta['patient'], fit, n, rng)
    vr = patient_rows(yv, vmeta['patient'], np.arange(len(val)), n, rng)
    cr = patient_rows(yt, tmeta['patient'], dev, n, rng)
    encoder = ECGEncoder()
    manifests, results = registry(), {}
    summary = []
    for name, source in manifests.items():
        if source.get('kind') == 'control':
            xs, ys = np.asarray(train[cr]), yt[cr]
            synthetic_groups = np.asarray([f'p:{p}' for p in tmeta['patient'][cr]])
        else:
            from pathlib import Path
            path = Path(source['path'])
            if not path.exists():
                source['status'] = 'unavailable: pool not generated/imported'
                continue
            with np.load(path, allow_pickle=False) as data:
                xs, ys = data['X'], data['y'].astype(np.int64)
            synthetic_groups = None
            source['sha256'] = digest(path)
        sr, rr, tt = [], [], []
        for c in source['supported']:
            a, b, d = np.flatnonzero(ys == c), vr[yv[vr] == c], tr[yt[tr] == c]
            count = min(n, len(a), len(b), len(d))
            if count < 6:
                continue
            sr.extend(rng.choice(a, count, replace=False)); rr.extend(b[:count]); tt.extend(d[:count])
        if not sr:
            source['status'] = 'unavailable: fewer than six rows per supported class'
            continue
        sr, rr, tt = map(np.asarray, (sr, rr, tt))
        # Shared patient IDs stay shared when a source is an internal real control.
        gs = synthetic_groups[sr] if synthetic_groups is not None else None
        result = score_pool(np.asarray(val[rr]), yv[rr], xs[sr], ys[sr], np.asarray(train[tt]), yt[tt], encoder,
                             np.asarray([f'p:{p}' for p in vmeta['patient'][rr]]),
                             tmeta['patient'][tt], gs, source['privacy'], include_qt)
        source.update(status='scored', n=len(sr), labels=sorted(np.unique(ys[sr]).tolist()))
        results[name] = result
        write_json(ART / 'scorecard.json', dict(protocol=protocol(n), sources=manifests, results=results))
        for label, values in result['by_class'].items():
            row = dict(generator=name, rhythm=label, n=values['prdc'].get('n_synthetic'),
                       **{k:values['prdc'].get(k) for k in ['precision', 'recall', 'density', 'coverage']},
                       hr_wasserstein=values['features']['hr'].get('wasserstein'),
                       sdnn_wasserstein=values['features']['sdnn_ms'].get('wasserstein'),
                       qrs_width_wasserstein_ms=values['features']['qrs_width_ms'].get('wasserstein'),
                       peak_detectability_fraction=result['peak_detectability']['synthetic_fraction_ge3_peaks'],
                       qt_proxy_median_ms=result.get('qt_synthetic', {}).get('qt_ms_median'),
                       judge_auc=result['feature_judge']['auc_mean'],
                       membership_auc=values.get('membership', {}).get('auc'),
                       waveform_nn_ratio=values.get('waveform_privacy', {}).get('ratio_of_medians'))
            summary.append(row)
        print(name, 'scored', len(sr), 'feature judge', result['feature_judge']['auc_mean'], flush=True)
    write_json(ART / 'scorecard.json', dict(protocol=protocol(n), sources=manifests, results=results))
    if summary:
        with (ART / 'scorecard_summary.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    audit('phase4_5_scorecard', dict(sources=manifests, protocol=protocol(n)))


def protocol(n):
    return dict(seed=SEED, maximum_n_per_class=n, reference='real validation, one recording per patient within rhythm',
                privacy_reference='internal fit subset, one recording per patient within rhythm, same n per class',
                embeddings='frozen baseline CNN-LSTM last hidden, 64 dimensions, standardized using fit subset only',
                encoder_sha256=digest(ROOT / 'baseline/cnn_lstm_baseline.pt'),
                controls='internal development real, independent of fit and validation patients',
                uncertainty='small exploratory per-class samples; PRDC has no confidence interval; judge uses 3 grouped splits',
                fairness='same real rows for each rhythm and count; ECGSYN lacks AFIB; compare per rhythm, not pooled aggregate',
                provenance='historical hybrid trained on full train; new models use internal fit; budgets are not matched',
                test_split='never loaded')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--qt', action='store_true')
    args = p.parse_args()
    if args.n < 6:
        p.error('PRDC needs at least six rows per class')
    benchmark(args.n, args.qt)
