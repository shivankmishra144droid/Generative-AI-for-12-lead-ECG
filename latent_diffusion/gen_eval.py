"""Study 9 common generative scorecard. Pure metrics: no training/output side effects.

Embeddings use the frozen real-only ECG CNN-LSTM, before its classification head.
PRDC follows Naeem et al. (2020); no image-FID or detector-AUC proxy is used.
"""
from __future__ import annotations

import warnings
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits

from latent_diffusion.common import CLASSES, ROOT


def validate_ecgs(x):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (1000, 12) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite, nonempty (N,1000,12) standardized ECGs at 100 Hz')
    return x


class ECGEncoder:
    def __init__(self):
        from generator_comparison.tstr import CNNLSTM
        self.model = CNNLSTM().eval().requires_grad_(False)
        checkpoint = torch.load(ROOT / 'baseline/cnn_lstm_baseline.pt', map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    @torch.no_grad()
    def __call__(self, x):
        validate_ecgs(x)
        result = []
        for start in range(0, len(x), 64):
            a = torch.from_numpy(np.asarray(x[start:start+64], np.float32).copy())
            a = self.model.cnn(a.transpose(1, 2)).transpose(1, 2)
            h, _ = self.model.lstm(a)
            result.append(h[:, -1].numpy())
        return np.concatenate(result)


def _matrix(a, minimum=1):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or len(a) < minimum or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError('Expected finite embedding matrix with enough rows')
    return a


def prdc(real, synthetic, k=5):
    real, synthetic = _matrix(real, k+1), _matrix(synthetic, k+1)
    if real.shape[1] != synthetic.shape[1] or k < 1:
        raise ValueError('Invalid PRDC dimensions or k')
    rr, ss = cdist(real, real), cdist(synthetic, synthetic)
    np.fill_diagonal(rr, np.inf); np.fill_diagonal(ss, np.inf)
    r = np.partition(rr, k-1, axis=1)[:, k-1]
    s = np.partition(ss, k-1, axis=1)[:, k-1]
    cross = cdist(real, synthetic)
    return dict(precision=float((cross <= r[:, None]).any(0).mean()),
                recall=float((cross <= s[None, :]).any(1).mean()),
                density=float((cross <= r[:, None]).sum(0).mean()/k),
                coverage=float((cross.min(1) <= r).mean()), k=k,
                n_real=len(real), n_synthetic=len(synthetic))


def nearest_distance(query, reference, block=128):
    """Full waveform RMS or embedding Euclidean / sqrt(dim), bounded query memory."""
    q, r = _matrix(query), _matrix(reference)
    if q.shape[1] != r.shape[1]:
        raise ValueError('Distance dimensions differ')
    return np.concatenate([np.sqrt(np.maximum(euclidean_distances(a, r, squared=True).min(1), 0)/q.shape[1])
                           for a in np.array_split(q, max(1, int(np.ceil(len(q)/block))))])


def privacy_distances(train, heldout, synthetic):
    """Controls must be disjoint patients; caller records reference sample size."""
    ts = nearest_distance(synthetic, train)
    rs = nearest_distance(heldout, train)
    return dict(synthetic_to_train_median=float(np.median(ts)), real_to_train_median=float(np.median(rs)),
                ratio_of_medians=float(np.median(ts)/max(np.median(rs), 1e-12)),
                exact_copy_fraction=float(np.mean(ts < 1e-6)),
                n_train=len(train), n_heldout=len(heldout), n_synthetic=len(synthetic))


def membership_attack(member, nonmember, synthetic, member_groups=None, nonmember_groups=None, seed=9009):
    """Uncalibrated black-box proximity attack: closer to released pool -> member.

    AUC and patient-cluster bootstrap CI; no threshold selected on attack test rows.
    This is a limited attack, not a privacy guarantee or a shadow-model attack.
    """
    dm, dn = nearest_distance(member, synthetic), nearest_distance(nonmember, synthetic)
    y, score = np.r_[np.ones(len(dm)), np.zeros(len(dn))], -np.r_[dm, dn]
    mg = np.arange(len(dm)) if member_groups is None else np.asarray(member_groups)
    ng = np.arange(len(dn)) if nonmember_groups is None else np.asarray(nonmember_groups)
    rng, runs = np.random.default_rng(seed), []
    for _ in range(200):
        mi = np.concatenate([np.flatnonzero(mg == g) for g in rng.choice(np.unique(mg), len(np.unique(mg)), replace=True)])
        ni = np.concatenate([np.flatnonzero(ng == g) for g in rng.choice(np.unique(ng), len(np.unique(ng)), replace=True)])
        runs.append(roc_auc_score(np.r_[np.ones(len(mi)), np.zeros(len(ni))], -np.r_[dm[mi], dn[ni]]))
    return dict(auc=float(roc_auc_score(y, score)), ci95=np.quantile(runs, [.025, .975]).tolist(),
                n_member=len(dm), n_nonmember=len(dn), attack='negative distance to synthetic release; no threshold fitting')


def clinical_features(x):
    from synthetic_generation.realism.features import features_12
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return features_12(validate_ecgs(x))


def feature_distribution(real_features, synthetic_features):
    from synthetic_generation.realism.features import NAMES
    result = {}
    for j, name in enumerate(NAMES):
        a, b = real_features[:, j], synthetic_features[:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        result[name] = dict(n_real_valid=len(a), n_synthetic_valid=len(b),
                            missing_real_fraction=1-len(a)/len(real_features),
                            missing_synthetic_fraction=1-len(b)/len(synthetic_features))
        if len(a) and len(b):
            iqr = float(np.subtract(*np.quantile(a, [.75, .25])))
            result[name].update(wasserstein=float(wasserstein_distance(a, b)),
                                wasserstein_over_real_iqr=float(wasserstein_distance(a, b)/iqr) if iqr > 1e-9 else None,
                                median_real=float(np.median(a)), median_synthetic=float(np.median(b)),
                                std_real=float(a.std()), std_synthetic=float(b.std()),
                                iqr_real=iqr, iqr_synthetic=float(np.subtract(*np.quantile(b, [.75, .25]))))
    return result


def feature_judge(fr, fs, yr, ys, groups_real, groups_synthetic, seed=9009):
    """Clinical feature judge, source-patient grouping when available.

    Real/synthetic counts and class mix must already be matched by the caller.
    Group IDs must share a namespace when synthetic rows have known real sources.
    """
    x = np.r_[fr, fs].copy(); x[~np.isfinite(x)] = np.nan
    x = np.c_[x, np.eye(4)[np.r_[yr, ys]]]
    labels = np.r_[np.zeros(len(fr)), np.ones(len(fs))]
    groups = np.r_[groups_real, groups_synthetic]
    runs = []
    for offset in range(3):
        fit, dev = next(GroupShuffleSplit(n_splits=1, test_size=.3, random_state=seed+offset).split(x, labels, groups))
        if len(np.unique(labels[fit])) < 2 or len(np.unique(labels[dev])) < 2:
            continue
        with threadpool_limits(limits=2):
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                                   random_state=seed+offset).fit(x[fit], labels[fit])
            runs.append(float(roc_auc_score(labels[dev], model.predict_proba(x[dev])[:, 1])))
    return dict(auc_mean=float(np.mean(runs)) if runs else None, runs=runs,
                note='Exploratory 3 patient-grouped splits; no model selection on this AUC')


def qt_summary(x):
    """Optional NeuroKit DWT delineation in mV; 100 Hz estimates, not clinical truth."""
    try:
        import neurokit2 as nk
    except ImportError:
        return dict(status='unavailable: optional neurokit2 dependency')
    import json
    stats = json.loads((ROOT / 'processed_data/normalization_stats.json').read_text())
    values, failed, valid_recordings = [], 0, 0
    for row in validate_ecgs(x):
        sig = row[:, 1]*stats['std'][1]+stats['mean'][1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                clean = nk.ecg_clean(sig, sampling_rate=100)
                _, rp = nk.ecg_peaks(clean, sampling_rate=100)
                _, waves = nk.ecg_delineate(clean, rp, sampling_rate=100, method='dwt', show=False)
            q, t = np.asarray(waves['ECG_R_Onsets']), np.asarray(waves['ECG_T_Offsets'])
            dt = (t-q)*10
            valid = dt[np.isfinite(dt) & (dt >= 100) & (dt <= 700)]
            values.extend(valid.tolist()); valid_recordings += bool(len(valid))
        except (ValueError, IndexError, KeyError, ZeroDivisionError, TypeError):
            failed += 1
    return dict(qt_ms_median=float(np.median(values)) if values else None, valid_intervals=len(values),
                n_recordings=len(x), recordings_with_valid_intervals=valid_recordings, failures=failed,
                note='Lead II DWT R-onset to T-offset proxy; valid 100–700 ms only; 10 ms sampling resolution')


def paired_rpeak_check(source, synthetic, tolerance=5):
    """One-to-one peak matching; valid only for aligned, source-timed transformations."""
    from synthetic_generation.beats import detect_r_peaks
    from synthetic_generation.leads import to_independent_8
    a, b = to_independent_8(validate_ecgs(source)), to_independent_8(validate_ecgs(synthetic))
    if len(a) != len(b):
        raise ValueError('Paired ECG counts differ')
    total = missed = extra = 0
    for aa, bb in zip(a, b):
        r, s = detect_r_peaks(aa), detect_r_peaks(bb)
        edges = sorted((abs(int(q)-int(p)), i, j) for i, q in enumerate(r) for j, p in enumerate(s)
                       if abs(int(q)-int(p)) <= tolerance)
        ri, si = set(), set()
        for _, i, j in edges:
            if i not in ri and j not in si:
                ri.add(i); si.add(j)
        total += len(r); missed += len(r)-len(ri); extra += len(s)-len(si)
    return dict(source_beats=total, missed_fraction=missed/total if total else None,
                extra_per_source_beat=extra/total if total else None, tolerance_ms=tolerance*10)


def score_pool(real, yr, synthetic, ys, train, yt, encoder, groups_real, groups_train,
               synthetic_groups=None, allow_privacy=True, include_qt=False, k=5):
    """Caller supplies class/count-matched patient-disjoint references and provenance."""
    real, synthetic, train = map(validate_ecgs, (real, synthetic, train))
    yr, ys, yt = (np.asarray(v) for v in (yr, ys, yt))
    for values, ecgs in ((yr, real), (ys, synthetic), (yt, train)):
        if values.shape != (len(ecgs),) or not np.isin(values, range(4)).all() or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Expected aligned integer rhythm labels in 0..3')
    if len(groups_real) != len(real) or len(groups_train) != len(train):
        raise ValueError('Patient groups must align with reference rows')
    if synthetic_groups is not None and len(synthetic_groups) != len(synthetic):
        raise ValueError('Source-patient groups must align with synthetic rows')
    er, es, et = encoder(real), encoder(synthetic), encoder(train)
    # Fixed train-only scaling; do not let any generator change the embedding metric.
    mean, scale = et.mean(0), np.maximum(et.std(0), 1e-6)
    er, es, et = (er-mean)/scale, (es-mean)/scale, (et-mean)/scale
    fr, fs = clinical_features(real), clinical_features(synthetic)
    groups_synthetic = np.asarray([f'syn:{i}' for i in range(len(synthetic))]) if synthetic_groups is None else synthetic_groups
    result = dict(by_class={}, feature_judge=feature_judge(fr, fs, yr, ys, groups_real, groups_synthetic),
                  peak_detectability=dict(real_fraction_ge3_peaks=float((fr[:, 4] >= 3).mean()),
                                          synthetic_fraction_ge3_peaks=float((fs[:, 4] >= 3).mean()),
                                          note='Detection availability, not sensitivity: unpaired generations have no known beat truth'))
    for c in np.unique(ys):
        r, s, t = yr == c, ys == c, yt == c
        if not r.any() or not t.any():
            raise ValueError('Missing reference class')
        row = dict(features=feature_distribution(fr[r], fs[s]))
        row['prdc'] = prdc(er[r], es[s], k) if min(r.sum(), s.sum()) > k else dict(status='too few rows for PRDC')
        if allow_privacy:
            row['embedding_privacy'] = privacy_distances(et[t], er[r], es[s])
            row['waveform_privacy'] = privacy_distances(train[t].reshape(t.sum(), -1), real[r].reshape(r.sum(), -1),
                                                       synthetic[s].reshape(s.sum(), -1))
            row['membership'] = membership_attack(et[t], er[r], es[s], groups_train[t], groups_real[r])
        result['by_class'][CLASSES[c]] = row
    if include_qt:
        result['qt_real'], result['qt_synthetic'] = qt_summary(real), qt_summary(synthetic)
    result['privacy_status'] = 'limited proximity attack on matched reference subsets' if allow_privacy else 'excluded: incompatible or unknown training provenance'
    return result
