"""Phase F: final realism benchmark on the validation split (run once; sampler frozen in artifacts/v2/sampling.json).

Reference: all validation AFIB/STACH/SBRAD (224/125/96) + 256 SR (SR cap 256, amendment 2: every frozen v1 pool
has 256/class). Floor: two disjoint validation patient halves at feasible matched counts.
Fit reference for privacy: fit rows at the same counts. AFIB irregularity threshold: the Phase A dev value.
"""
import json
import pickle
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from .common import ROOT, ART as V1_ART, CLASSES, digest, load_split
from .v2_common import ART, RESULTS, AUDITS, SEED, training_data, register, write_json, gpu_lease, now
from .phase_a_v2 import cohort_setup, prepare, subset, select
from .gen_eval_v2 import score_prepared, feature_judges
from .sample_v2 import Generator, fit_requests, detected_peaks, adherence
from .rhythm_v2 import MAXP
from .dev_eval_v2 import gate_metrics
from .phase_b_v2 import reconstruct
from .ae_ladder_v2 import load_rung

VAL_COUNTS = [256, 224, 125, 96]


def val_cohorts(encoders, scalers, afib_cv):
    xv, yv = load_split('val')
    vp = np.load(V1_ART/'val_conditions.npz')['patient']
    rows = select(yv, np.arange(len(yv)), VAL_COUNTS, SEED + 40)
    groups = np.asarray([f'v:{p}' for p in vp])
    real = prepare('val_real', np.asarray(xv[rows]), yv[rows], groups[rows], encoders, scalers, afib_cv)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED + 41).split(rows, yv[rows], groups[rows]))
    ca = [int((yv[rows[a]] == c).sum()) for c in range(4)]
    cb = [int((yv[rows[b]] == c).sum()) for c in range(4)]
    n = [min(p, q) for p, q in zip(ca, cb)]
    fa = np.concatenate([a[yv[rows[a]] == c][:k] for c, k in enumerate(n)])
    fb = np.concatenate([b[yv[rows[b]] == c][:k] for c, k in enumerate(n)])
    x, y, patient, fit, dev = training_data()
    trows = select(y, fit, VAL_COUNTS, SEED + 42)
    fitref = prepare('val_fit_reference', np.asarray(x[trows]), y[trows], np.asarray([f'p:{p}' for p in patient[trows]]),
                     encoders, scalers, afib_cv)
    return real, subset(real, fa), subset(real, fb), fitref, rows, n


def rhythm_model_requests(counts, seed):
    """Hybrid's own RhythmModel (beat_templates/assembly_models.pkl, fitted on train), bounds from assembly.py."""
    from synthetic_generation.assembly import MEAN_RR_BOUNDS
    models = pickle.load(open(ROOT/'generator_comparison/beat_templates/assembly_models.pkl', 'rb'))
    y, v, _, _ = fit_requests(counts, seed)
    rng = np.random.default_rng(seed)
    peaks = np.full((len(y), MAXP), -1, np.int32)
    for i, c in enumerate(y):
        m = models[int(c)]['rhythm']
        m.mean_rr_bounds = MEAN_RR_BOUNDS[int(c)]
        p = np.asarray(m.sample_r_positions(rng))
        p = p[(p >= 0) & (p < 1000)][:MAXP]
        peaks[i, :len(p)] = p
        if len(p) >= 2:                                  # condition HR follows the requested rhythm
            v[i, 0] = (6000/np.diff(p).mean() - 75)/40
            v[i, 3] = 1
    return y, v, peaks


def run():
    sampling = json.loads((ART/'sampling.json').read_text())
    rc = json.loads((RESULTS/'rhythm_control.json').read_text())
    gate_d = rc.get('gate_D', {}).get('passed', False)
    register('phaseF', dict(sampling=sampling, val_counts=VAL_COUNTS, replicates=500, gate_D_passed=gate_d,
                            decision='better iff main judge (mean+2SD) < hybrid (mean-2SD) AND emb_task coverage CI not '
                                     'entirely below hybrid in >=3/4 classes AND HR-W CI not entirely above hybrid in >=3/4 '
                                     'classes; worse iff hybrid judge (mean+2SD) < main (mean-2SD); else not distinguishable'))
    data, enc, sc, afib = cohort_setup()
    real, fa, fb, fitref, vrows, nfloor = val_cohorts(enc, sc, afib)
    counts = VAL_COUNTS
    pools = {}
    pools['hybrid'] = dict(path=ROOT/'generator_comparison/tstr/pool_beatgen.npz', privacy=True)
    for k in ('latent_diffusion', 'vae', 'ecgsyn'):
        pools[{'latent_diffusion': 'latent_v1'}.get(k, k)] = dict(path=ART/'pools'/f'{k}_v1_expanded.npz', privacy=k != 'ecgsyn')
    gen_specs = {'latent_v2': ('cs', None)}
    if (ART/'diffusion_ds'/'last.pt').exists():
        gen_specs['latent_v2_rhythm_bank'] = ('ds', 'bank')
        gen_specs['latent_v2_rhythm_model'] = ('ds', 'model')
    requested = {}
    for tag, (name, src) in gen_specs.items():
        path = ART/'pools'/f'F_{tag}.npz'
        if not path.exists():
            if src == 'model':
                y, v, pk = rhythm_model_requests(counts, 9610)
            else:
                y, v, pk, _ = fit_requests(counts, 9600)
            g_cfg = sampling if sampling['main'] == name else dict(guidance=2., steps=50, eta=0.)
            with gpu_lease(f'phaseF_{tag}') as device:
                gen = Generator(name, device)
                X = gen(y, v, pk, g_cfg['guidance'], g_cfg['steps'], g_cfg['eta'], seed=9620)
            np.savez_compressed(path, X=X, y=y, peaks=pk, conditions=v)
        pools[tag] = dict(path=path, privacy=True)
    # AE reconstruction ceiling on validation (paired)
    rec_path = ART/'pools'/'F_ae_recon.npz'
    if not rec_path.exists():
        xv, yv = load_split('val')
        vc = np.load(V1_ART/'val_conditions.npz')['values']
        info = json.loads((ART/'chosen_ae.json').read_text())
        with gpu_lease('phaseF_recon') as device:
            rec = reconstruct(load_rung(info['rung'], device), xv[vrows], yv[vrows], vc[vrows], device)
        np.savez_compressed(rec_path, X=rec, y=yv[vrows])
    out = dict(created=now(), sampling=sampling, gate_D_passed=gate_d, floor_counts=nfloor, val_counts=counts, results={})
    floor = score_prepared(fa, fb, fitref, ('emb_task', 'emb_ssl'), afib, privacy=False)
    floor['gate_metrics'] = gate_metrics(floor)
    write_json(RESULTS/'F_floor.json', floor)
    out['results']['real_floor'] = floor['gate_metrics']
    rec = np.load(rec_path)
    rs = prepare('F_ae_recon', rec['X'], real['y'], real['groups'], enc, sc, afib)
    res = score_prepared(real, rs, fitref, ('emb_task', 'emb_ssl'), afib, paired=True, privacy=False)
    res['gate_metrics'] = gate_metrics(res)
    write_json(RESULTS/'F_ae_recon.json', res)
    out['results']['ae_recon'] = res['gate_metrics']
    for tag, spec in pools.items():
        p = RESULTS/f'F_{tag}.json'
        if p.exists():
            res = json.loads(p.read_text())
        else:
            d = np.load(spec['path'])
            cls = [c for c in range(4) if (d['y'] == c).sum() >= counts[c]]
            rng = np.random.default_rng(SEED + 50)
            idx = np.concatenate([rng.choice(np.flatnonzero(d['y'] == c), counts[c], replace=False) for c in cls])
            syn = prepare(f'F_{tag}', d['X'][idx], d['y'][idx], np.asarray([f'{tag}:{i}' for i in idx]), enc, sc, afib)
            keep = np.isin(real['y'], cls)
            r, t = subset(real, np.flatnonzero(keep)), subset(fitref, np.flatnonzero(np.isin(fitref['y'], cls)))
            res = score_prepared(r, syn, t, ('emb_task', 'emb_ssl'), afib, privacy=spec['privacy'])
            res['gate_metrics'] = gate_metrics(res)
            if 'peaks' in d.files and tag.startswith('latent_v2_rhythm'):   # only R-map pools saw these peaks
                res['adherence'] = adherence(d['peaks'][idx], detected_peaks(d['X'][idx]))
            res['source_sha256'] = digest(spec['path'])
            write_json(p, res)
        out['results'][tag] = dict(res['gate_metrics'], adherence=res.get('adherence'))
        print(tag, round(res['gate_metrics']['full_judge'], 3), flush=True)
    # three sampling seeds of the main generator (sampling variance, not training variance)
    main_name = sampling['main']
    seeds = {}
    for s in (9630, 9631, 9632):
        p = ART/'pools'/f'F_main_seed{s}.npz'
        if not p.exists():
            y, v, pk, _ = fit_requests(counts, s)
            with gpu_lease(f'phaseF_seed{s}') as device:
                X = Generator(main_name, device)(y, v, pk, sampling['guidance'], sampling['steps'], sampling['eta'], seed=s)
            np.savez_compressed(p, X=X, y=y)
        d = np.load(p)
        syn = prepare(f'F_main_seed{s}', d['X'], d['y'], np.asarray([f'seed{s}:{i}' for i in range(len(d['y']))]), enc, sc, afib)
        fj = feature_judges(real['features'], syn['features'], real['y'], syn['y'], real['groups'], syn['groups'], importance=False)
        seeds[s] = {k: v['value'] for k, v in fj.items()}
    out['main_sampling_seeds'] = seeds
    out['decision'] = decide(out, 'latent_v2_rhythm_bank' if main_name == 'ds' else 'latent_v2')
    write_json(RESULTS/'benchmark_val.json', out)
    table(out)
    print('decision', out['decision'], flush=True)
    return out


def decide(out, main):
    m = json.loads((RESULTS/f'F_{main}.json').read_text())
    h = json.loads((RESULTS/'F_hybrid.json').read_text())
    mj, hj = m['feature_judges']['full'], h['feature_judges']['full']
    cov_ok = sum(m['by_class'][c]['emb_task']['coverage']['ci95'][1] >= h['by_class'][c]['emb_task']['coverage']['ci95'][0]
                 for c in CLASSES)
    hr_ok = sum(m['by_class'][c]['features']['hr']['wasserstein']['ci95'][0] <= h['by_class'][c]['features']['hr']['wasserstein']['ci95'][1]
                for c in CLASSES)
    if mj['value'] + 2*mj['sd'] < hj['value'] - 2*hj['sd'] and cov_ok >= 3 and hr_ok >= 3:
        verdict = 'better'
    elif hj['value'] + 2*hj['sd'] < mj['value'] - 2*mj['sd']:
        verdict = 'worse'
    else:
        verdict = 'not distinguishable'
    return dict(main=main, verdict=verdict, main_judge=[mj['value'], mj['sd']], hybrid_judge=[hj['value'], hj['sd']],
                coverage_not_worse_classes=int(cov_ok), hr_not_worse_classes=int(hr_ok))


def table(out):
    rows, html = [], []
    cols = ['full_judge', 'rhythm_judge', 'mean_hr_wasserstein', 'mean_coverage_task', 'mean_rhythm_failure']
    for tag, m in out['results'].items():
        g = m.get('groups', {})
        a = (m.get('adherence') or {}) if tag.startswith('latent_v2_rhythm') else {}
        rows.append([tag] + [f"{m[c]:.3f}" for c in cols] + [f"{g.get(k, float('nan')):.3f}" for k in ('morphology', 'noise_wander', 'global')]
                    + [f"{a.get('missed_fraction', float('nan')):.3f}"])
    head = ['pool'] + cols + ['morph_judge', 'noise_judge', 'global_judge', 'missed_rpeaks']
    csv = '\n'.join(','.join(r) for r in [head] + rows)
    (RESULTS/'benchmark_val.csv').write_text(csv, encoding='utf-8')
    body = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in r) + '</tr>' for r in rows)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Study 9 V2 Benchmark</title><style>body{{font-family:system-ui;margin:16px;background:#fff;color:#111}}
table{{border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
td:first-child{{text-align:left}}@media(prefers-color-scheme:dark){{body{{background:#111;color:#eee}}td,th{{border-color:#444}}}}</style></head>
<body><h1>Study 9 V2: validation realism benchmark</h1><p>Phase F, validation split, run once. Judge AUC 0.5 = real-like.
Decision: <b>{out['decision']['verdict']}</b> ({out['decision']['main']} vs hybrid). Details: results/v2/benchmark_val.json.</p>
<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{body}</table></body></html>"""
    (RESULTS.parent.parent/'BENCHMARK_V2.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    run()
