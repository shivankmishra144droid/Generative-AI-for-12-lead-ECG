"""Phases C-E on development patients: Gate C, Gate D (adherence + HR control sweep), reduced sampler grid.

py -3.14 -m latent_diffusion.dev_eval_v2 gates     # Gate C (cs) and Gate D (ds), CFG 2, 50 DDIM steps
py -3.14 -m latent_diffusion.dev_eval_v2 sweep     # HR control sweeps (cs, ds, v1 descriptive)
py -3.14 -m latent_diffusion.dev_eval_v2 grid      # Phase E reduced grid on the main generator
py -3.14 -m latent_diffusion.dev_eval_v2 snapshots # descriptive quality-vs-step curve
Validation is never loaded here.
"""
import json
import sys
import numpy as np
from .common import CLASSES
from .v2_common import ART, RESULTS, AUDITS, SEED, register, write_json, gpu_lease, now
from .phase_a_v2 import cohort_setup, prepare, subset
from .gen_eval_v2 import score_prepared
from .sample_v2 import Generator, fit_requests, detected_peaks, adherence
from .rhythm_v2 import regular_peaks

FULL = [256, 170, 82, 63]


def gate_metrics(res):
    fj = res['feature_judges']
    by = res['by_class']
    hrw = {c: r['features']['hr']['wasserstein']['value'] for c, r in by.items()}
    cov = {c: r['emb_task']['coverage']['value'] for c, r in by.items()}
    fail = {}
    for c, r in by.items():
        s = r['rhythm_synthetic']
        fail[c] = 1 - s['fraction_ge3_peaks']['value'] + s['hr_outside_30_200']['value'] + s['any_rr_gap_over_2s']['value']
    return dict(full_judge=fj['full']['value'], rhythm_judge=fj['rhythm']['value'],
                groups={k: v['value'] for k, v in fj.items() if k != 'full'},
                hr_wasserstein=hrw, mean_hr_wasserstein=float(np.mean(list(hrw.values()))),
                coverage_task=cov, mean_coverage_task=float(np.mean(list(cov.values()))),
                rhythm_failure=fail, mean_rhythm_failure=float(np.mean(list(fail.values()))))


def score_pool(tag, X, y, data, encoders, scalers, afib_cv, replicates=100, real_key='real_full'):
    path = RESULTS/f'dev_{tag}.json'
    if path.exists():
        return json.loads(path.read_text())
    syn = prepare(f'dev_{tag}', X, y, np.asarray([f'{tag}:{i}' for i in range(len(y))]), encoders, scalers, afib_cv)
    real = data[real_key]
    counts = [min((real['y'] == c).sum(), (syn['y'] == c).sum()) for c in range(4)]
    pick = lambda d: np.concatenate([np.flatnonzero(d['y'] == c)[:n] for c, n in enumerate(counts)])
    r, s, t = subset(real, pick(real)), subset(syn, pick(syn)), subset(data['fit_reference'], pick(data['fit_reference']))
    res = score_prepared(r, s, t, ('emb_task', 'emb_ssl'), afib_cv, replicates=replicates, privacy=True)
    res['gate_metrics'] = gate_metrics(res)
    write_json(path, res)
    return res


def generate_dev(name, counts, guidance=2., steps=50, eta=0., seed=9300, ckpt='last.pt'):
    y, v, peaks, rows = fit_requests(counts, seed)
    with gpu_lease(f'dev_gen_{name}') as device:
        gen = Generator(name, device, ckpt)
        X = gen(y, v, peaks, guidance, steps, eta, seed)
    return X, y, peaks, gen


def gates():
    data, enc, sc, afib = cohort_setup()
    ref = {k: json.loads((RESULTS/f'rescore_{k}_full.json').read_text()) for k in ('hybrid', 'latent_diffusion')}
    base = {k: gate_metrics(v) for k, v in ref.items()}
    out = dict(created=now(), setting='CFG 2, DDIM 50, eta 0, EMA 0.999 last checkpoint', baselines=base)
    for name in ('cs', 'ds'):
        if not (ART/f'diffusion_{name}'/'last.pt').exists():
            continue
        X, y, peaks, gen = generate_dev(name, FULL)
        res = score_pool(f'{name}_cfg2_s50', X, y, data, enc, sc, afib)
        m = res['gate_metrics']
        m['step'] = gen.step
        if gen.rhythm:
            m['adherence'] = adherence(peaks, detected_peaks(X))
        out[name] = m
        print(name, json.dumps({k: m[k] for k in ('full_judge', 'rhythm_judge', 'mean_hr_wasserstein',
                                                  'mean_coverage_task', 'mean_rhythm_failure')}), flush=True)
    hyb = base['hybrid']
    if 'cs' in out:
        cs = out['cs']
        hr_wins = sum(cs['hr_wasserstein'][c] <= hyb['hr_wasserstein'][c] for c in CLASSES)
        out['gate_C'] = dict(success=bool(cs['full_judge'] < hyb['full_judge'] or hr_wins >= 3),
                             full_judge_below_hybrid=bool(cs['full_judge'] < hyb['full_judge']),
                             hr_classes_at_or_below_hybrid=int(hr_wins),
                             note='single capacity rung (C-S) tonight; capacity choice trivially C-S')
    write_json(RESULTS/'gate_CD.json', out)
    return out


def sweep():
    """Requested vs measured HR, 32 per cell, matched seeds across HR; regular RR (AFIB gamma cv .25)."""
    hrs = [40, 50, 60, 70, 90, 110, 130, 150]
    out = dict(created=now(), hrs=hrs, models={})
    rng = np.random.default_rng(SEED + 11)
    for name in ('cs', 'ds'):
        if not (ART/f'diffusion_{name}'/'last.pt').exists():
            continue
        res = {}
        with gpu_lease(f'sweep_{name}') as device:
            gen = Generator(name, device)
            for c in range(4):
                y, v, _, _ = fit_requests([32 if k == c else 0 for k in range(4)], SEED + 20 + c)
                req, meas, adh = [], [], []
                for hr in hrs:
                    vv = v.copy()
                    vv[:, 0] = (hr - 75)/40
                    vv[:, 3] = 1
                    pk = np.stack([regular_peaks(hr, np.random.default_rng(1000*c + j), cv=.25 if c == 1 else 0.,
                                                 irregular=(c == 1)) for j in range(32)])
                    X = gen(y, vv, pk, 2., 50, 0., seed=SEED + 30 + c)   # same latent seed for every HR
                    det = detected_peaks(X)
                    m = [6000/np.diff(d).mean() if len(d) >= 3 else np.nan for d in det]
                    req += [hr]*32; meas += m
                    if gen.rhythm:
                        adh.append(dict(hr=hr, **adherence(pk, det)))
                req, meas = np.array(req, float), np.array(meas, float)
                ok = np.isfinite(meas)
                slope, icpt = np.polyfit(req[ok], meas[ok], 1)
                r2 = 1 - np.sum((meas[ok] - (slope*req[ok] + icpt))**2)/np.sum((meas[ok] - meas[ok].mean())**2)
                res[CLASSES[c]] = dict(slope=float(slope), intercept=float(icpt), r2=float(r2), valid=float(ok.mean()),
                                       median_by_hr={h: float(np.nanmedian(meas[req == h])) for h in hrs}, adherence=adh)
                print(name, CLASSES[c], round(slope, 3), round(r2, 3), flush=True)
        out['models'][name] = res
    if 'ds' in out['models']:
        d = out['models']['ds']
        gd = json.loads((RESULTS/'gate_CD.json').read_text())
        ds, cs = gd.get('ds'), gd.get('cs')
        ok_sweep = all(.9 <= d[c]['slope'] <= 1.1 and d[c]['r2'] >= .9 for c in ('SR', 'STACH', 'SBRAD'))
        a = ds['adherence']
        out['gate_D'] = dict(missed_ok=a['missed_fraction'] <= .05, extra_ok=a['extra_per_beat'] <= .05,
                             sweep_ok=ok_sweep, rhythm_judge_ok=ds['rhythm_judge'] <= .70,
                             full_judge_ok=ds['full_judge'] <= cs['full_judge'] + .02)
        out['gate_D']['passed'] = all(out['gate_D'].values())
        print('Gate D', out['gate_D'], flush=True)
    write_json(RESULTS/'rhythm_control.json', out)
    return out


def grid():
    rc = json.loads((RESULTS/'rhythm_control.json').read_text())
    main = 'ds' if rc.get('gate_D', {}).get('passed') else 'cs'
    register('phaseE', dict(main=main, cfg=[1, 2, 3], steps=[50, 100], eta=[0], n_per_class=63,
                            rule='mean rank over Gate C metrics (+missed beats if R-map); eligible if emb_task coverage >= 0.8 x best'))
    data, enc, sc, afib = cohort_setup()
    rows = {}
    for g in (1., 2., 3.):
        for st in (50, 100):
            tag = f'E_{main}_cfg{g:g}_s{st}'
            X, y, peaks, gen = generate_dev(main, [63]*4, g, st, 0., seed=9400)
            res = score_pool(tag, X, y, data, enc, sc, afib, replicates=50)
            m = res['gate_metrics']
            if gen.rhythm:
                m['adherence'] = adherence(peaks, detected_peaks(X))
            rows[f'cfg{g:g}_s{st}'] = m
            print(tag, round(m['full_judge'], 3), round(m['mean_coverage_task'], 3), flush=True)
    keys = [('full_judge', 1), ('rhythm_judge', 1), ('mean_hr_wasserstein', 1), ('mean_coverage_task', -1),
            ('mean_rhythm_failure', 1)]
    best_cov = max(r['mean_coverage_task'] for r in rows.values())
    elig = [k for k, r in rows.items() if r['mean_coverage_task'] >= .8*best_cov]
    ranks = {k: 0. for k in elig}
    for key, sign in keys + ([('missed', 1)] if main == 'ds' else []):
        val = {k: sign*(rows[k]['adherence']['missed_fraction'] if key == 'missed' else rows[k][key]) for k in elig}
        for r, k in enumerate(sorted(elig, key=lambda q: val[q])):
            ranks[k] += r
    chosen = min(elig, key=lambda k: (ranks[k], k))
    g, st = chosen.replace('cfg', '').split('_s')
    sampling = dict(main=main, guidance=float(g), steps=int(st), eta=0., checkpoint='last.pt', ema='ema0.999',
                    chosen_by='phase E mean rank on dev', frozen_utc=now())
    write_json(RESULTS/'sampler_sweep.json', dict(rows=rows, eligible=elig, ranks=ranks, chosen=chosen))
    write_json(ART/'sampling.json', sampling)
    print('chosen', sampling, flush=True)
    return sampling


def snapshots():
    data, enc, sc, afib = cohort_setup()
    out = {}
    for name in ('cs', 'ds'):
        folder = ART/f'diffusion_{name}'
        for ck in sorted(folder.glob('step*.pt'), key=lambda p: int(p.stem[4:])):
            X, y, peaks, gen = generate_dev(name, [63]*4, 2., 50, 0., seed=9500, ckpt=ck.name)
            res = score_pool(f'snap_{name}_{ck.stem}', X, y, data, enc, sc, afib, replicates=20)
            m = res['gate_metrics']
            out.setdefault(name, {})[gen.step] = dict(full=m['full_judge'], rhythm=m['rhythm_judge'],
                                                       hrw=m['mean_hr_wasserstein'], fail=m['mean_rhythm_failure'],
                                                       cov=m['mean_coverage_task'])
            print(name, gen.step, out[name][gen.step], flush=True)
    write_json(RESULTS/'snapshot_curves.json', out)


if __name__ == '__main__':
    dict(gates=gates, sweep=sweep, grid=grid, snapshots=snapshots)[sys.argv[1]]()
