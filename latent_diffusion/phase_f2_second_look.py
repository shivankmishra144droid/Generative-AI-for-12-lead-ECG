"""Amendment 8: validation second look for cs 120k, ds 120k and cm 100k (run once).

Same reference, floor, fit reference, counts, requests, sampler and decision rule as Phase F (phase_f_v2.py).
New tags (F2_*) so no Phase F cache or result is reused or overwritten. cm needs the conjugate S4 kernel
(amendment 6); cs/ds need the fallback kernel.
py -3.14 -m latent_diffusion.phase_f2_second_look
"""
import contextlib
import json
import numpy as np
from .common import digest
from .v2_common import ART, RESULTS, SEED, register, write_json, gpu_lease, now
from .phase_a_v2 import cohort_setup, prepare, subset
from .gen_eval_v2 import score_prepared, feature_judges
from .sample_v2 import Generator, fit_requests, detected_peaks, adherence
from .dev_eval_v2 import gate_metrics
from .phase_f_v2 import VAL_COUNTS, val_cohorts
from .s4_kernel import conj_kernel, assert_fallback_kernel

SPECS = {                      # tag: (model, checkpoint, kernel, uses R-map)
    'cs120k': ('cs', 'last.pt', 'fallback', False),
    'ds120k_bank': ('ds', 'last.pt', 'fallback', True),
    'cm100k': ('cm', 'step100000.pt', 'conj', False),
}
SAMPLER = dict(guidance=2., steps=50, eta=0.)


def kernel(kind):
    if kind == 'conj':
        return conj_kernel()
    assert_fallback_kernel()
    return contextlib.nullcontext()


def generate(tag, model, ckpt, kind, seed, request_seed=9600):
    path = ART/'pools'/f'F2_{tag}.npz' if seed == 9620 else ART/'pools'/f'F2_{tag}_seed{seed}.npz'
    if path.exists():
        return path
    y, v, pk, _ = fit_requests(VAL_COUNTS, request_seed)
    with gpu_lease(f'phaseF2_{tag}_{seed}') as device, kernel(kind):
        gen = Generator(model, device, ckpt)
        X = gen(y, v, pk, SAMPLER['guidance'], SAMPLER['steps'], SAMPLER['eta'], seed=seed)
    np.savez_compressed(path, X=X, y=y, peaks=pk, conditions=v, step=gen.step)
    return path


def decide(m, h):
    """Copied unchanged from phase_f_v2.decide (amendment 8), taking the result dicts directly."""
    from .common import CLASSES
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
    return dict(verdict=verdict, judge=[mj['value'], mj['sd']], hybrid_judge=[hj['value'], hj['sd']],
                coverage_not_worse_classes=int(cov_ok), hr_not_worse_classes=int(hr_ok))


def run():
    register('phaseF2_second_look', dict(specs={k: list(v) for k, v in SPECS.items()}, sampler=SAMPLER, val_counts=VAL_COUNTS,
                                         request_seed=9600, sample_seed=9620, extra_cm_seeds=[9630, 9631, 9632],
                                         rule='phase_f_v2.decide unchanged; amendment 8'))
    data, enc, sc, afib = cohort_setup()
    real, fa, fb, fitref, vrows, nfloor = val_cohorts(enc, sc, afib)
    hybrid = json.loads((RESULTS/'F_hybrid.json').read_text())
    out = dict(created=now(), label='second look (amendment 8); checkpoints selected on dev; validation used before in Phase F/G',
               sampler=SAMPLER, val_counts=VAL_COUNTS, floor=json.loads((RESULTS/'F_floor.json').read_text())['gate_metrics'],
               hybrid=hybrid['gate_metrics'], results={}, decisions={})
    for tag, (model, ckpt, kind, rmap) in SPECS.items():
        p = RESULTS/f'F2_{tag}.json'
        if p.exists():
            res = json.loads(p.read_text())
        else:
            path = generate(tag, model, ckpt, kind, 9620)
            d = np.load(path)
            rng = np.random.default_rng(SEED + 50)
            idx = np.concatenate([rng.choice(np.flatnonzero(d['y'] == c), VAL_COUNTS[c], replace=False) for c in range(4)])
            syn = prepare(f'F2_{tag}', d['X'][idx], d['y'][idx], np.asarray([f'F2_{tag}:{i}' for i in idx]), enc, sc, afib)
            res = score_prepared(real, syn, fitref, ('emb_task', 'emb_ssl'), afib, privacy=True)
            res['gate_metrics'] = gate_metrics(res)
            if rmap:
                res['adherence'] = adherence(d['peaks'][idx], detected_peaks(d['X'][idx]))
            res['source_sha256'] = digest(path)
            res['checkpoint_step'] = int(d['step'])
            write_json(p, res)
        out['results'][tag] = dict(res['gate_metrics'], adherence=res.get('adherence'), step=res.get('checkpoint_step'))
        out['decisions'][tag] = decide(res, hybrid)
        print(tag, 'judge', round(res['gate_metrics']['full_judge'], 3), out['decisions'][tag]['verdict'], flush=True)
    seeds = {}
    for s in (9630, 9631, 9632):                      # sampling variance of the primary pool (judge only)
        d = np.load(generate('cm100k', 'cm', 'step100000.pt', 'conj', s, request_seed=s))
        syn = prepare(f'F2_cm100k_seed{s}', d['X'], d['y'], np.asarray([f'F2s{s}:{i}' for i in range(len(d['y']))]), enc, sc, afib)
        fj = feature_judges(real['features'], syn['features'], real['y'], syn['y'], real['groups'], syn['groups'], importance=False)
        seeds[s] = {k: v['value'] for k, v in fj.items()}
    out['cm100k_sampling_seeds'] = seeds
    write_json(RESULTS/'benchmark_val_second_look.json', out)
    print(json.dumps(out['decisions'], indent=1), flush=True)
    return out


if __name__ == '__main__':
    run()
