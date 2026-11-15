"""Amendment 3/5: dev evaluation of cs and ds after continued training to 120 k (second look, dev only).

Same requests, seeds, counts and sampler as the 60 k gates; `_120k` tags keep the per-tag caches of
dev_eval_v2 / phase_a_v2 from returning the 60 k results. Outputs go to results/v2/ext120k/.
py -3.14 -m latent_diffusion.eval_120k_v2 gates | sweep | guidance
"""
import json
import sys
import numpy as np
from . import dev_eval_v2 as D
from .v2_common import RESULTS, write_json, now
from .phase_a_v2 import cohort_setup
from .sample_v2 import detected_peaks, adherence

OUT = RESULTS/'ext120k'


def gates():
    data, enc, sc, afib = cohort_setup()
    out = dict(created=now(), setting='CFG 2, DDIM 50, eta 0, EMA 0.999, last.pt (120 k)',
               at60k=json.loads((RESULTS/'gate_CD_60k.json').read_text()))
    for name in ('cs', 'ds'):
        X, y, peaks, gen = D.generate_dev(name, D.FULL)
        res = D.score_pool(f'{name}_cfg2_s50_120k', X, y, data, enc, sc, afib)
        m = res['gate_metrics']
        m['step'] = gen.step
        if gen.rhythm:
            m['adherence'] = adherence(peaks, detected_peaks(X))
        out[name] = m
        print(name, gen.step, json.dumps({k: round(m[k], 4) for k in ('full_judge', 'rhythm_judge', 'mean_hr_wasserstein',
                                                                      'mean_coverage_task', 'mean_rhythm_failure')}), flush=True)
    write_json(OUT/'gate_CD.json', out)


def sweep():
    """dev_eval_v2.sweep unchanged, pointed at the 120 k gate file and writing into OUT."""
    D.RESULTS = OUT
    D.sweep()


def guidance():
    data, enc, sc, afib = cohort_setup()
    rows = {}
    for g in (1., 1.5, 2.):
        X, y, peaks, gen = D.generate_dev('ds', [63]*4, g, 50, 0., seed=9400)
        res = D.score_pool(f'G_ds_cfg{g:g}_120k', X, y, data, enc, sc, afib, replicates=50)
        m = res['gate_metrics']
        m['adherence'] = adherence(peaks, detected_peaks(X))
        f = {c: r['features']['tpl_p2p'] for c, r in res['by_class'].items()}
        m['tpl_p2p_ratio'] = {c: f[c]['median_synthetic']['value']/f[c]['median_real']['value'] for c in f}
        rows[f'cfg{g:g}'] = m
        print('ds cfg', g, 'full', round(m['full_judge'], 3), 'morph', round(m['groups']['morphology'], 3),
              'rhythm', round(m['rhythm_judge'], 3), 'qrs_ratio', {c: round(v, 2) for c, v in m['tpl_p2p_ratio'].items()},
              'missed', round(m['adherence']['missed_fraction'], 3), flush=True)
    write_json(OUT/'ds_guidance.json', dict(created=now(), note='descriptive (amendment 5); not used for Gate D', rows=rows))


def cm():
    """Amendment 4: C-M checkpoints on the snapshot protocol (63/class, CFG 2, DDIM 50, seed 9500), the same
    protocol as dev_eval_v2.snapshots for cs, so cs step120000 is the paired comparison. Includes privacy."""
    from .s4_kernel import conj_kernel
    data, enc, sc, afib = cohort_setup()
    rows = {}
    for ck in ('step25000.pt', 'step50000.pt', 'step54000_colab.pt', 'step75000.pt', 'step100000.pt'):   # cached tags reuse earlier scores
        with conj_kernel():                  # amendment 6: cm was trained with pykeops (conjugate-pair kernel)
            X, y, peaks, gen = D.generate_dev('cm', [63]*4, 2., 50, 0., seed=9500, ckpt=ck)
        res = D.score_pool(f'snap_cm_step{gen.step}', X, y, data, enc, sc, afib, replicates=20)
        m = res['gate_metrics']
        m['full_sd'] = res['feature_judges']['full']['sd']
        m['membership_auc_emb_task'] = {c: r['privacy_emb_task']['membership_auc']['value'] for c, r in res['by_class'].items()}
        m['nn_ratio_emb_task'] = {c: r['privacy_emb_task']['ratio_of_medians']['value'] for c, r in res['by_class'].items()}
        rows[gen.step] = m
        print('cm', gen.step, 'full', round(m['full_judge'], 3), '+-', round(m['full_sd'], 3), 'rhythm', round(m['rhythm_judge'], 3),
              'hrw', round(m['mean_hr_wasserstein'], 2), 'fail', round(m['mean_rhythm_failure'], 3),
              'cov', round(m['mean_coverage_task'], 3), 'memb', {c: round(v, 2) for c, v in m['membership_auc_emb_task'].items()},
              'nn_ratio', {c: round(v, 2) for c, v in m['nn_ratio_emb_task'].items()}, flush=True)
    cs = json.loads((RESULTS/'dev_snap_cs_step120000.json').read_text())
    ref = dict(full=cs['feature_judges']['full']['value'], sd=cs['feature_judges']['full']['sd'],
               membership={c: r['privacy_emb_task']['membership_auc']['value'] for c, r in cs['by_class'].items()})
    print('cs 120000 full', round(ref['full'], 3), '+-', round(ref['sd'], 3), 'memb', {c: round(v, 2) for c, v in ref['membership'].items()})
    write_json(OUT/'cm_checkpoints.json', dict(created=now(), protocol='snapshot protocol (63/class, CFG2, DDIM50, seed 9500)',
                                               rule='amendment 4: cm beats cs if full judge lower by more than max(SD)',
                                               cm=rows, cs_step120000=ref))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    dict(gates=gates, sweep=sweep, guidance=guidance, cm=cm)[sys.argv[1]]()
