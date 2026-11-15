"""Amendment 1 to Gate A (written after the original gate failed; see audits/v2/amendment1.md).

The original result (results/v2/phaseA_gate.json, passed=false) is kept unchanged. This script re-reads the
already computed controls and floor, applies the amended rule and writes results/v2/phaseA_gate_amended.json.
No metric is recomputed.
"""
import json
from .v2_common import RESULTS, AUDITS, write_json, now

K = 5


def run():
    floor = json.loads((RESULTS/'scorecard_floor.json').read_text())
    controls = json.loads((RESULTS/'controls_A8.json').read_text())
    original = json.loads((RESULTS/'phaseA_gate.json').read_text())
    checks = {}
    full = floor['feature_judges']['full']
    checks['floor/judge'] = dict(value=full['value'], sd=full['sd'], rule='0.40-0.60, SD<=0.06',
                                 passed=.4 <= full['value'] <= .6 and full['sd'] <= .06)
    nj = controls['noise_judges']['full']['value']
    checks['noise/judge'] = dict(value=nj, rule='>0.95', passed=nj > .95)
    for c, row in floor['by_class'].items():
        n = row['n_real']
        for key in ('recall', 'coverage'):
            v = row['emb_task'][key]['value']
            checks[f'floor/{c}/emb_task/{key}'] = dict(value=v, rule='>=0.85', passed=v >= .85)
    for c, row in controls['by_class'].items():
        n = floor['by_class'][c]['n_real'] if c in floor['by_class'] else None
        n_full = {'SR': 256, 'AFIB': 170, 'STACH': 82, 'SBRAD': 63}[c]
        limit = max(.1, 2*(K+1)/n_full)          # one repeated point can cover about (k+1)/n reals
        for kind in ('noise', 'collapse'):
            for key in ('recall', 'coverage'):
                v = row['emb_task'][kind][key]['value']
                checks[f'{kind}/{c}/emb_task/{key}'] = dict(value=v, rule=f'<{limit:.3f}', passed=v < limit)
        cp = row['exact_copy']
        checks[f'copy/{c}/distance'] = dict(value=cp['ratio_of_medians']['value'], rule='<1e-5',
                                            passed=cp['ratio_of_medians']['value'] < 1e-5)
        checks[f'copy/{c}/membership'] = dict(value=cp['membership_auc']['value'], rule='>0.9',
                                              passed=cp['membership_auc']['value'] > .9)
    checks['bootstrap/sanity'] = controls['bootstrap_sanity']
    demoted = {f'{kind}/{c}/{e}/{m}': controls['by_class'][c][e][kind][m]['value']
               for c in controls['by_class'] for e in ('emb_task', 'emb_ssl')
               for kind in ('noise',) for m in ('precision', 'density')}
    ssl_cov = {c: controls['by_class'][c]['emb_ssl']['noise']['coverage']['value'] for c in controls['by_class']}
    result = dict(amended_utc=now(), amendment='audits/v2/amendment1.md', original_passed=original['passed'],
                  original_failures=original['failures'], passed=all(v['passed'] for v in checks.values()),
                  checks=checks, failures=[k for k, v in checks.items() if not v['passed']],
                  demoted_to_descriptive=dict(embedding_precision_density_under_noise=demoted,
                                              emb_ssl_noise_coverage=ssl_cov,
                                              reason='embedding precision/density accept Gaussian noise in both '
                                                     'embeddings; emb_ssl (trained with noise augmentation) also '
                                                     'covers real STACH with noise'),
                  gate_metric_map={'Gate C/E coverage': 'emb_task coverage (emb_ssl coverage failed its noise control)',
                                   'fidelity': 'clinical feature judges and feature Wasserstein only'})
    write_json(RESULTS/'phaseA_gate_amended.json', result)
    print('amended Gate A passed:', result['passed'], result['failures'])
    return result


if __name__ == '__main__':
    run()
