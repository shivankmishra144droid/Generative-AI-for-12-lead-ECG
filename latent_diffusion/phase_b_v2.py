"""Phase B1: realism of v1 AE reconstructions of the development reference cohort (paired, grouped by patient)."""
import json
import numpy as np
import torch
from .common import ART as V1_ART, digest
from .v2_common import ART, RESULTS, AUDITS, SEED, training_data, register, write_json, gpu_lease, now
from .phase_a_v2 import cohort_setup, prepare
from .gen_eval import paired_rpeak_check
from .gen_eval_v2 import score_prepared
from .generate import load_ae, project_leads
from .train import decoder_condition


@torch.no_grad()
def reconstruct(ae, x, y, cond, device, batch=64):
    out = []
    for i in range(0, len(x), batch):
        a = torch.from_numpy(np.asarray(x[i:i+batch], np.float32)).transpose(1, 2).to(device)
        c = decoder_condition(torch.as_tensor(y[i:i+batch], device=device), torch.as_tensor(cond[i:i+batch], device=device))
        out.append(ae(a, c, sample=False)[0].transpose(1, 2).cpu().numpy())
    return project_leads(np.concatenate(out))


def run(tag='v1ae', loader=None):
    """loader(device) -> AE; None = the frozen v1 AE."""
    register(f'phaseB1_{tag}', dict(ae=tag, cohort='real_full (256/170/82/63 dev)', paired=True,
                                    gate='full judge <=0.65, every group <=0.75, missed R-peaks <=3 %'))
    data, encoders, scalers, afib_cv = cohort_setup()
    real, fit_ref = data['real_full'], data['fit_reference']
    x, y, patient, fit, dev = training_data()
    rows = np.load(ART/'evaluation_cohorts.npz')['full']
    cond = np.load(V1_ART/'train_conditions.npz')['values'][rows]
    assert np.array_equal(y[rows], real['y'])
    path = ART/'pools'/f'recon_{tag}.npz'
    if not path.exists():
        with gpu_lease(f'phaseB1_{tag}') as device:
            ae = load_ae('ae', device) if loader is None else loader(device)
            rec = reconstruct(ae, x[rows], y[rows], cond, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=rec.astype(np.float32), y=y[rows])
    rec = np.load(path)['X']
    syn = prepare(f'recon_{tag}', rec, real['y'], real['groups'], encoders, scalers, afib_cv)
    result = score_prepared(real, syn, fit_ref, ('emb_task', 'emb_ssl'), afib_cv, paired=True, privacy=False)
    result['rpeak'] = paired_rpeak_check(np.asarray(real['X']), rec)
    result['mse'] = float(np.mean((rec - real['X'])**2))
    fj = result['feature_judges']
    gate = dict(full=fj['full']['value'], groups={k: v['value'] for k, v in fj.items() if k != 'full'},
                missed=result['rpeak']['missed_fraction'])
    gate['passed'] = gate['full'] <= .65 and max(gate['groups'].values()) <= .75 and gate['missed'] <= .03
    result['gate_B1'] = gate
    result['ae_sha256'] = digest(V1_ART/'ae/best.pt') if loader is None else None
    write_json(RESULTS/f'ae_ceiling_{tag}.json', result)
    print(tag, json.dumps(gate), flush=True)
    return result


if __name__ == '__main__':
    run()
