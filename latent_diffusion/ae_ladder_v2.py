"""Phase B2 ladder: train AE rungs in order, judge each with Phase B1, stop at the first that passes.

Rungs (STUDY9_V2_INSTRUCTIONS.md B2): 1 = 60 epochs cosine LR; 2 = + multi-resolution STFT loss 0.1;
3 = + less compression (16x250); 4 = + width 64. Rung 5 (adversarial) is not run (amendment 1 time scope).
Epoch selection: lowest dev MSE. Fit patients train, dev patients select; validation never loaded.
"""
import json
import math
import time
import numpy as np
import torch
from .common import ART as V1_ART, digest
from .v2_common import ART, RESULTS, AUDITS, SEED, training_data, register, write_json, gpu_lease, seed_all, save_checkpoint
from .latent.vae_v2 import ECGAEv2, ae_loss
from .train import decoder_condition

RUNGS = {
    'r1': dict(epochs=60, stft=0., n_down=3, width=32),
    'r2': dict(epochs=60, stft=.1, n_down=3, width=32),
    'r3': dict(epochs=60, stft=.1, n_down=2, width=32),
    'r4': dict(epochs=60, stft=.1, n_down=2, width=64),
}


def load_rung(name, device='cpu'):
    ck = torch.load(ART/f'ae_{name}'/'best.pt', map_location=device, weights_only=False)
    m = ECGAEv2(**ck['config']).to(device).eval().requires_grad_(False)
    m.load_state_dict(ck['model'])
    return m


def train_rung(name, cfg, batch=32):
    folder = ART/f'ae_{name}'
    if (folder/'best.pt').exists() and (folder/'done.json').exists():
        return json.loads((folder/'done.json').read_text())
    register(f'phaseB2_{name}', dict(**cfg, batch=batch, lr='AdamW 1e-3 cosine to 1e-5', beta=1e-4, seed=SEED,
                                     selection='lowest dev MSE (mean code, true condition)'))
    seed_all(SEED)
    x, y, patient, fit, dev = training_data()
    c = np.load(V1_ART/'train_conditions.npz')['values']
    X = torch.from_numpy(np.asarray(x, np.float32)).transpose(1, 2).contiguous()
    with gpu_lease(f'phaseB2_{name}') as device:
        model = ECGAEv2(width=cfg['width'], n_down=cfg['n_down']).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        steps_per = math.ceil(len(fit)/batch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs']*steps_per, eta_min=1e-5)
        rng = np.random.default_rng(SEED)
        best, hist, t0 = float('inf'), [], time.monotonic()
        yt, ct = torch.as_tensor(y), torch.as_tensor(c)
        for ep in range(cfg['epochs']):
            model.train()
            for rows in np.array_split(rng.permutation(fit), steps_per):
                a = X[rows].to(device)
                cond = decoder_condition(yt[rows].to(device), ct[rows].to(device))
                rec, mu, lv = model(a, cond)
                loss, _ = ae_loss(rec, a, mu, lv, 1e-4, cfg['stft'])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                sched.step()
            model.eval()
            err = 0.
            with torch.no_grad():
                for rows in np.array_split(dev, math.ceil(len(dev)/64)):
                    a = X[rows].to(device)
                    rec = model(a, decoder_condition(yt[rows].to(device), ct[rows].to(device)), sample=False)[0]
                    err += float(torch.nn.functional.mse_loss(rec, a))*len(rows)
            err /= len(dev)
            hist.append(dict(epoch=ep+1, dev_mse=err))
            if err < best:
                best = err
                save_checkpoint(folder/'best.pt', dict(config=model.config, model=model.state_dict(), epoch=ep+1,
                                                       dev_mse=err, rung=name, cfg=cfg))
            if (ep+1) % 10 == 0:
                print(name, ep+1, round(err, 5), flush=True)
        out = dict(rung=name, cfg=cfg, best_dev_mse=best, history=hist, seconds=time.monotonic()-t0,
                   peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    write_json(folder/'done.json', out)
    return out


def run():
    from .phase_b_v2 import run as judge
    summary = {}
    for name, cfg in RUNGS.items():
        info = train_rung(name, cfg)
        res = judge(tag=f'ae_{name}', loader=lambda device, n=name: load_rung(n, device))
        summary[name] = dict(train={k: info[k] for k in ('best_dev_mse', 'seconds', 'peak_reserved_mib')},
                             gate=res['gate_B1'], sha256=digest(ART/f'ae_{name}'/'best.pt'))
        write_json(RESULTS/'ae_ladder.json', summary)
        print(name, summary[name], flush=True)
        if res['gate_B1']['passed']:
            break
    passing = [n for n, s in summary.items() if s['gate']['passed']]
    chosen = passing[0] if passing else min(summary, key=lambda n: summary[n]['gate']['full'])
    write_json(ART/'chosen_ae.json', dict(rung=chosen, path=str(ART/f'ae_{chosen}'/'best.pt'),
                                         sha256=summary[chosen]['sha256'], passed=bool(passing),
                                         rule='first passing rung; else lowest full judge AUC'))
    print('chosen', chosen, 'passed', bool(passing))


if __name__ == '__main__':
    run()
