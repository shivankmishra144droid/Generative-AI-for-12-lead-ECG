"""Phases C/D: class-balanced, HR-stratified latent diffusion on the chosen AE (artifacts/v2/chosen_ae.json).

py -3.14 -m latent_diffusion.train_diffusion_v2 --name cs --layers 8 --width 64 --state 32 --steps 60000
py -3.14 -m latent_diffusion.train_diffusion_v2 --name ds --layers 8 --width 64 --state 32 --steps 60000 --rhythm
Fit patients train; dev patients give the fixed per-noise-level loss diagnostics. Validation is never loaded.
Resumable from artifacts/v2/diffusion_<name>/last.pt; artifacts/v2/STOP stops at the next 1,000-step checkpoint.
"""
import argparse
import copy
import json
import shutil
import time
import numpy as np
import torch
from .common import ART as V1_ART, digest
from .v2_common import ART, RESULTS, AUDITS, SEED, training_data, register, write_json, gpu_lease, seed_all, save_checkpoint
from .latent.latent_diffusion_v2 import LatentSSSDv2, LatentDiffusionV2
from .ae_ladder_v2 import load_rung
from .rhythm_v2 import train_peaks, rmap

T_BINS = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]


def chosen_ae(device):
    info = json.loads((ART/'chosen_ae.json').read_text())
    if digest(ART/f"ae_{info['rung']}"/'best.pt') != info['sha256']:     # the file load_rung loads; portable across machines
        raise ValueError('Chosen AE changed')
    return load_rung(info['rung'], device), info


@torch.no_grad()
def encode(ae, x, rows, device, batch=128):
    out = []
    for i in range(0, len(rows), batch):
        a = torch.from_numpy(np.asarray(x[rows[i:i+batch]], np.float32)).transpose(1, 2).to(device)
        out.append(ae.encode(a)[0].cpu())
    return torch.cat(out)


def sampling_weights(y, hr, rows):
    """Class uniform; within class proportional to 1/density of 10-bpm HR bins, clipped to [0.2, 5]."""
    w = np.zeros(len(rows))
    for c in range(4):
        idx = np.flatnonzero(y[rows] == c)
        h = hr[rows[idx]]
        ok = np.isfinite(h)
        bins = np.floor(np.where(ok, h, 0)/10).astype(int)
        uniq, counts = np.unique(bins[ok], return_counts=True)
        dens = dict(zip(uniq, counts/ok.sum()))
        raw = np.array([1/dens[b] if o else np.nan for b, o in zip(bins, ok)])
        raw /= np.nanmedian(raw)
        raw = np.clip(raw, .2, 5)
        raw[~ok] = 1.                                   # missing HR: median weight (amendment 1)
        w[idx] = raw/raw.sum()/4
    return w/w.sum()


def train(args):
    proto = dict(name=args.name, layers=args.layers, width=args.width, state=args.state, steps=args.steps,
                 batch=args.batch, rhythm=args.rhythm, lr='AdamW 2e-4, 1000-step linear warmup, constant',
                 ema=[.999, .995], sampling='class uniform x inverse HR-bin density, clip [0.2,5]',
                 dropout='vector .1, map .1, both .05 (independent)', seed=args.seed)
    # amendment 3: continuing past 60 k registers separately and resumes a run whose protocol differs only in steps
    ext = args.name in ('cs', 'ds') and args.steps != 60000
    register(f'phaseCD_{args.name}' + (f'_ext{args.steps}' if ext else ''), proto)
    seed_all(args.seed)
    folder = ART/f'diffusion_{args.name}'
    x, y, patient, fit, dev = training_data()
    cond_all = np.load(V1_ART/'train_conditions.npz')
    cvals, hr = cond_all['values'], cond_all['hr']
    with gpu_lease(f'phaseCD_{args.name}') as device:
        ae, ae_info = chosen_ae(device)
        cache = ART/'latents_chosen_ae.pt'
        if cache.exists() and torch.load(cache, weights_only=False)['ae_sha256'] == ae_info['sha256']:
            lat = torch.load(cache, weights_only=False)
        else:
            allrows = np.arange(len(y))
            raw = encode(ae, x, allrows, device)
            mean = raw[fit].mean((0, 2), keepdim=True)
            std = raw[fit].std((0, 2), keepdim=True).clamp_min(1e-4)
            lat = dict(z=(raw - mean)/std, mean=mean, std=std, ae_sha256=ae_info['sha256'])
            torch.save(lat, cache)
        Z = lat['z']
        L = Z.shape[-1]
        peaks, valid = train_peaks()
        R = torch.from_numpy(rmap(peaks, valid, L)) if args.rhythm else None
        net = LatentSSSDv2(channels=Z.shape[1], length=L, width=args.width, layers=args.layers,
                           state=args.state, rhythm=args.rhythm).to(device)
        model = LatentDiffusionV2(net).to(device)
        emas = {d: LatentDiffusionV2(LatentSSSDv2(**net.config)).to(device).eval().requires_grad_(False)
                for d in (.999, .995)}
        for e in emas.values():
            e.load_state_dict(model.state_dict())
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=2e-4)
        w = sampling_weights(y, hr, fit)
        rng = np.random.default_rng(args.seed)
        start, history, elapsed = 0, [], 0.
        if (folder/'last.pt').exists():
            ck = torch.load(folder/'last.pt', map_location=device, weights_only=False)
            if {k: v for k, v in ck['protocol'].items() if k != 'steps'} != {k: v for k, v in proto.items() if k != 'steps'}:
                raise ValueError('Resume protocol mismatch')
            model.load_state_dict(ck['model'])
            for d, e in emas.items():
                e.load_state_dict(ck[f'ema{d}'])
            opt.load_state_dict(ck['optimizer'])
            start, history, elapsed = ck['step'], ck['history'], ck['elapsed']
            rng.bit_generator.state = ck['numpy_rng']
            torch.set_rng_state(ck['torch_rng'].cpu())
            torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        # fixed dev diagnostic cohort: 63 per class
        drng = np.random.default_rng(SEED + 7)
        drows = np.concatenate([drng.choice(dev[y[dev] == c], 63, replace=False) for c in range(4)])
        g = torch.Generator(device='cpu').manual_seed(SEED + 8)
        dnoise = torch.randn((len(T_BINS), len(drows), *Z.shape[1:]), generator=g)
        dts = [torch.randint(a, b, (len(drows),), generator=g) for a, b in T_BINS]
        ddrop = torch.rand(len(T_BINS), len(drows), 3, generator=g)
        snaps = {int(args.steps*f) for f in (.25, .5, .75, 1.)}
        t0 = time.monotonic()
        model.train()
        for step in range(start, args.steps):
            rows = fit[rng.choice(len(fit), args.batch, p=w)]
            zb = Z[rows].to(device)
            lb = torch.as_tensor(y[rows], device=device)
            vb = torch.as_tensor(cvals[rows], device=device)
            rb = R[rows].to(device) if R is not None else None
            for gparam in opt.param_groups:
                gparam['lr'] = 2e-4*min(1., (step + 1)/1000)
            loss = model.loss(zb, lb, vb, rb)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite diffusion loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            with torch.no_grad():
                mp = list(model.parameters())
                for d, e in emas.items():
                    for ep, p in zip(e.parameters(), mp):
                        ep.lerp_(p, 1 - d)
            if (step + 1) % 1000 == 0 or step + 1 == args.steps:
                e = emas[.999]
                bins = []
                with torch.no_grad():
                    for j in range(len(T_BINS)):
                        dz = Z[drows].to(device)
                        noisy, target = e.noise_target(dz, dnoise[j].to(device), dts[j].to(device))
                        u = ddrop[j].to(device)
                        dv = (u[:, 0] < .1) | (u[:, 2] < .05)
                        dm = (u[:, 1] < .1) | (u[:, 2] < .05)
                        rm = R[drows].to(device)*(~dm).view(-1, 1, 1) if R is not None else None
                        pred = e.denoiser(noisy, dts[j].to(device), torch.as_tensor(y[drows], device=device),
                                          torch.as_tensor(cvals[drows], device=device), dv, rm)
                        bins.append(float((pred - target).square().mean()))
                history.append(dict(step=step + 1, train_loss=float(loss.detach()), dev_loss_bins=bins,
                                    dev_loss=float(np.mean(bins))))
                now_elapsed = elapsed + time.monotonic() - t0
                payload = dict(protocol=proto, config=net.config, model=model.state_dict(),
                               **{f'ema{d}': e.state_dict() for d, e in emas.items()}, optimizer=opt.state_dict(),
                               step=step + 1, history=history, elapsed=now_elapsed, mean=lat['mean'], std=lat['std'],
                               ae=ae_info, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state())
                save_checkpoint(folder/'last.pt', payload)
                if step + 1 in snaps:
                    shutil.copyfile(folder/'last.pt', folder/f'step{step + 1}.pt')
                write_json(RESULTS/f'diffusion_training_{args.name}.json',
                           dict(protocol=proto, history=history, elapsed_seconds=now_elapsed,
                                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, latent_shape=list(Z.shape)))
                print(args.name, history[-1]['step'], round(history[-1]['dev_loss'], 4),
                      [round(b, 3) for b in bins], f'{now_elapsed/60:.1f} min', flush=True)
                if (ART/'STOP').exists():
                    print('STOP file: stopping at checkpoint', flush=True)
                    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--state', type=int, default=32)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--rhythm', action='store_true')
    train(p.parse_args())


if __name__ == '__main__':
    main()
