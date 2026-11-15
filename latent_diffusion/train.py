"""Reproducible, resumable Study 9 training; only study-owned checkpoints are loaded.

python -m latent_diffusion.train ae --epochs 15
python -m latent_diffusion.train ae --name vae --beta .01 --epochs 15
python -m latent_diffusion.train diffusion --steps 3000 --resume
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.nn import functional as F

from latent_diffusion.latent.vae import ECGVAE, vae_loss
from .common import ART, SEED, audit, digest, load_split, save_checkpoint, seed_all, write_json


def decoder_condition(y, values):
    return torch.cat((F.one_hot(y, 4).float(), values), dim=1)


def tensors(x, y, c, rows, device):
    return (torch.from_numpy(np.asarray(x[rows], np.float32).copy()).transpose(1, 2).to(device),
            torch.as_tensor(y[rows], device=device), torch.as_tensor(c[rows], device=device))


def device_for(name):
    device = torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else
                          'cpu' if name == 'auto' else name)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    return device


def memory(device):
    return dict(device=str(device), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20) if device.type == 'cuda' else dict(device='cpu')


def train_ae(args):
    seed_all(args.seed)
    device = device_for(args.device)
    x, y = load_split('train')
    c = np.load(ART / 'train_conditions.npz')['values']
    part = np.load(ART / 'partition.npz')
    fit, dev = part['fit'], part['dev']
    model = ECGVAE(width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    history, best, start = [], float('inf'), 0
    elapsed_before = 0.
    folder = ART / args.name
    if args.resume and (folder / 'last.pt').exists():
        ck = torch.load(folder / 'last.pt', map_location=device, weights_only=False)
        if ck['config'] != model.config or ck['beta'] != args.beta or ck['seed'] != args.seed:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        history, best, start = ck['history'], ck['best'], ck['epoch']
        elapsed_before = ck.get('elapsed_seconds', 0.)
        rng.bit_generator.state = ck['numpy_rng']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
        if start >= args.epochs:
            print(f'{args.name}: requested {args.epochs} epochs already complete; preserving training evidence', flush=True)
            return
    t0 = time.monotonic()
    for epoch in range(start, args.epochs):
        model.train()
        total, count = 0., 0
        for rows in np.array_split(rng.permutation(fit), max(1, int(np.ceil(len(fit)/args.batch)))):
            a, b, v = tensors(x, y, c, rows, device)
            opt.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(a, decoder_condition(b, v))
            loss, _ = vae_loss(reconstruction, a, mu, logvar, args.beta)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite VAE loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach())*len(rows); count += len(rows)
        model.eval()
        error, n = 0., 0
        with torch.no_grad():
            for rows in np.array_split(dev, max(1, int(np.ceil(len(dev)/args.batch)))):
                a, b, v = tensors(x, y, c, rows, device)
                recon, _, _ = model(a, decoder_condition(b, v), sample=False)
                error += float(F.mse_loss(recon, a))*len(rows); n += len(rows)
        improved = error/n < best
        best = min(best, error/n)
        history.append(dict(epoch=epoch+1, train_loss=total/count, dev_mse=error/n))
        ck = dict(config=model.config, model=model.state_dict(), optimizer=opt.state_dict(),
                  epoch=epoch+1, beta=args.beta, history=history, best=best, seed=args.seed,
                  elapsed_seconds=elapsed_before+time.monotonic()-t0,
                  numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        save_checkpoint(folder / 'last.pt', ck)
        if improved:
            save_checkpoint(folder / 'best.pt', ck)
        write_json(folder / 'history.json', history)
        print(args.name, history[-1], flush=True)
    evidence = dict(history=history, elapsed_seconds=elapsed_before+time.monotonic()-t0, **memory(device))
    write_json(folder / 'training.json', evidence)
    audit(f'phase2_{args.name}', evidence)


@torch.no_grad()
def encode_latents(ae, x, rows, device, batch):
    result = []
    for start in range(0, len(rows), batch):
        a = torch.from_numpy(np.asarray(x[rows[start:start+batch]], np.float32).copy()).transpose(1, 2).to(device)
        result.append(ae.encode(a)[0].cpu())
    return torch.cat(result)


def train_diffusion(args):
    from latent_diffusion.latent.latent_diffusion import LatentSSSD, LatentDiffusion
    seed_all(args.seed)
    device = device_for(args.device)
    ae_path = ART / 'ae' / 'best.pt'
    ck = torch.load(ae_path, map_location='cpu', weights_only=False)
    ae = ECGVAE(**ck['config']).to(device).eval().requires_grad_(False)
    ae.load_state_dict(ck['model'])
    x, y = load_split('train')
    c = np.load(ART / 'train_conditions.npz')['values']
    part = np.load(ART / 'partition.npz')
    fit, dev = part['fit'], part['dev']
    rawfit = encode_latents(ae, x, fit, device, 64)
    rawdev = encode_latents(ae, x, dev, device, 64)
    mean = rawfit.mean((0, 2), keepdim=True)
    std = rawfit.std((0, 2), keepdim=True).clamp_min(1e-4)
    zfit, zdev = (rawfit-mean)/std, (rawdev-mean)/std
    net = LatentSSSD(width=args.width, layers=args.layers).to(device)
    model = LatentDiffusion(net).to(device)
    # Legacy weight_norm stores non-leaf tensors and cannot be deep-copied.
    ema = LatentDiffusion(LatentSSSD(**net.config)).to(device).eval().requires_grad_(False)
    ema.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    rng = np.random.default_rng(args.seed)
    history, start = [], 0
    elapsed_before = 0.
    folder = ART / 'diffusion'
    if args.resume and (folder / 'last.pt').exists():
        old = torch.load(folder / 'last.pt', map_location=device, weights_only=False)
        if old['config'] != net.config or old['ae_sha256'] != digest(ae_path) or old['seed'] != args.seed:
            raise ValueError('Diffusion resume model/AE/seed mismatch')
        model.load_state_dict(old['model']); ema.load_state_dict(old['ema']); opt.load_state_dict(old['optimizer'])
        history, start = old['history'], old['step']
        elapsed_before = old.get('elapsed_seconds', 0.)
        rng.bit_generator.state = old['numpy_rng']
        torch.set_rng_state(old['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in old['cuda_rng']])
        if start >= args.steps:
            print(f'diffusion: requested {args.steps} steps already complete; preserving training evidence', flush=True)
            return
    t0 = time.monotonic()
    model.train()
    for step in range(start, args.steps):
        j = rng.integers(len(fit), size=args.batch)
        labels = torch.as_tensor(y[fit[j]], device=device)
        vals = torch.as_tensor(c[fit[j]], device=device)
        opt.zero_grad(set_to_none=True)
        loss = model.loss(zfit[j].to(device), labels, vals)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite diffusion loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, .005)
        if (step+1) % 50 == 0 or step+1 == args.steps:
            # Fixed validation noise makes the diagnostic comparable across steps.
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                torch.manual_seed(args.seed+1)
                with torch.no_grad():
                    ndev = min(32, len(dev))
                    model.eval()
                    devloss = model.loss(zdev[:ndev].to(device), torch.as_tensor(y[dev[:ndev]], device=device),
                                         torch.as_tensor(c[dev[:ndev]], device=device))
                    model.train()
            history.append(dict(step=step+1, train_loss=float(loss.detach()), dev_loss=float(devloss)))
            payload = dict(config=net.config, model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                           mean=mean, std=std, history=history, step=step+1, seed=args.seed,
                           elapsed_seconds=elapsed_before+time.monotonic()-t0,
                           ae_sha256=digest(ae_path), numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
            save_checkpoint(folder / 'last.pt', payload)
            write_json(folder / 'history.json', history)
            print('diffusion', history[-1], 'elapsed', round(time.monotonic()-t0), flush=True)
        if (ART / 'STOP').exists():
            # Stop only at a durable checkpoint, no checkpoint silently discarded.
            if (step+1) % 50 == 0:
                break
    evidence = dict(history=history, elapsed_seconds=elapsed_before+time.monotonic()-t0, terminal_alpha_bar=float(model.alpha_bar[-1]),
                    latent_shape=list(zfit.shape), **memory(device))
    write_json(folder / 'training.json', evidence)
    audit('phase3_diffusion', evidence)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['ae', 'diffusion'])
    p.add_argument('--name', choices=['ae', 'vae'], default='ae')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--beta', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--device', default='auto')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.steps, args.batch) < 1:
        p.error('Epochs, steps and batch must be positive')
    (train_ae if args.stage == 'ae' else train_diffusion)(args)


if __name__ == '__main__':
    main()
