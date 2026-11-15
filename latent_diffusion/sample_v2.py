"""Sampling from v2 diffusion checkpoints (EMA 0.999), decoding with the chosen AE, limb projection."""
import json
import numpy as np
import torch
from .common import ART as V1_ART, digest
from .v2_common import ART, training_data
from .latent.latent_diffusion_v2 import LatentSSSDv2, LatentDiffusionV2
from .ae_ladder_v2 import load_rung
from .generate import project_leads
from .train import decoder_condition
from .rhythm_v2 import train_peaks, rmap, rr_bank_peaks


class Generator:
    def __init__(self, name, device, ckpt='last.pt', ema='ema0.999'):
        path = ART/f'diffusion_{name}'/ckpt
        ck = torch.load(path, map_location=device, weights_only=False)
        self.config, self.step = ck['config'], ck['step']
        self.model = LatentDiffusionV2(LatentSSSDv2(**ck['config'])).to(device).eval()
        self.model.load_state_dict(ck[ema])
        info = ck['ae']
        if digest(ART/f"ae_{info['rung']}"/'best.pt') != info['sha256']:   # portable: checkpoints may come from Colab
            raise ValueError('AE changed since diffusion training')
        self.ae = load_rung(info['rung'], device)
        self.mean, self.std = ck['mean'].to(device), ck['std'].to(device)
        self.device, self.name, self.sha = device, name, digest(path)
        self.rhythm = ck['config']['rhythm']
        self.L = ck['config']['length']

    @torch.no_grad()
    def __call__(self, labels, values, peaks=None, guidance=2., steps=50, eta=0., seed=9300, batch=64):
        out = []
        for i in range(0, len(labels), batch):
            lb = torch.as_tensor(labels[i:i+batch], device=self.device)
            vb = torch.as_tensor(values[i:i+batch], device=self.device)
            rb = None
            if self.rhythm:
                rows = peaks[i:i+batch]
                rb = torch.from_numpy(rmap(rows, np.ones(len(rows), bool), self.L)).to(self.device)
            z = self.model.sample(lb, vb, rb, steps, guidance, seed + i, eta)
            z = z*self.std + self.mean
            out.append(self.ae.decode(z, decoder_condition(lb, vb)).transpose(1, 2).cpu().numpy())
        return project_leads(np.concatenate(out)).astype(np.float32)


def fit_requests(counts, seed):
    """Per class `counts[c]` fit rows with a plausible R-peak detection: labels, condition vectors and the
    rr_bank R positions of the same row (shifted real RR sequence), so HR and R-map agree."""
    x, y, patient, fit, dev = training_data()
    cvals = np.load(V1_ART/'train_conditions.npz')['values']
    peaks, valid = train_peaks()
    rng = np.random.default_rng(seed)
    rows = np.concatenate([rng.choice(fit[(y[fit] == c) & valid[fit]], n, replace=True) for c, n in enumerate(counts) if n])
    return y[rows], cvals[rows].copy(), rr_bank_peaks(peaks, valid, rows, rng), rows


def detected_peaks(X):
    from synthetic_generation.beats import detect_r_peaks
    from synthetic_generation.leads import to_independent_8
    return [np.asarray(detect_r_peaks(s)) for s in to_independent_8(np.asarray(X, np.float32))]


def adherence(requested, detected, tol=5):
    """One-to-one matching of requested vs detected R positions (tolerance tol samples = 50 ms)."""
    total = missed = extra = 0
    any_err = 0
    for req, det in zip(requested, detected):
        r = req[req >= 0]
        edges = sorted((abs(int(a)-int(b)), i, j) for i, a in enumerate(r) for j, b in enumerate(det) if abs(int(a)-int(b)) <= tol)
        ri, di = set(), set()
        for _, i, j in edges:
            if i not in ri and j not in di:
                ri.add(i); di.add(j)
        m, e = len(r)-len(ri), len(det)-len(di)
        total += len(r); missed += m; extra += e; any_err += (m + e) > 0
    return dict(requested_beats=total, missed_fraction=missed/max(total, 1), extra_per_beat=extra/max(total, 1),
                ecg_any_error_fraction=any_err/max(len(requested), 1), tolerance_ms=tol*10)
