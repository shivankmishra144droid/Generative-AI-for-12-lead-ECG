"""Configurable whole-ECG AE for the Phase B2 ladder: (B,12,1000) <-> (B,C,1000/2^n_down).

Same block structure as latent/vae.py (v1) when width=32, latent_channels=16, n_down=3; v1 stays untouched.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ECGAEv2(nn.Module):
    def __init__(self, width=32, latent_channels=16, condition_dim=10, n_down=3):
        super().__init__()
        self.config = dict(width=width, latent_channels=latent_channels, condition_dim=condition_dim, n_down=n_down)
        self.length = 1000 // 2**n_down
        chans = [width] + [2*width]*(n_down-1)
        enc = [nn.Conv1d(12, width, 7, padding=3), nn.SiLU()]
        prev = width
        for c in chans:
            enc += [nn.Conv1d(prev, c, 4, stride=2, padding=1), nn.SiLU()]
            prev = c
        enc += [nn.Conv1d(prev, 2*latent_channels, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)
        self.decoder_condition = nn.Linear(condition_dim, latent_channels)
        dec = [nn.Conv1d(latent_channels, prev, 3, padding=1), nn.SiLU()]
        for c in reversed([width] + chans[:-1]):
            dec += [nn.ConvTranspose1d(prev, c, 4, stride=2, padding=1), nn.SiLU()]
            prev = c
        dec += [nn.Conv1d(prev, 12, 7, padding=3)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        if x.ndim != 3 or x.shape[1:] != (12, 1000):
            raise ValueError('Expected channels-first (B,12,1000)')
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-12, 8)

    def decode(self, z, condition=None):
        if z.ndim != 3 or z.shape[1:] != (self.config['latent_channels'], self.length):
            raise ValueError(f"Expected (B,{self.config['latent_channels']},{self.length})")
        if condition is not None:
            z = z + self.decoder_condition(condition).unsqueeze(-1)
        return self.decoder(z)

    def forward(self, x, condition=None, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu)*torch.exp(.5*logvar) if sample else mu
        return self.decode(z, condition), mu, logvar


def stft_loss(a, b, windows=(32, 64, 128)):
    """Multi-resolution STFT magnitude L1 over all leads; a, b (B,12,T)."""
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    total = 0.
    for w in windows:
        win = torch.hann_window(w, device=a.device)
        sa = torch.stft(a, w, w//4, window=win, return_complex=True).abs()
        sb = torch.stft(b, w, w//4, window=win, return_complex=True).abs()
        total = total + F.l1_loss(sa, sb)
    return total/len(windows)


def ae_loss(rec, x, mu, logvar, beta=1e-4, stft=0.):
    mse = F.mse_loss(rec, x)
    slope = F.l1_loss(torch.diff(rec, dim=-1), torch.diff(x, dim=-1))
    kl = .5*(mu.square() + logvar.exp() - 1 - logvar).mean()
    loss = mse + .1*slope + beta*kl
    if stft:
        loss = loss + stft*stft_loss(rec, x)
    return loss, dict(mse=float(mse.detach()), kl=float(kl.detach()))
