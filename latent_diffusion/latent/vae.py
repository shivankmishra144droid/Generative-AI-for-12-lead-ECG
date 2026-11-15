"""Small whole-ECG convolutional VAE: (B,12,1000) <-> (B,16,125)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ECGVAE(nn.Module):
    def __init__(self, width=32, latent_channels=16, condition_dim=10):
        super().__init__()
        self.config = dict(width=width, latent_channels=latent_channels, condition_dim=condition_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(12, width, 7, padding=3), nn.SiLU(),
            nn.Conv1d(width, width, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(width, 2*width, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(2*width, 2*width, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(2*width, 2*latent_channels, 3, padding=1))
        self.decoder_condition = nn.Linear(condition_dim, latent_channels)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, 2*width, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose1d(2*width, 2*width, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose1d(2*width, width, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(width, 12, 7, padding=3))

    def encode(self, x):
        if x.ndim != 3 or x.shape[1:] != (12, 1000):
            raise ValueError('Expected channels-first (B,12,1000)')
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-12, 8)

    def decode(self, z, condition=None):
        if z.ndim != 3 or z.shape[1:] != (self.config['latent_channels'], 125):
            raise ValueError('Expected (B,latent_channels,125)')
        if condition is not None:
            z = z + self.decoder_condition(condition).unsqueeze(-1)
        return self.decoder(z)

    def forward(self, x, condition=None, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(.5*logvar) if sample else mu
        return self.decode(z, condition), mu, logvar


def vae_loss(reconstruction, x, mu, logvar, beta=1e-4):
    mse = F.mse_loss(reconstruction, x)
    slope = F.l1_loss(torch.diff(reconstruction, dim=-1), torch.diff(x, dim=-1))
    kl = .5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
    return mse + .1*slope + beta*kl, {'mse': mse.detach(), 'slope_l1': slope.detach(), 'kl': kl.detach()}
