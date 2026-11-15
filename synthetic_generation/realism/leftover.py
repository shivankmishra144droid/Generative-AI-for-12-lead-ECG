"""Study 4 leftover models beyond the stationary Gaussian process (Phase 2 escalation, D5 in report/WORKFLOW_4.md).

Phase 2-3 audit (NOTES_phase2_3.md): the GP leftover alone is detected at AUC 0.999 although its spectrum and its
per-recording scale spread match the real V2 leftover. What it lacks:
  - edge transients: real leftover power in the first and last second is 3-5x the middle (high-pass filter start-up
    and beats clipped by the edge, whose deviations stay in the leftover);
  - beat locking (class dependent): QRS / rest power ratio SR 1.1, AFIB 2.6, STACH 0.7, SBRAD 9.0;
  - heavy tails: kurtosis median 25-31 (7-19 without the edge seconds) against 3.0.

EnvelopeLeftover multiplies a stationary GP sample by sqrt(g(t) * h(offset from the nearest R-peak)), with g the
per-time power profile and h the beat-phase power profile of the real leftover (both normalised to mean 1), so the
expected power follows real edge and beat structure. Beat locking also raises the kurtosis of the mixture."""
from __future__ import annotations

import numpy as np

OFF_LO, OFF_HI = -50, 100                        # samples from the nearest R-peak (-0.5 ... +1.0 s)


def r_offsets(r: np.ndarray, T: int = 1000) -> np.ndarray:
    """Offset of every sample from the preceding R-peak, or from the next one when that is closer than 0.5 s before
    it; clipped to [OFF_LO, OFF_HI]. With no R-peaks: OFF_HI everywhere (between beats)."""
    t = np.arange(T)
    if not len(r):
        return np.full(T, OFF_HI)
    k = np.clip(np.searchsorted(r, t, side="right") - 1, 0, len(r) - 1)
    off = t - r[k]
    nxt = np.clip(k + 1, 0, len(r) - 1)
    before = t - r[nxt]
    off = np.where((before >= OFF_LO) & (before < 0) & (nxt != k), before, off)
    return np.clip(off, OFF_LO, OFF_HI)


def smooth(v: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w) / w
    return np.convolve(np.pad(v, w // 2, mode="edge"), k, mode="valid")[:len(v)]


class EnvelopeLeftover:
    def __init__(self, gp, leftovers: list, r_list: list, edge: bool = True, beat: bool = True, w_t: int = 9,
                 w_off: int = 5):
        """gp: a fitted stationary process with .sample(rng) -> (8, T); leftovers[i] (8, T) real leftovers with
        R-peaks r_list[i]. g(t): pooled power per time sample; h(offset): power per beat phase after dividing out g.
        Each is smoothed and normalised to mean 1 (h over the pooled offset distribution)."""
        self.gp = gp
        L = np.stack(leftovers)
        T = L.shape[2]
        p = (L ** 2).mean(1)                                                 # (n, T) power per sample
        g = smooth(p.mean(0), w_t) if edge else np.ones(T)
        self.g = g / g.mean()
        n_off = OFF_HI - OFF_LO + 1
        s, n = np.zeros(n_off), np.zeros(n_off)
        for pi, r in zip(p, r_list):
            o = r_offsets(r, T) - OFF_LO
            s += np.bincount(o, weights=pi / self.g, minlength=n_off)
            n += np.bincount(o, minlength=n_off)
        h = smooth(s / np.maximum(n, 1), w_off) if beat else np.ones(n_off)
        self.h = h / ((h * n).sum() / n.sum())
        self.edge, self.beat = edge, beat

    def envelope(self, r: np.ndarray, T: int = 1000) -> np.ndarray:
        return np.sqrt(self.g[:T] * self.h[r_offsets(r, T) - OFF_LO])

    def sample(self, rng, r: np.ndarray, T: int = 1000) -> np.ndarray:
        return self.gp.sample(rng) * self.envelope(r, T)[None]


def phase_randomise(x: np.ndarray, rng) -> np.ndarray:
    """Same random phase per frequency on every lead: keeps each lead's spectrum and the cross-lead cross-spectrum,
    removes time-domain structure (edges, beat locking, heavy tails). Diagnostic only."""
    F = np.fft.rfft(x, axis=-1)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape[-1]))
    ph[0] = 1
    if x.shape[-1] % 2 == 0:
        ph[-1] = 1
    return np.fft.irfft(F * ph, n=x.shape[-1], axis=-1)
