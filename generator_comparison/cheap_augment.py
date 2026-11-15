"""Cheap-augmentation baseline for study 5 (report/PREREGISTRATION_5.md, condition `augment`).

Each augmented copy is one real minority ECG (the same source rows `oversample` duplicates) with standard
time-series transforms, magnitudes fixed in advance (MAG):

  stretch   slow the whole recording by f ~ U(1, 1 + stretch), then crop a random 1,000-sample window.
            Only AFIB and SBRAD: slowing keeps SBRAD < 60 bpm and AFIB irregular; slowing STACH could push it below
            100 bpm, so STACH is never warped. Nothing is ever sped up, so no padding is needed.
  scale     global amplitude x U(1 - scale, 1 + scale), and each independent lead x U(1 - lead_scale, 1 + lead_scale)
  wander    baseline wander: one frequency U(0.05, 0.5) Hz per copy, amplitude U(0, wander) and phase per lead
  noise     white Gaussian noise, sigma ~ U(0, noise) per copy

Amplitude transforms are made on the 8 independent leads (I, II, V1-V6) and carried to the derived leads through
reconstruct_12, so III = II - I and the Goldberger relations still hold. They are added as a difference,
x + reconstruct_12(x8 + d8) - reconstruct_12(x8), so with every magnitude at 0 a copy is bit-identical to its
source. Cross-recording mixup is not used: adding two recordings with different beat times roughly doubles the
apparent rate, which relabels SBRAD.
"""
from __future__ import annotations

import numpy as np

from synthetic_generation.leads import reconstruct_12, to_independent_8

FS = 100
MAG = {"stretch": 0.10, "scale": 0.10, "lead_scale": 0.05, "wander": 0.10, "noise": 0.05}
STRETCH_CLASSES = (1, 3)                        # AFIB, SBRAD; never STACH (2) or SR (0)


def stretch(x12: np.ndarray, f: float, start_frac: float) -> np.ndarray:
    """Slow (T, 12) by f >= 1 (linear interpolation), then crop 1,000 samples starting at start_frac of the slack."""
    t = x12.shape[0]
    n = int(round(t * f))
    if n == t:
        return x12
    grid = np.linspace(0, t - 1, n)
    y = np.stack([np.interp(grid, np.arange(t), x12[:, k]) for k in range(x12.shape[1])], axis=1)
    s = int(start_frac * (n - t + 1))
    return y[s:s + t].astype(x12.dtype)


def augment_copies(X: np.ndarray, cls: int, rng: np.random.Generator, mag: dict | None = None) -> np.ndarray:
    """One augmented copy of every row of X (N, 1000, 12), all of class `cls`. Uses a fixed number of draws per row,
    so the random stream does not depend on the magnitudes."""
    m = {**MAG, **(mag or {})}
    out = np.empty_like(X)
    t = np.arange(X.shape[1]) / FS
    for i, x in enumerate(X):
        u = rng.random(6)
        phase = rng.uniform(0, 2 * np.pi, 8)
        amp_w = rng.random(8)
        lead_s = rng.uniform(-1, 1, 8)
        eps = rng.standard_normal((8, X.shape[1]))
        f = 1 + m["stretch"] * u[0] if cls in STRETCH_CLASSES else 1.0
        xw = stretch(x, f, u[1])
        x8 = to_independent_8(xw[None])[0].astype(np.float64)
        gain = (1 + m["scale"] * (2 * u[2] - 1)) * (1 + m["lead_scale"] * lead_s)
        freq = 0.05 + 0.45 * u[3]
        d8 = (gain[:, None] - 1) * x8
        d8 += (m["wander"] * amp_w)[:, None] * np.sin(2 * np.pi * freq * t[None] + phase[:, None])
        d8 += m["noise"] * u[4] * eps
        if not d8.any():
            out[i] = xw
            continue
        pair = reconstruct_12(np.stack([x8 + d8, x8]).astype(np.float32))
        out[i] = xw + (pair[0] - pair[1])
    return out


def fill(Xs: np.ndarray, ys: np.ndarray, need: dict, rng: np.random.Generator, augment: bool = True,
         mag: dict | None = None):
    """Rows added per minority class: the same cycled source rows `oversample` duplicates (np.resize), augmented
    when `augment` is True. Returns (X_add, y_add) in class order."""
    xs, yl = [], []
    for c, n in need.items():
        src = np.resize(np.where(ys == c)[0], n)
        xa = Xs[src]
        xs.append(augment_copies(xa, c, rng, mag) if augment else xa)
        yl.append(np.full(n, c, np.int64))
    return np.concatenate(xs), np.concatenate(yl)
