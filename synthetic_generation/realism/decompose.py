"""Per-beat decomposition of a real ECG (study 4, report/WORKFLOW_4.md, "The decomposition").

On the 8 independent leads, after the frozen 0.5 Hz high-pass:

    hp  =  sum_k place(template + d_k, R_k)  +  r          and   x8 = hp + w   (w = wander, < 0.5 Hz)

- template: the recording's median beat (frozen beats.recording_template), tapered;
- d_k = U a_k: beat k's deviation in a low-rank basis U (per class, PCA of real tapered beat deviations, fit part);
- the coefficients a_k of all beats are fitted jointly by ridge least squares, so overlapping windows (STACH) and beats
  clipped at the edges are handled;
- r: what is left.
d_k = 0 gives back the current generator's residual (r = hp - place(template)), so nothing is counted twice."""
from __future__ import annotations

import numpy as np

from synthetic_generation.beats import POST, PRE, TAPER_W, WIN, highpass, place_beats, recording_template

D = 8 * WIN                      # one beat window, flattened


def beat_deviations(hp: np.ndarray, tpl: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(n_full_beats, 8 * WIN): tapered beat windows minus the template, full windows only."""
    full = r[(r - PRE >= 0) & (r + POST <= hp.shape[1])]
    if not len(full):
        return np.zeros((0, D))
    w = np.stack([hp[:, p - PRE:p + POST] * TAPER_W for p in full])
    return (w - tpl[None]).reshape(len(full), D)


def fit_basis(deviations: np.ndarray, rank: int) -> dict:
    """PCA of pooled beat deviations. Returns the basis U (rank, D) and the explained-variance curve."""
    dev = deviations - deviations.mean(0)
    _, s, vt = np.linalg.svd(dev, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    return {"U": vt[:rank], "explained": np.cumsum(ev)[:max(rank, 64)], "mean": deviations.mean(0)}


def fit_coefficients(hp: np.ndarray, tpl: np.ndarray, r: np.ndarray, U: np.ndarray, ridge: float = 0.1) -> np.ndarray:
    """(n_beats, rank) coefficients minimising ||hp - place(tpl) - sum_k place(U a_k, R_k)||^2 + ridge ||a||^2.
    Beats whose window is clipped by the recording edge keep a_k = 0: with part of the window unobserved their
    coefficients are ill-posed (Phase 1 audit: |a| up to ~120 and a spurious leftover bump at the edges)."""
    target = (hp - place_beats(tpl, r, hp.shape[1])).ravel()
    K = len(U)
    full = np.where((r - PRE >= 0) & (r + POST <= hp.shape[1]))[0]
    a = np.zeros((len(r), K))
    if not len(full):
        return a
    cols = []
    for p in r[full]:
        for j in range(K):
            cols.append(place_beats(U[j].reshape(8, WIN), [p], hp.shape[1]).ravel())
    A = np.stack(cols, 1)                                                  # (8 * T, n_full_beats * K)
    a[full] = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ target).reshape(len(full), K)
    return a


def beats_from_coefficients(tpl: np.ndarray, a: np.ndarray, U: np.ndarray) -> np.ndarray:
    """(n_beats, 8, WIN) per-beat templates tpl + U a_k."""
    return tpl[None] + (a @ U).reshape(len(a), 8, WIN)


def decompose(x8: np.ndarray, U: np.ndarray | None):
    """One recording -> dict(template, r_peaks, coef, beats_signal, leftover, wander), or None if no template.
    U None: d_k = 0 (the current generator's split)."""
    tpl, r, _ = recording_template(x8)
    if tpl is None:
        return None
    hp = highpass(x8)
    if U is None or not len(U):
        a = np.zeros((len(r), 0))
        beats = place_beats(tpl, r, hp.shape[1])
    else:
        a = fit_coefficients(hp, tpl, r, U)
        beats = place_beats(beats_from_coefficients(tpl, a, U), r, hp.shape[1])
    return {"template": tpl, "r": r, "coef": a, "beats": beats, "leftover": hp - beats, "wander": x8 - hp}
