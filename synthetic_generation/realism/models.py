"""Study 4 noise and variation models (Phases 2-3 of report/WORKFLOW_4.md), fitted on the 'fit' part only.

BeatVariation (fix 2, "V1"):
    every full beat k gets its own deviation d_k = U a_k around the ECG's template. Phase 1 found the coefficients
    independent across beats (lag-1 -0.23 ... 0) and mildly coupled to the preceding RR, so
        a_k = mu + beta * rr_rel_k + s * L z_k,   z_k ~ N(0, I) independent per beat,
    mu ~ N(0, between-recording covariance) a per-ECG offset (the mean beat differs slightly from the median template),
    rr_rel = preceding RR / mean RR - 1, L L^T the pooled within-recording covariance of the RR-adjusted coefficients,
    s a per-recording scale drawn from the real recordings' scales (some ECGs vary much more than others).
    Beats clipped by the recording edge keep d_k = 0, as in the decomposition.
Leftover (fix 1, "R0"): the frozen stationary cross-spectral Gaussian process, refitted on the per-beat-fit leftover.
WanderPCA (fix 1, "W1"):
    Phase 1/2 found real wander low-dimensional (40 components = 96 % of its variance) and non-stationary in a fixed
    way: most of its change sits in the first and last second (filter transients at the recording edges), which a
    stationary process cannot produce. So wander is drawn as a Gaussian in the PCA space of real 8-lead wander
    (time-domain covariance, edges included), times a per-recording scale from the real scale distribution."""
from __future__ import annotations

import numpy as np

from synthetic_generation.assembly import CrossSpectralGP
from synthetic_generation.beats import FS, POST, PRE, WIN, place_beats
from synthetic_generation.leads import reconstruct_12


def rr_rel(r: np.ndarray) -> np.ndarray:
    """Preceding RR over the mean RR, minus 1 (0 for the first beat)."""
    if len(r) < 2:
        return np.zeros(len(r))
    rr = np.diff(r).astype(float)
    return np.concatenate([[0.0], rr / rr.mean() - 1])


def full_beats(r: np.ndarray, T: int = 1000) -> np.ndarray:
    return (r - PRE >= 0) & (r + POST <= T)


class BeatVariation:
    def __init__(self, U: np.ndarray, coefs: list, r_list: list):
        """U (K, 8*WIN); coefs[i] (n_beats_i, K) from decompose.fit_coefficients; r_list[i] the R-peaks."""
        self.U = U
        A, R, rec = [], [], []
        for i, (a, r) in enumerate(zip(coefs, r_list)):
            m = full_beats(r)
            if m.sum() < 3:
                continue
            A.append(a[m])
            R.append(rr_rel(r)[m])
            rec.append(np.full(m.sum(), i))
        A, R, rec = np.concatenate(A), np.concatenate(R), np.concatenate(rec)
        self.beta = (R @ A) / max(R @ R, 1e-12)                            # (K,) RR coupling
        E = A - R[:, None] * self.beta
        ids = np.unique(rec)
        mu = np.stack([E[rec == i].mean(0) for i in ids])                  # per-recording offset (mean vs median beat)
        E = E - mu[np.searchsorted(ids, rec)]                              # within-recording part
        cov = E.T @ E / len(E)
        self.L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov)))
        cov_b = np.cov(mu.T) if len(mu) > 2 else np.zeros_like(cov)
        self.L_between = np.linalg.cholesky(cov_b + 1e-9 * np.eye(len(cov)))
        W = np.linalg.solve(self.L, E.T).T                                  # whitened
        sc = np.array([np.sqrt((W[rec == i] ** 2).mean()) for i in np.unique(rec)])
        self.scales = sc / np.sqrt((sc ** 2).mean())                        # mean s^2 = 1
        self.n_beats, self.n_recordings = len(A), len(self.scales)

    def sample(self, r: np.ndarray, rng, T: int = 1000) -> np.ndarray:
        """(n_beats, K) coefficients for R-peaks r; zero for beats clipped by the edges."""
        s = rng.choice(self.scales)
        mu = self.L_between @ rng.standard_normal(len(self.beta))
        a = mu + rr_rel(r)[:, None] * self.beta + s * rng.standard_normal((len(r), len(self.beta))) @ self.L.T
        a[~full_beats(r, T)] = 0
        return a

    def beats(self, tpl: np.ndarray, a: np.ndarray) -> np.ndarray:
        return tpl[None] + (a @ self.U).reshape(len(a), 8, WIN)


def template_features(tpl: np.ndarray) -> np.ndarray:
    """(8, WIN, 3): per lead the template, its first and second time derivative. A beat that is the template
    scaled by (1 + alpha), shifted by tau and widened by omega is, to first order, tpl + alpha tpl - tau tpl' + omega tpl''."""
    t1 = np.gradient(tpl, axis=1)
    return np.stack([tpl, t1, np.gradient(t1, axis=1)], -1)


RIDGE_REL = 0.05


def template_coefficients(dev: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """dev (n_beats, 8, WIN) beat deviations -> (n_beats, 24) per-lead (amplitude, shift, width) coefficients.
    Ridge per feature, RIDGE_REL x that feature's mean energy over the 8 leads: a nearly flat lead otherwise gets
    huge coefficients that explain almost nothing in its own recording but blow up any other template they are
    applied to (Phase 2 audit: SBRAD samples reached |z| ~ 1e6)."""
    F = template_features(tpl)
    lam = RIDGE_REL * (F ** 2).sum(1).mean(0)                               # (3,)
    out = np.zeros((len(dev), 8, 3))
    for l in range(8):
        out[:, l] = np.linalg.solve(F[l].T @ F[l] + np.diag(lam), F[l].T @ dev[:, l, :].T).T
    return out.reshape(len(dev), 24)


def template_deviation(tpl: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(n_beats, 24) coefficients -> (n_beats, 8, WIN) deviations built from this template."""
    return np.einsum("lwf,nlf->nlw", template_features(tpl), p.reshape(len(p), 8, 3))


class TemplateVariation(BeatVariation):
    """Fix 2, "V2": beat-to-beat variation expressed relative to the ECG's OWN template (per lead: amplitude, time
    shift, width; 24 numbers per beat), so a deviation always fits the beat it is applied to. Phase 2 audit: generic
    deviations (V1, or real ones borrowed from another recording) made the R-peak detector miss 12-28 % of beats
    (real: 0 %); within-recording variation is template-specific. Same statistics as BeatVariation (independent per
    beat, RR coupling, per-recording scale and offset), fitted on these coefficients. U is unused."""

    def __init__(self, coefs: list, r_list: list):
        super().__init__(np.zeros((24, 0)), coefs, r_list)

    def beats(self, tpl: np.ndarray, a: np.ndarray) -> np.ndarray:
        return tpl[None] + template_deviation(tpl, a)


def template_decompose(x8: np.ndarray):
    """V2 split of one recording: hp = sum_k place(tpl + template_deviation(tpl, p_k), R_k) + leftover, x8 = hp + wander.
    Coefficients per full beat by least squares on its own window; clipped beats keep p_k = 0. None if no template."""
    from synthetic_generation.beats import TAPER_W, highpass, recording_template
    tpl, r, _ = recording_template(x8)
    if tpl is None:
        return None
    hp = highpass(x8)
    full = full_beats(r, hp.shape[1])
    p = np.zeros((len(r), 24))
    if full.any():
        B = np.stack([hp[:, q - PRE:q + POST] * TAPER_W for q in r[full]])
        p[full] = template_coefficients(B - tpl[None], tpl)
    beats = place_beats(tpl[None] + template_deviation(tpl, p), r, hp.shape[1])
    return {"template": tpl, "r": r, "coef": p, "beats": beats, "leftover": hp - beats, "wander": x8 - hp}


class WanderPCA:
    def __init__(self, wander: np.ndarray, rank: int = 60):
        """wander (n, 8, T) real < 0.5 Hz signal."""
        n = len(wander)
        Xf = wander.reshape(n, -1)
        self.mean = Xf.mean(0)
        _, s, vt = np.linalg.svd(Xf - self.mean, full_matrices=False)
        self.V = vt[:rank]
        self.sd = s[:rank] / np.sqrt(n)
        c = (Xf - self.mean) @ self.V.T / self.sd                           # whitened coefficients
        sc = np.sqrt((c ** 2).mean(1))
        self.scales = sc / np.sqrt((sc ** 2).mean())
        self.shape = wander.shape[1:]
        self.explained = float((s[:rank] ** 2).sum() / (s ** 2).sum())
        self.train_flat = Xf                                                # for the memorisation check only

    def sample(self, rng) -> np.ndarray:
        g = rng.choice(self.scales)
        c = g * rng.standard_normal(len(self.sd)) * self.sd
        return (self.mean + c @ self.V).reshape(self.shape)


def assemble_v2(template, rhythm, variation: BeatVariation, leftover: CrossSpectralGP, wander: WanderPCA, rng,
                T: int = 1000):
    """One 12-lead ECG (T, 12): sampled R-peaks, per-beat varied templates, leftover and wander.
    No separate respiratory amplitude modulation: real per-beat deviations already contain it."""
    r = rhythm.sample_r_positions(rng)
    a = variation.sample(r, rng, T)
    x8 = place_beats(variation.beats(template, a), r, T) + leftover.sample(rng) + wander.sample(rng)
    return reconstruct_12(x8[None].astype(np.float32))[0]


def nearest_corr(samples: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Max Pearson correlation of each sample (flattened) to any training row: > 0.98 counts as a copy."""
    a = samples.reshape(len(samples), -1)
    b = train.reshape(len(train), -1)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-12)
    return (a @ b.T / a.shape[1]).max(1)
