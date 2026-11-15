"""Interpretable ECG features for the feature-level realism judge (study 4, generator_comparison/study4/PLAN_features.md).

All features are computed on the 8 independent leads (I, II, V1-V6; z-scored units) with the frozen R-peak detector,
median template and 0.5 Hz high-pass, the same way for real and synthetic ECGs. Lead II is index 1. Feature
definitions were fixed in the plan before any data was looked at."""
from __future__ import annotations

import numpy as np

from synthetic_generation.beats import FS, POST, PRE, TAPER_W, detect_r_peaks, highpass, place_beats, recording_template
from synthetic_generation.leads import to_independent_8
from synthetic_generation.realism.leftover import r_offsets

II = 1
NAMES = ("hr", "sdnn_ms", "rmssd_ms", "rr_cv", "n_beats",
         "qrs_width_ms", "r_amp_II", "s_depth_II", "t_amp_II", "st_level_II", "tpl_p2p",
         "beat_corr_median", "r_amp_rel_sd",
         "res_rms", "res_kurtosis", "res_edge_ratio", "res_qrs_ratio", "res_hf_share",
         "wander_rms", "wander_inc_kurtosis_mid", "wander_edge_inc_ratio",
         "cross_lead_corr", "spectral_entropy_II", "total_rms", "max_abs")


def kurtosis(a: np.ndarray, axis=-1) -> np.ndarray:
    z = a - a.mean(axis, keepdims=True)
    return (z ** 4).mean(axis) / ((z ** 2).mean(axis) ** 2 + 1e-12)


def edge_ratio(p: np.ndarray, edge: int = 100) -> float:
    """p: power per time sample. Mean over the first and last `edge` samples / mean of the middle."""
    return float(np.r_[p[:edge], p[-edge:]].mean() / (p[edge:-edge].mean() + 1e-12))


def qrs_width_ms(tpl: np.ndarray, frac: float = 0.2) -> float:
    """Contiguous span around R (index PRE) where the 8-lead derivative energy of the template is >= frac x its peak
    within R +/- 15 samples."""
    e = (np.gradient(tpl, axis=1) ** 2).sum(0)
    lo, hi = PRE - 15, PRE + 16
    thr = frac * e[lo:hi].max()
    k = lo + int(np.argmax(e[lo:hi]))
    a, b = k, k
    while a - 1 >= 0 and e[a - 1] >= thr:
        a -= 1
    while b + 1 < len(e) and e[b + 1] >= thr:
        b += 1
    return float((b - a + 1) * 1000 / FS)


def spectral_entropy(x: np.ndarray, lo: float = 0.5, hi: float = 40.0) -> float:
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1 / FS)
    p = p[(f >= lo) & (f <= hi)]
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def features_x8(x8: np.ndarray) -> np.ndarray:
    """(8, T) float -> feature vector in NAMES order (NaN where undefined)."""
    x8 = x8.astype(np.float64)
    T = x8.shape[1]
    f = dict.fromkeys(NAMES, np.nan)
    r = detect_r_peaks(x8)
    f["n_beats"] = float(len(r))
    if len(r) >= 3:
        rr = np.diff(r) / FS
        f["hr"] = 60 / rr.mean()
        f["sdnn_ms"] = rr.std() * 1000
        f["rmssd_ms"] = np.sqrt((np.diff(rr) ** 2).mean()) * 1000 if len(rr) > 1 else np.nan
        f["rr_cv"] = rr.std() / rr.mean()
    hp = highpass(x8)
    w = x8 - hp
    tpl, r2, _ = recording_template(x8)
    if tpl is not None:
        t = tpl[II]
        f["qrs_width_ms"] = qrs_width_ms(tpl)
        f["r_amp_II"] = t[PRE]
        f["s_depth_II"] = t[PRE:PRE + 8].min()
        tw = t[PRE + 15:PRE + 40]
        f["t_amp_II"] = tw[np.argmax(np.abs(tw))]
        f["st_level_II"] = t[PRE + 8:PRE + 12].mean()
        f["tpl_p2p"] = np.ptp(tpl, axis=1).mean()
        full = r2[(r2 - PRE >= 0) & (r2 + POST <= T)]
        if len(full) >= 2:
            B = np.stack([hp[:, q - PRE:q + POST] * TAPER_W for q in full])
            tf = tpl.ravel() - tpl.mean()
            cs = [np.corrcoef(b.ravel(), tf)[0, 1] for b in B]
            f["beat_corr_median"] = float(np.median(cs))
            ra = hp[II, full]
            f["r_amp_rel_sd"] = ra.std() / (np.abs(ra.mean()) + 1e-6)
        res = hp - place_beats(tpl, r2, T)
        p = (res ** 2).mean(0)
        f["res_rms"] = np.sqrt(p.mean())
        f["res_kurtosis"] = float(np.median(kurtosis(res)))
        f["res_edge_ratio"] = edge_ratio(p)
        o = r_offsets(r2, T)[100:-100]
        q = np.abs(o) <= 8
        pm = p[100:-100]
        if q.any() and (~q).any():
            f["res_qrs_ratio"] = pm[q].mean() / (pm[~q].mean() + 1e-12)
        P = (np.abs(np.fft.rfft(res, axis=1)) ** 2).sum(0)
        fr = np.fft.rfftfreq(T, 1 / FS)
        f["res_hf_share"] = P[fr > 15].sum() / (P.sum() + 1e-12)
    f["wander_rms"] = np.sqrt((w ** 2).mean())
    dw = np.diff(w, axis=1)
    f["wander_inc_kurtosis_mid"] = float(np.median(kurtosis(dw[:, 100:-100])))
    f["wander_edge_inc_ratio"] = edge_ratio((dw ** 2).mean(0))
    c = np.corrcoef(x8)
    f["cross_lead_corr"] = float(np.nanmean(c[np.triu_indices(8, 1)]))
    f["spectral_entropy_II"] = spectral_entropy(x8[II])
    f["total_rms"] = np.sqrt((x8 ** 2).mean())
    f["max_abs"] = np.abs(x8).max()
    return np.array([f[k] for k in NAMES], dtype=np.float64)


def features_12(X12: np.ndarray) -> np.ndarray:
    """(n, T, 12) -> (n, len(NAMES))."""
    x8 = to_independent_8(np.asarray(X12, np.float32))
    return np.stack([features_x8(x) for x in x8])
