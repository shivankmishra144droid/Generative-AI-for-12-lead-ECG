"""Detector protocol v2 (study 4, Phase 0): can a small CNN tell synthetic ECGs from real ones?

Changes from generator_comparison/realism_detector.py (whose `Det` network is reused unchanged):
- the real side is an explicit list of X_train rows (study 4: the patient-level 'dev' part), so recordings used to
  build or fit anything never appear on it;
- 5 seeds by default, with a t-interval over seeds, overall and per class;
- CPU by default with a thread cap, so it can run while the GPU is busy;
- `band` retrains the detector on one part of the signal only, to show where the giveaway is:
    full     the 12-lead ECG as is
    low      < 0.5 Hz (baseline wander)
    mid      0.5-40 Hz
    beats    only R - 250 ms ... R + 450 ms around each detected R-peak (zero elsewhere)
    between  only the samples outside those windows
  The masks are computed per ECG with the frozen R-peak detector, on real and synthetic ECGs alike.

AUC ~0.5 = indistinguishable; 1.0 = perfectly separable."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from scipy.stats import t as t_dist
from sklearn.metrics import roc_auc_score

from generator_comparison.eval_metrics import CLASS_ORDER
from generator_comparison.realism_detector import Det
from synthetic_generation.beats import FS, POST, PRE, detect_r_peaks
from synthetic_generation.leads import to_independent_8

BANDS = ("full", "low", "mid", "beats", "between")
SEEDS = (0, 1, 2, 3, 4)
_LOW = butter(2, 0.5 / (FS / 2), btype="low")
_MID = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")


def beat_mask(x12: np.ndarray) -> np.ndarray:
    """(N, 1000) bool: True inside R - PRE ... R + POST of any detected beat."""
    x8 = to_independent_8(x12)
    m = np.zeros(x12.shape[:2], bool)
    for i in range(len(x12)):
        for r in detect_r_peaks(x8[i].astype(np.float64)):
            m[i, max(0, r - PRE):min(x12.shape[1], r + POST)] = True
    return m


def apply_band(x12: np.ndarray, band: str) -> np.ndarray:
    """(N, 1000, 12) -> the same shape, keeping only one part of the signal."""
    x = np.asarray(x12, np.float64)
    if band == "full":
        out = x
    elif band == "low":
        out = filtfilt(*_LOW, x, axis=1)
    elif band == "mid":
        out = filtfilt(*_MID, x, axis=1)
    elif band in ("beats", "between"):
        m = beat_mask(x12)
        out = x * (m if band == "beats" else ~m)[:, :, None]
    else:
        raise ValueError(band)
    return np.ascontiguousarray(out, dtype=np.float32)


def group_split(groups, rng, train_share=0.7):
    """Train/test row indices with every group (patient) on one side only."""
    ug = rng.permutation(np.unique(groups))
    sizes = np.array([np.sum(groups == g) for g in ug])
    tr_groups = set(ug[np.cumsum(sizes) <= train_share * len(groups)].tolist())
    in_tr = np.array([g in tr_groups for g in groups])
    return rng.permutation(np.where(in_tr)[0]), np.where(~in_tr)[0]


def run(x_real, y_real, x_syn, y_syn, seed=0, epochs=15, device="cpu", g_real=None, g_syn=None):
    """One detector. Equal real/synthetic counts per class (n_c = the smaller side), ~70/30 train/test split by group.

    g_real / g_syn: patient ids of the real ECGs / of the source recording of each synthetic ECG (None: every ECG is its
    own group). Grouping matters: without it a patient with several recordings sits on both sides of the split and the
    detector learns the patient, not the realism (real-vs-real AFIB reached AUC ~0.58 that way, Phase 0 audit).
    Real and synthetic groups are kept apart even if a synthetic ECG was built from a real patient's recording.
    Returns overall and per-class test AUC."""
    rng = np.random.default_rng(seed)
    g_real = np.arange(len(y_real)) if g_real is None else np.asarray(g_real)
    g_syn = np.arange(len(y_syn)) if g_syn is None else np.asarray(g_syn)
    n_c = [min(int((y_real == c).sum()), int((y_syn == c).sum())) for c in range(4)]
    ri = np.concatenate([rng.choice(np.where(y_real == c)[0], n_c[c], replace=False) for c in range(4)])
    si = np.concatenate([rng.choice(np.where(y_syn == c)[0], n_c[c], replace=False) for c in range(4)])
    xa = np.concatenate([x_real[ri], x_syn[si]]).astype(np.float32)
    ca = np.concatenate([y_real[ri], y_syn[si]])
    lab = np.concatenate([np.zeros(len(ri)), np.ones(len(si))]).astype(np.float32)
    groups = np.concatenate([[f"r{g}" for g in g_real[ri]], [f"s{g}" for g in g_syn[si]]])
    tr, te = group_split(groups, rng)
    dev = torch.device(device)
    torch.manual_seed(seed)
    net = Det().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt, Lt = torch.from_numpy(xa), torch.from_numpy(lab)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 64)):
            opt.zero_grad()
            lossf(net(Xt[b].to(dev)), Lt[b].to(dev)).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(net(Xt[b].to(dev))).cpu()
                       for b in np.array_split(te, max(1, len(te) // 256))]).numpy()
    out = {"auc": float(roc_auc_score(lab[te], p)), "n_per_class": n_c, "n_test": int(len(te)), "per_class": {}}
    for c in range(4):
        m = ca[te] == c
        if len(np.unique(lab[te][m])) == 2:
            out["per_class"][CLASS_ORDER[c]] = float(roc_auc_score(lab[te][m], p[m]))
    return out


def summarise(values):
    v = np.asarray(values, float)
    if len(v) < 2:
        return {"mean": float(v.mean()), "ci95": [float(v.mean())] * 2}
    h = t_dist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci95": [float(v.mean() - h), float(v.mean() + h)]}


def evaluate(x_real, y_real, x_syn, y_syn, band="full", seeds=SEEDS, epochs=15, threads=4, g_real=None, g_syn=None):
    """The protocol: one detector per seed on the given band. x_real must already exclude every source recording;
    g_real / g_syn are patient ids (see run)."""
    torch.set_num_threads(threads)
    xr, xs = apply_band(x_real, band), apply_band(x_syn, band)
    runs = {f"seed{s}": run(xr, y_real, xs, y_syn, seed=s, epochs=epochs, g_real=g_real, g_syn=g_syn)
            for s in seeds}
    out = {"band": band, "runs": runs, "auc": summarise([r["auc"] for r in runs.values()]), "per_class": {}}
    for name in CLASS_ORDER:
        vals = [r["per_class"][name] for r in runs.values() if name in r["per_class"]]
        if vals:
            out["per_class"][name] = summarise(vals)
    return out
