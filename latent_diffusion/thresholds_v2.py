"""Per-class log-prob offset tuning, copied unchanged from _threshold_explore.py (tune, conf, f1_from_conf, predict).

Copied rather than imported so that no module touching study 3/5 paths is imported by Study 9.
"""
import numpy as np

K = 4
GRID = np.round(np.arange(-4.0, 4.0 + 1e-9, 0.05), 2)
SWEEPS = 4


def f1_from_conf(cm):
    """Macro-F1 from (..., 4, 4) confusion counts [true, pred]; sklearn semantics (0/0 -> 0)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(-2) - tp
    fn = cm.sum(-1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean(-1)


def conf(y, pred):
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(np.float64)


def predict(logp, b):
    return np.argmax(logp + b, axis=1)


def tune(logp, y):
    """Coordinate search over GRID for b_AFIB, b_STACH, b_SBRAD (b_SR = 0), maximising macro-F1."""
    b = np.zeros(K)
    n = len(y)
    for _ in range(SWEEPS):
        moved = False
        for c in (1, 2, 3):
            s = logp + b
            oth = s.copy()
            oth[:, c] = -np.inf
            k = oth.argmax(1)
            d = oth[np.arange(n), k] - logp[:, c]
            order = np.argsort(d, kind="stable")
            pair = (y * K + k)[order]
            onehot = np.zeros((n + 1, K * K))
            onehot[np.arange(1, n + 1), pair] = 1
            cum = onehot.cumsum(0)
            moved_cnt = cum[np.searchsorted(d[order], GRID, side="left")].reshape(len(GRID), K, K)
            cm = conf(y, k)[None] - moved_cnt
            cm[:, :, c] += moved_cnt.sum(2)
            sc = f1_from_conf(cm)
            best = np.flatnonzero(sc >= sc.max() - 1e-12)
            g = GRID[best[np.argmin(np.abs(GRID[best] - b[c]))]]
            if g != b[c]:
                b[c], moved = g, True
        if not moved:
            break
    return b, float(sc.max())


def logp_of(prob):
    return np.log(np.clip(prob.astype(np.float64), np.finfo(np.float32).tiny, None))
