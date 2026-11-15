"""R-peak caches and R-maps (Phase D). Train split (fit + dev) only; validation peaks are computed only in Phase F."""
import numpy as np
from .v2_common import ART, write_json, RESULTS

MAXP = 40


def detect_all(x):
    from synthetic_generation.beats import detect_r_peaks
    from synthetic_generation.leads import to_independent_8
    peaks = np.full((len(x), MAXP), -1, np.int32)
    for start in range(0, len(x), 256):
        block = to_independent_8(np.asarray(x[start:start+256], np.float32))
        for j, sig in enumerate(block):
            p = np.asarray(detect_r_peaks(sig))[:MAXP]
            peaks[start+j, :len(p)] = p
    return peaks


def plausible(peaks):
    """Valid detection: >=3 peaks, mean HR 30-200 bpm, no RR gap > 2 s."""
    ok = np.zeros(len(peaks), bool)
    for i, row in enumerate(peaks):
        p = row[row >= 0]
        if len(p) >= 3:
            rr = np.diff(p)
            hr = 6000./rr.mean()
            ok[i] = 30 <= hr <= 200 and rr.max() <= 200
    return ok


def train_peaks():
    path = ART/'rpeaks_train.npz'
    if path.exists():
        d = np.load(path)
        return d['peaks'], d['valid']
    from .v2_common import training_data
    x, *_ = training_data()
    peaks = detect_all(x)
    valid = plausible(peaks)
    np.savez_compressed(path, peaks=peaks, valid=valid)
    return peaks, valid


def rmap(peaks_rows, valid, length, sigma=2.):
    """(N, 2, length): Gaussian bumps (sigma samples at 100 Hz) avg-pooled to `length`, plus the valid flag."""
    t = np.arange(1000, dtype=np.float32)
    out = np.zeros((len(peaks_rows), 2, length), np.float32)
    f = 1000 // length
    for i, (row, ok) in enumerate(zip(peaks_rows, valid)):
        if not ok:
            continue
        p = row[row >= 0].astype(np.float32)
        m = np.exp(-.5*((t[None] - p[:, None])/sigma)**2).max(0) if len(p) else np.zeros(1000, np.float32)
        out[i, 0] = m.reshape(length, f).mean(1)
        out[i, 1] = 1.
    return out


def regular_peaks(hr, rng, cv=0., irregular=False, n=None):
    """R positions for a requested HR: regular (sinus sweep) or gamma RRs (AFIB-like)."""
    mean = 60./hr
    rr = []
    total = rng.uniform(0, mean)
    while total < 10.2:
        if irregular:
            k = 1/max(cv, .05)**2
            rr.append(rng.gamma(k, mean/k))
        else:
            rr.append(mean*(1 + cv*rng.standard_normal()))
        total += rr[-1]
    pos = np.round((rng.uniform(0, mean) + np.concatenate([[0], np.cumsum(rr)]))*100).astype(int)
    pos = pos[(pos >= 0) & (pos < 1000)][:MAXP]
    row = np.full(MAXP, -1, np.int32)
    row[:len(pos)] = pos
    return row


def rr_bank_peaks(peaks, valid, rows, rng):
    """Real RR sequence of a random fit row (same class chosen by caller), shifted by a random offset."""
    out = np.full((len(rows), MAXP), -1, np.int32)
    for i, r in enumerate(rows):
        p = peaks[r][peaks[r] >= 0]
        rr = np.diff(p)
        start = rng.integers(0, max(1, rr[0] if len(rr) else 50))
        pos = start + np.concatenate([[0], np.cumsum(rr)])
        pos = pos[pos < 1000][:MAXP]
        out[i, :len(pos)] = pos
    return out
