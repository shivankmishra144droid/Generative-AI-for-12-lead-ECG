"""Study 4, feature question F (follow-up to the feature judge): does the noise model break R-peak detection?

The feature judge found synthetic ECGs 'irregular' (SDNN ~120-150 ms vs real 36 ms) even for C, which keeps each
source's own real R-peak timing. Every set here keeps its source's own R-peaks and template, so the detector's output
on the real source is the truth. A beat is matched within 5 samples (50 ms).

Detectors:
  project  the frozen synthetic_generation.beats.detect_r_peaks (8-lead, Lead-II envelope + derivative refinement)
  xqrs     wfdb.processing.xqrs_detect on Lead II in mV (an independent, widely used detector; E, C, D only)
Sets:
  E, C, D               baselines (real rebuild / GP residual / GP wander)
  R2, P_rand, R1        own V2 beats + own wander + leftover = stationary GP / own real leftover phase-randomised /
                        envelope GP. P_rand has the real leftover's exact spectrum and scale: if it corrupts
                        detection, the cause is Gaussian-like structure; if not, excess amplitude.
  VS                    own real V2 coefficients shuffled across beats + own leftover (variation only)
  LR                    learned residual (residual/variants.npz), when it exists
Writes generator_comparison/study4/features/rpeak_check.json. Usage: py -3.14 _study4_rpeak_check.py"""
from __future__ import annotations

import json

import numpy as np

import _study4_common as C
from synthetic_generation.beats import FS, detect_r_peaks
from synthetic_generation.leads import to_independent_8

_NORM = json.loads((C.DATA / "normalization_stats.json").read_text())
_II = _NORM["lead_order"].index("II")


def peaks_project(x12):
    return detect_r_peaks(to_independent_8(np.asarray(x12)[None])[0].astype(np.float64))


def peaks_xqrs(x12):
    from wfdb import processing
    sig = np.asarray(x12)[:, _II].astype(np.float64) * _NORM["std"][_II] + _NORM["mean"][_II]
    return np.asarray(processing.xqrs_detect(sig=sig, fs=FS, verbose=False), dtype=np.int64)


def compare(X_syn, src, X, detector):
    miss = extra = tot = 0
    any_err, sd_true, sd_syn = [], [], []
    for x, i in zip(X_syn, src):
        rt, rs = detector(X[i]), detector(x)
        m = sum(not len(rs) or np.min(np.abs(rs - q)) > 5 for q in rt)
        e = sum(not len(rt) or np.min(np.abs(rt - q)) > 5 for q in rs)
        miss, extra, tot = miss + m, extra + e, tot + len(rt)
        any_err.append(m + e > 0)
        if len(rt) > 2 and len(rs) > 2:
            sd_true.append(np.diff(rt).std() / FS * 1000)
            sd_syn.append(np.diff(rs).std() / FS * 1000)
    return {"n_ecg": int(len(src)), "true_beats": int(tot), "missed_frac": miss / max(tot, 1),
            "spurious_frac": extra / max(tot, 1), "ecg_with_any_error_frac": float(np.mean(any_err)),
            "sdnn_ms_median_true_vs_measured": [float(np.median(sd_true)), float(np.median(sd_syn))]}


def main():
    C.check_frozen()
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    sets = {}
    for k in ("E", "C", "D"):
        d = np.load(C.S4 / "baselines" / f"{k}.npz")
        sets[k] = (d["X"], d["src_idx"])
    r2 = np.load(C.S4 / "fixes" / "R2.npz")
    sets["R2"] = (r2["X"], r2["src_idx"])
    lo = np.load(C.S4 / "leftover" / "variants.npz")
    for k in ("P_rand", "R1"):
        sets[k] = (lo[k], lo["src_idx"])
    va = np.load(C.S4 / "variation" / "variants.npz")
    sets["VS"] = (va["VS"], va["src_idx"])
    if (C.S4 / "residual" / "variants.npz").exists():
        lr = np.load(C.S4 / "residual" / "variants.npz")
        sets["LR"] = (lr["LR"], lr["src_idx"])
    p = C.S4 / "features" / "rpeak_check.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    out = {k: v for k, v in out.items() if isinstance(v, dict) and "project" in v}     # old flat format is redone
    for k, (Xs, src) in sets.items():
        if k in out:
            continue
        out[k] = {"project": compare(Xs, src, X, peaks_project)}
        if k in ("E", "C", "D"):
            out[k]["xqrs"] = compare(Xs, src, X, peaks_xqrs)
        print(k, json.dumps(out[k]), flush=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    C.check_frozen()


if __name__ == "__main__":
    main()
