"""Study 4 control (post hoc, prompted by a code review): is part of the detector's signal the 1 uV quantisation grid?

Real PTB-XL samples lie exactly on a 1 uV grid (100 % of samples, every lead; checked on 200 train ECGs), while
synthetic sums are continuous (off-grid: gen pool ~90 %, C ~90 %, V2-type rebuilds 68-80 %). If the CNN detector
senses the grid, part of every AUC (e.g. the 0.77 'pairing floor') is a storage artefact, not unrealistic signal.

Control: put each synthetic ECG on the grid (z -> mV with the train normalisation stats, round to 1 uV, back to z)
and rerun detector protocol v2 (same real side, groups and seeds), full band:
   VS_g     variation/variants.npz VS (0.779 unrounded)    the rebuild floor
   R2_g     fixes/R2.npz            (0.999 unrounded)      GP leftover
   gen_g    tstr/pool_beatgen.npz   (1.000 unrounded)      the current generator
Writes generator_comparison/study4/grid/. Usage: py -3.14 _study4_grid.py"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

import _study4_common as C
from synthetic_generation.realism.detect import evaluate

OUT = C.S4 / "grid"


def to_grid(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(n, T, 12) z-scored -> the same on the 1 uV grid."""
    mv = x.astype(np.float64) * sd + mu
    return ((np.round(mv * 1000) / 1000 - mu) / sd).astype(np.float32)


def offgrid_frac(x, mu, sd):
    mv = x.astype(np.float64) * sd + mu
    return float((np.abs(mv * 1000 - np.round(mv * 1000)) > 0.05).mean())


def main():
    C.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((C.DATA / "normalization_stats.json").read_text())
    mu, sd = np.array(s["mean"]), np.array(s["std"])
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    dev = C.part_index("dev")
    rng2 = np.random.default_rng(42)                                     # same real side as _study4_fixes.py
    sr = dev[y[dev] == 0]
    real_rows = np.sort(np.concatenate([rng2.choice(sr, 600, replace=False), dev[y[dev] != 0]]))
    xr, yr = np.asarray(X[real_rows], np.float32), y[real_rows]
    v = np.load(C.S4 / "variation" / "variants.npz")
    r2 = np.load(C.S4 / "fixes" / "R2.npz")
    g = np.load(C.ROOT / "generator_comparison" / "tstr" / "pool_beatgen.npz")
    sets = {"VS_g": (v["VS"], v["y"], pid[v["src_idx"]]), "R2_g": (r2["X"], r2["y"], pid[r2["src_idx"]]),
            "gen_g": (g["X"], g["y"], None)}
    summary = json.loads((OUT / "detector.json").read_text()) if (OUT / "detector.json").exists() else {}
    summary["offgrid_real"] = offgrid_frac(xr[:300], mu, sd)
    for k, (xs, ys, gs) in sets.items():
        if k in summary:
            continue
        xg = to_grid(xs, mu, sd)
        rms_change = float(np.sqrt(((xg - xs) ** 2).mean()))
        t0 = time.time()
        res = evaluate(xr, yr, xg, ys, band="full", threads=4, g_real=pid[real_rows], g_syn=gs)
        res["real_rows_sha256"] = hashlib.sha256(real_rows.tobytes()).hexdigest()
        res["post_hoc"] = True
        (OUT / f"detector_{k}_full.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        summary[k] = {"auc": res["auc"], "per_class": {c: round(q["mean"], 3) for c, q in res["per_class"].items()},
                      "offgrid_before": offgrid_frac(xs[:300], mu, sd), "offgrid_after": offgrid_frac(xg[:300], mu, sd),
                      "rounding_rms_z": rms_change}
        (OUT / "detector.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{k:6s} AUC {res['auc']['mean']:.3f} [{res['auc']['ci95'][0]:.3f}, {res['auc']['ci95'][1]:.3f}] "
              f"{summary[k]['per_class']} offgrid {summary[k]['offgrid_before']:.2f}->{summary[k]['offgrid_after']:.2f} "
              f"{time.time() - t0:.0f}s", flush=True)
    C.check_frozen()


if __name__ == "__main__":
    main()
