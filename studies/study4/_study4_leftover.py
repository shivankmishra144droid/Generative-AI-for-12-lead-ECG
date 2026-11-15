"""Study 4, Phase 2 escalation: leftover models (CPU, train split only; report/WORKFLOW_4.md D5).

Same 1,600-ish fit-part source recordings as _study4_fixes.py's R2 (own V2 beats + own wander), leftover replaced by:

   P_rand  own leftover, phase-randomised (same phases on all leads)   diagnostic: is the giveaway time structure?
   R1e     GP leftover x edge envelope g(t)
   R1b     GP leftover x beat-phase envelope h(offset from R)
   R1      GP leftover x both envelopes (the cyclostationary model of D5, plus edges)

1. Envelopes fitted on N_ENV fit-part recordings per class (their own seed).
2. Calibration on the leftover itself, before any detector: RMS quantiles, kurtosis (all / without edge seconds),
   edge-to-middle power, QRS-to-rest power; real leftover of the same sources as reference.
3. Detector protocol v2 (as _study4_fixes.py: dev real side, patient-grouped, 5 seeds), full band.
Writes generator_comparison/study4/leftover/. Usage: py -3.14 _study4_leftover.py"""
from __future__ import annotations

import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd

import _study4_common as C
from _study4_fixes import load_rec
from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.leads import reconstruct_12
from synthetic_generation.realism import leftover as LO
from synthetic_generation.realism import models as M
from synthetic_generation.realism.detect import evaluate

OUT = C.S4 / "leftover"
FIX = C.S4 / "fixes"
N_ENV, SEED = 400, 71
VARIANTS = ("P_rand", "R1e", "R1b", "R1")
RUNS = {"P_rand": ["full"], "R1e": ["full"], "R1b": ["full"], "R1": ["full", "between"]}


def leftover_stats(L: np.ndarray, r_list: list) -> dict:
    """L (n, 8, T) leftovers with their R-peaks."""
    z = L - L.mean(-1, keepdims=True)

    def kurt(a):
        return np.median((a ** 4).mean(-1) / ((a ** 2).mean(-1) ** 2 + 1e-12), -1)
    p = (L ** 2).mean(1)
    qrs, rest = [], []
    for pi, r in zip(p, r_list):
        o = LO.r_offsets(r, L.shape[2])[100:-100]
        q = np.abs(o) <= 8
        qrs.append(pi[100:-100][q].sum())
        rest.append(pi[100:-100][~q].sum())
        qrs[-1] /= max(q.sum(), 1)
        rest[-1] /= max((~q).sum(), 1)
    rms = np.sqrt((L ** 2).mean((1, 2)))
    e = p.mean(0)
    return {"rms_q10_50_90": np.quantile(rms, [.1, .5, .9]).round(3).tolist(),
            "kurtosis_median": float(np.median(kurt(z))), "kurtosis_mid_median": float(np.median(kurt(z[:, :, 100:-100]))),
            "edge_to_middle_power": float(np.r_[e[:100], e[-100:]].mean() / e[100:-100].mean()),
            "qrs_to_rest_power": float(np.mean(qrs) / np.mean(rest))}


def main():
    C.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    fit, dev = C.part_index("fit"), C.part_index("dev")
    with open(FIX / "models.pkl", "rb") as f:
        models = pickle.load(f)["models"]
    src = np.load(FIX / "R2.npz")["src_idx"]
    rng = np.random.default_rng(SEED)

    if not (OUT / "variants.npz").exists():
        env = {}
        for c in range(4):
            Ls, rs = [], []
            for i in rng.choice(fit[y[fit] == c], min(N_ENV, (y[fit] == c).sum()), replace=False):
                x8, tpl, r = load_rec(X, i)
                if tpl is None:
                    continue
                d2 = M.template_decompose(x8)
                Ls.append(d2["leftover"])
                rs.append(d2["r"])
            gp = models[c]["leftover2"]
            env[c] = {k: LO.EnvelopeLeftover(gp, Ls, rs, edge=e, beat=b)
                      for k, e, b in (("R1e", True, False), ("R1b", False, True), ("R1", True, True))}
            print(CLASS_ORDER[c], "envelopes from", len(Ls), "recordings; edge g max", env[c]["R1"].g.max().round(2),
                  "beat h max", env[c]["R1"].h.max().round(2), flush=True)
        out = {k: [] for k in ("real",) + VARIANTS}
        rr = []
        for i in src:
            x8, tpl, r = load_rec(X, i)
            d2 = M.template_decompose(x8)
            c = int(y[i])
            leftovers = {"real": d2["leftover"], "P_rand": LO.phase_randomise(d2["leftover"], rng)}
            for k in ("R1e", "R1b", "R1"):
                leftovers[k] = env[c][k].sample(rng, d2["r"])
            for k, lft in leftovers.items():
                out[k].append(lft)
            rr.append(d2["r"])
            for k in VARIANTS:
                out.setdefault("X_" + k, []).append(
                    reconstruct_12((d2["beats"] + leftovers[k] + d2["wander"])[None].astype(np.float32))[0])
        cal = {}
        for c in range(4):
            m = y[src] == c
            cal[CLASS_ORDER[c]] = {k: leftover_stats(np.stack(out[k])[m], [rr[j] for j in np.where(m)[0]])
                                   for k in ("real",) + VARIANTS}
        (OUT / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
        print(json.dumps(cal, indent=1), flush=True)
        np.savez_compressed(OUT / "variants.npz", src_idx=src, y=y[src],
                            **{k: np.stack(out["X_" + k]) for k in VARIANTS})

    d = np.load(OUT / "variants.npz")
    rng2 = np.random.default_rng(42)                                     # same real side as _study4_fixes.py
    sr = dev[y[dev] == 0]
    real_rows = np.sort(np.concatenate([rng2.choice(sr, 600, replace=False), dev[y[dev] != 0]]))
    assert not set(src.tolist()) & set(real_rows.tolist())
    xr, yr = np.asarray(X[real_rows], np.float32), y[real_rows]
    summary = json.loads((OUT / "detector.json").read_text()) if (OUT / "detector.json").exists() else {}
    for v, bands in RUNS.items():
        for band in bands:
            if band in summary.get(v, {}):
                continue
            t0 = time.time()
            res = evaluate(xr, yr, d[v], d["y"], band=band, threads=4, g_real=pid[real_rows], g_syn=pid[src])
            res["real_rows_sha256"] = hashlib.sha256(real_rows.tobytes()).hexdigest()
            (OUT / f"detector_{v}_{band}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            summary.setdefault(v, {})[band] = {"auc": res["auc"],
                                               "per_class": {k: round(s["mean"], 3) for k, s in res["per_class"].items()}}
            (OUT / "detector.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"{v:7s} {band:8s} AUC {res['auc']['mean']:.3f} [{res['auc']['ci95'][0]:.3f}, "
                  f"{res['auc']['ci95'][1]:.3f}] {summary[v][band]['per_class']} {time.time() - t0:.0f}s", flush=True)
    C.check_frozen()


if __name__ == "__main__":
    main()
