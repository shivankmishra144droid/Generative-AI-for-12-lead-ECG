"""Study 4, side question F: which clinical features give synthetic ECGs away? (CPU; PLAN_features.md)

Real side: the CNN detector's real rows (dev part, 600 SR + all minority, seed 42).
Sets: realreal (fit-part real ECGs, E's sources, as they are) and E = controls; C, D, oracle (baselines), gen (the
current generator = study 1 pool), and LR (learned residual) if residual/variants.npz exists.

1. Features (synthetic_generation/realism/features.py) per ECG, cached in features/feat_<set>.npz.
2. Univariate: per feature, class-stratified AUC (mean over the 4 classes of the per-class AUC synthetic vs real),
   NaNs dropped, with real and synthetic medians.
3. Multivariate judge: HistGradientBoosting on all features + class one-hot; per seed, equal real/synthetic counts per
   class and a patient-grouped 70/30 split (detect.group_split), 5 seeds, t-interval; permutation importance (AUC
   drop) on the held-out side.
Controls first: if realreal or E gives a multivariate AUC >= 0.6 or any univariate AUC >= 0.65, stop (PLAN F1).
Usage: py -3.14 _study4_features.py"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

import _study4_common as C
from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.realism import features as F
from synthetic_generation.realism.detect import group_split, summarise

OUT = C.S4 / "features"
SEEDS = (0, 1, 2, 3, 4)


def cached(name, X12):
    p = OUT / f"feat_{name}.npz"
    if p.exists():
        return np.load(p)["F"]
    t0 = time.time()
    Fm = F.features_12(X12)
    np.savez_compressed(p, F=Fm)
    print(f"features {name}: {len(Fm)} ECGs, {time.time() - t0:.0f}s", flush=True)
    return Fm


def univariate(Fr, yr, Fs, ys):
    out = {}
    for j, name in enumerate(F.NAMES):
        aucs = []
        for c in range(4):
            a, b = Fr[yr == c, j], Fs[ys == c, j]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) > 5 and len(b) > 5 and np.ptp(np.r_[a, b]) > 0:
                aucs.append(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
        m = float(np.mean(aucs)) if aucs else np.nan
        out[name] = {"auc_syn_higher": m, "separation": abs(m - 0.5) + 0.5 if aucs else np.nan,
                     "median_real": float(np.nanmedian(Fr[:, j])), "median_syn": float(np.nanmedian(Fs[:, j])),
                     "nan_real": int(np.isnan(Fr[:, j]).sum()), "nan_syn": int(np.isnan(Fs[:, j]).sum())}
    return out


def judge(Fr, yr, gr, Fs, ys, gs):
    runs, imps = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n_c = [min((yr == c).sum(), (ys == c).sum()) for c in range(4)]
        ri = np.concatenate([rng.choice(np.where(yr == c)[0], n_c[c], replace=False) for c in range(4)])
        si = np.concatenate([rng.choice(np.where(ys == c)[0], n_c[c], replace=False) for c in range(4)])
        Xa = np.r_[Fr[ri], Fs[si]]
        Xa = np.c_[Xa, np.eye(4)[np.r_[yr[ri], ys[si]]]]
        lab = np.r_[np.zeros(len(ri)), np.ones(len(si))]
        groups = np.concatenate([[f"r{g}" for g in gr[ri]], [f"s{g}" for g in gs[si]]])
        tr, te = group_split(groups, rng)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(Xa[tr], lab[tr])
        runs.append(roc_auc_score(lab[te], m.predict_proba(Xa[te])[:, 1]))
        pi = permutation_importance(m, Xa[te], lab[te], scoring="roc_auc", n_repeats=5, random_state=seed)
        imps.append(pi.importances_mean[:len(F.NAMES)])
    imp = np.mean(imps, 0)
    order = np.argsort(-imp)
    return {"auc": summarise(runs), "runs": runs,
            "importance": {F.NAMES[k]: round(float(imp[k]), 4) for k in order}}


def main():
    C.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    dev = C.part_index("dev")
    rng2 = np.random.default_rng(42)                                     # same real side as the CNN detector
    sr = dev[y[dev] == 0]
    real_rows = np.sort(np.concatenate([rng2.choice(sr, 600, replace=False), dev[y[dev] != 0]]))
    Fr, yr, gr = cached("real_dev", np.asarray(X[real_rows])), y[real_rows], pid[real_rows]
    B = C.S4 / "baselines"
    e = np.load(B / "E.npz")
    sets = {"realreal": (np.asarray(X[np.sort(e["src_idx"])]), y[np.sort(e["src_idx"])], pid[np.sort(e["src_idx"])]),
            "E": (e["X"], e["y"], pid[e["src_idx"]])}
    for k in ("C", "D"):
        d = np.load(B / f"{k}.npz")
        sets[k] = (d["X"], d["y"], pid[d["src_idx"]])
    d = np.load(B / "oracle.npz")
    sets["oracle"] = (d["X"], d["y"], np.arange(len(d["y"])))
    g = np.load(C.ROOT / "generator_comparison" / "tstr" / "pool_beatgen.npz")
    sets["gen"] = (g["X"], g["y"], np.arange(len(g["y"])))
    lr = C.S4 / "residual" / "variants.npz"
    if lr.exists():
        d = np.load(lr)
        sets["LR"] = (d["LR"], d["y"], pid[d["src_idx"]])
    res = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    for name, (Xs, ys, gs) in sets.items():
        if name in res:
            continue
        Fs = cached(name, Xs)
        r = {"univariate": univariate(Fr, yr, Fs, ys), "judge": judge(Fr, yr, gr, Fs, ys, gs), "n": int(len(ys))}
        res[name] = r
        (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        top = sorted(r["univariate"].items(), key=lambda kv: -np.nan_to_num(kv[1]["separation"]))[:5]
        print(f"{name:8s} judge AUC {r['judge']['auc']['mean']:.3f} [{r['judge']['auc']['ci95'][0]:.3f}, "
              f"{r['judge']['auc']['ci95'][1]:.3f}] | top univariate: "
              + ", ".join(f"{k} {v['auc_syn_higher']:.2f}" for k, v in top), flush=True)
        if name in ("realreal", "E"):
            worst = max(v["separation"] for v in r["univariate"].values() if np.isfinite(v["separation"]))
            if r["judge"]["auc"]["mean"] >= 0.6 or worst >= 0.65:
                raise SystemExit(f"CONTROL FAILED on {name}: judge {r['judge']['auc']['mean']:.3f}, "
                                 f"max univariate separation {worst:.3f}. Stop and find the leak (PLAN F1).")
    C.check_frozen()


if __name__ == "__main__":
    main()
