"""Study 3, Phase 2 checks (train split only; no val/test). Per (level, draw):
  1. leak: the generator's training templates are exactly the clean templates of the subset, and every template
     inside the fitted assembly models is a subset template (none from a dropped recording)
  2. memorisation: share of generated minority templates with corr > 0.98 to a template of the SAME subset
     (real-to-real reference 0.5-3 %); also median nearest-neighbour corr ("realism")
  3. replay: re-assembling every ECG from its stored template and the pool seed reproduces X; the placed R-peak
     rate is out of class (STACH <= 100 bpm, SBRAD >= 60 bpm) for <= 2 %
  4. sanity: counts, no NaN / inf, share with |z| > 20
Writes generator_comparison/study3/checks.json.

Usage: py -3.14 _check_study3.py"""
from __future__ import annotations

import json
import pickle

import numpy as np

import _study3_common as C
from synthetic_generation.assembly import MEAN_RR_BOUNDS, assemble

OUT_OF_CLASS = {2: lambda hr: hr <= 100, 3: lambda hr: hr >= 60}
TT = np.load(C.ROOT / "generator_comparison" / "beat_templates" / "templates_train.npz")


def zrows(a):
    a = a.reshape(len(a), -1).astype(np.float64)
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)


def key(t):
    return np.asarray(t, np.float32).round(5).tobytes()


def subset_clean(draw, level):
    keep = TT["clean"] & np.isin(TT["src_idx"], C.train_index(draw, level))
    return keep


def leak_check(draw, level, models):
    keep = subset_clean(draw, level)
    out = {}
    if level != 100:
        meta = json.loads((C.gen_dir(draw, level) / "meta.json").read_text())
        out["generator_templates_equal_subset"] = sorted(meta["template_src_idx"]) == sorted(TT["src_idx"][keep].tolist())
    inside = {key(t) for t in TT["templates"][keep]}
    outside = {key(t) for t in TT["templates"][~np.isin(TT["src_idx"], C.train_index(draw, level))]}
    for c in C.MINORITY:
        ts = [key(t) for t in models[c]["templates"]]
        out[f"assembly_c{c}_in_subset"] = float(np.mean([t in inside for t in ts]))
        out[f"assembly_c{c}_from_dropped"] = int(sum(t in outside for t in ts))
    out["pass"] = bool(out.get("generator_templates_equal_subset", True)
                       and all(out[f"assembly_c{c}_from_dropped"] == 0 for c in C.MINORITY)
                       and all(out[f"assembly_c{c}_in_subset"] == 1.0 for c in C.MINORITY))
    return out


def memorisation(draw, level, T, y):
    keep = subset_clean(draw, level)
    out = {}
    for c in C.MINORITY:
        ref = zrows(TT["templates"][keep & (TT["y"] == c)])
        g = zrows(T[y == c])
        nn = np.concatenate([(g[i:i + 512] @ ref.T / g.shape[1]).max(1) for i in range(0, len(g), 512)])
        out[c] = {"n_ref": int(len(ref)), "copy_frac": float((nn > 0.98).mean()), "nn_corr_median": float(np.median(nn))}
    return out


def replay(draw, level, X, y, T, models):
    rng = np.random.default_rng(C.pool_seed(draw, level))
    err, hr = 0.0, {c: [] for c in C.MINORITY}
    for i in range(len(y)):
        c = int(y[i])
        state = rng.bit_generator.state
        r = models[c]["rhythm"].sample_r_positions(rng)
        rng.bit_generator.state = state
        err = max(err, float(np.abs(assemble(T[i].astype(np.float64), models[c], rng) - X[i]).max()))
        if c in hr:
            hr[c].append(60 * 100 / np.diff(r).mean())
    return err, {c: float(np.mean(OUT_OF_CLASS[c](np.array(hr[c])))) for c in OUT_OF_CLASS}, \
        {c: float(np.median(v)) for c, v in hr.items()}


if __name__ == "__main__":
    C.check_frozen()
    res = {"subsets_sha256": {f"draw{d}": C.sha256(C.subset_path(d)) for d in C.DRAWS}, "pools": {}}
    for level in C.LEVELS:
        for d in C.DRAWS:
            name = f"L{level}_d{d}"
            pool = np.load(C.pool_path(d, level))
            X, y, T = pool["X"], pool["y"], pool["templates"]
            models = pickle.load(open(C.models_path(d, level), "rb"))
            for c, m in models.items():
                m["rhythm"].mean_rr_bounds = MEAN_RR_BOUNDS[c]
            r = {"counts": np.bincount(y, minlength=4).tolist(),
                 "leak": leak_check(d, level, models), "memorisation": memorisation(d, level, T, y)}
            r["replay_max_abs_err"], r["true_out_of_class"], r["true_median_hr"] = replay(d, level, X, y, T, models)
            r["nonfinite"] = int((~np.isfinite(X)).any(axis=(1, 2)).sum())
            r["abs_z_gt_20_frac"] = float((np.abs(X).max(axis=(1, 2)) > 20).mean())
            r["sha256"] = {"pool": C.sha256(C.pool_path(d, level)), "ckpt": C.sha256(C.ckpt_path(d, level)),
                           "models": C.sha256(C.models_path(d, level))}
            r["pass"] = bool(r["leak"]["pass"] and r["replay_max_abs_err"] < 1e-4
                             and all(v <= 0.02 for v in r["true_out_of_class"].values())
                             and r["nonfinite"] == 0
                             and r["counts"] == [C.N_SR_POOL, C.TARGET[1], C.TARGET[2], C.TARGET[3]])
            res["pools"][name] = r
            mem = " ".join(f"{c}:{v['copy_frac']:.3f}" for c, v in r["memorisation"].items())
            print(f"{name}: pass {r['pass']} | leak {r['leak']['pass']} | copy {mem} | out-of-class "
                  f"{r['true_out_of_class']} | replay {r['replay_max_abs_err']:.1e} | |z|>20 {r['abs_z_gt_20_frac']:.4f}",
                  flush=True)
    res["all_pass"] = all(v["pass"] for v in res["pools"].values())
    C.check_frozen()
    (C.S3 / "checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("ALL PASS" if res["all_pass"] else "SOME CHECK FAILED")
