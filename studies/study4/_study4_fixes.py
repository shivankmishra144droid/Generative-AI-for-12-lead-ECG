"""Study 4, Phases 2-3 on real templates (CPU, train split only; report/WORKFLOW_4.md).

1. Fit the models of synthetic_generation/realism/models.py on the 'fit' part: per class the beat-variation model
   (basis U from Phase 1) and the leftover Gaussian process; the wander PCA pooled over classes.
2. Semi-synthetic variants of fit-part recordings (own template at own R-peaks unless stated):

   ref_C   d_k = 0 + current GP residual + own wander             (same sources; baseline C)
   ref_D   own residual + current GP wander                        (same sources; baseline D)
   V1      MODELLED generic d_k (PCA basis) + own leftover + own wander
   V2      MODELLED template-relative d_k (per lead amplitude/shift/width) + own V2 leftover + own wander
   R2      own V2 d_k + GP LEFTOVER (fitted on V2 leftovers) + own wander
   V2R2    MODELLED V2 d_k + GP LEFTOVER + own wander              (full residual replacement; compare ref_C)
   W1      own residual + PCA WANDER                               (wander fix alone; compare ref_D)
   full    MODELLED V2 d_k + GP LEFTOVER + PCA WANDER              (both fixes)
   oracle2 real templates (fit part) + sampled rhythm + v2 assembly (V2, GP leftover, PCA wander)
   gen2    generated templates (study 1 pool) + sampled rhythm + v2 assembly

   Phase 2 audit: V1 (generic deviations) made the R-peak detector miss ~20-28 % of beats on re-detection (real 0 %;
   real deviations borrowed from another same-class recording 12-19 %), so the template-relative V2 was added. The
   first build (V1 only) is kept in generator_comparison/study4/fixes_v1_attempt/.

3. Calibration (before any detector): statistics of the variants vs real fit-part ECGs, re-measured the same way.
4. Memorisation: nearest correlation of sampled wander to the training wander.
5. Detector protocol v2 (dev real side, patient-grouped, 5 seeds) on the bands listed in RUNS.

Writes generator_comparison/study4/fixes/. Usage: py -3.14 _study4_fixes.py [--skip-detector]"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd

import _study4_common as C
from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.assembly import CrossSpectralGP, fit_class_models
from synthetic_generation.beats import highpass, place_beats, recording_template
from synthetic_generation.leads import reconstruct_12, to_independent_8
from synthetic_generation.realism import decompose as Dc
from synthetic_generation.realism import models as M
from synthetic_generation.realism.detect import evaluate

OUT = C.S4 / "fixes"
DIAG = C.S4 / "diagnostics"
N_MODEL, N_SRC, N_POOL, WANDER_RANK, SEED = 400, 400, 400, 60, 61
RUNS = {"ref_C": ["full"], "ref_D": ["full", "low"], "V1": ["full"], "V2": ["full", "beats"], "R2": ["full"],
        "V2R2": ["full", "beats", "between"], "W1": ["full", "low"], "full": ["full", "low", "beats", "between"],
        "oracle2": ["full", "low", "beats", "between"], "gen2": ["full", "low"]}


def load_rec(X, i):
    x8 = to_independent_8(X[i:i + 1])[0].astype(np.float64)
    tpl, r, _ = recording_template(x8)
    return x8, tpl, r


def fit_models(X, y, fit, rng):
    models, wander_all = {}, []
    for c in range(4):
        U = np.load(DIAG / f"basis_{CLASS_ORDER[c]}.npz")["U"]
        ic = fit[y[fit] == c]
        coefs, rs, left, coefs2, left2 = [], [], [], [], []
        for i in rng.choice(ic, min(N_MODEL, len(ic)), replace=False):
            x8, tpl, r = load_rec(X, i)
            if tpl is None:
                continue
            d = Dc.decompose(x8, U)
            coefs.append(d["coef"])
            rs.append(d["r"])
            left.append(d["leftover"])
            wander_all.append(d["wander"])
            d2 = M.template_decompose(x8)
            coefs2.append(d2["coef"])
            left2.append(d2["leftover"])
        old = fit_class_models(X[fit], y[fit], c, 600, rng)                # current models (rhythm, GP residual/wander)
        models[c] = {"U": U, "variation": M.BeatVariation(U, coefs, rs), "leftover": CrossSpectralGP(np.stack(left)),
                     "tvar": M.TemplateVariation(coefs2, rs), "leftover2": CrossSpectralGP(np.stack(left2)),
                     "rhythm": old["rhythm"], "gp_residual": old["residual"], "gp_wander": old["wander"],
                     "templates": old["templates"], "n_used": len(coefs)}
        print(CLASS_ORDER[c], "models from", len(coefs), "recordings; K", len(U), flush=True)
    W = M.WanderPCA(np.stack(wander_all), WANDER_RANK)
    print(f"wander PCA rank {WANDER_RANK}: explains {W.explained:.3f} of real wander variance", flush=True)
    return models, W


def build_variants(X, y, fit, models, W, rng, gen_templates):
    out = {k: ([], [], []) for k in RUNS}

    def add(k, x8, c, src):
        out[k][0].append(reconstruct_12(x8[None].astype(np.float32))[0])
        out[k][1].append(c)
        out[k][2].append(src)

    for c in range(4):
        m = models[c]
        ic = fit[y[fit] == c]
        for i in rng.choice(ic, min(N_SRC, len(ic)), replace=False):
            x8, tpl, r = load_rec(X, i)
            if tpl is None:
                continue
            d = Dc.decompose(x8, m["U"])
            d2 = M.template_decompose(x8)
            hp = x8 - d["wander"]
            tv = m["tvar"]
            mod_v1 = place_beats(m["variation"].beats(tpl, m["variation"].sample(r, rng)), r)
            mod_v2 = place_beats(tv.beats(tpl, tv.sample(r, rng)), r)
            plain = place_beats(tpl, r)
            gp_left2, pca_w = m["leftover2"].sample(rng), W.sample(rng)
            add("ref_C", plain + m["gp_residual"].sample(rng) + d["wander"], c, i)
            add("ref_D", hp + m["gp_wander"].sample(rng), c, i)
            add("V1", mod_v1 + d["leftover"] + d["wander"], c, i)
            add("V2", mod_v2 + d2["leftover"] + d["wander"], c, i)
            add("R2", d2["beats"] + gp_left2 + d["wander"], c, i)
            add("V2R2", mod_v2 + gp_left2 + d["wander"], c, i)
            add("W1", hp + pca_w, c, i)
            add("full", mod_v2 + gp_left2 + pca_w, c, i)
        T = m["templates"]
        for _ in range(N_POOL):
            out["oracle2"][0].append(M.assemble_v2(T[rng.integers(len(T))], m["rhythm"], m["tvar"],
                                                   m["leftover2"], W, rng))
            out["oracle2"][1].append(c)
        G = gen_templates[c]
        for k in range(N_POOL):
            out["gen2"][0].append(M.assemble_v2(G[k % len(G)].astype(np.float64), m["rhythm"], m["tvar"],
                                                m["leftover2"], W, rng))
            out["gen2"][1].append(c)
        print(CLASS_ORDER[c], "variants built", flush=True)
    res = {}
    for k, (xs, ys, src) in out.items():
        res[k] = {"X": np.stack(xs).astype(np.float32), "y": np.asarray(ys, np.int64),
                  "src_idx": np.asarray(src if src else fit, np.int64)}
    return res


def ecg_stats(X12, y, models):
    """Per class, re-measured on 12-lead ECGs the way Phase 1 measured real ones: per-beat coefficient variance and
    lag-1 correlation (own template re-extracted), wander edge/middle increment ratio, kurtosis without the edge
    seconds, RMS quantiles."""
    out = {}
    for c in range(4):
        U = models[c]["U"]
        coefs, wan = [], []
        for x in X12[y == c][:150]:
            x8 = to_independent_8(x[None])[0].astype(np.float64)
            d = Dc.decompose(x8, U)
            if d is None:
                continue
            m = M.full_beats(d["r"])
            if m.sum() >= 3:
                coefs.append(d["coef"][m])
            wan.append(d["wander"])
        A = np.concatenate(coefs)
        num = sum(((a[1:] - a.mean(0)) * (a[:-1] - a.mean(0))).sum(0) for a in coefs)
        den = sum(((a - a.mean(0)) ** 2).sum(0) for a in coefs)
        w = np.stack(wan)
        dw = np.diff(w, axis=2)
        inc = (dw ** 2).mean(axis=(0, 1))
        mid = dw[:, :, 100:-100]
        kurt = ((mid - mid.mean(2, keepdims=True)) ** 4).mean(2) / (mid.var(2) ** 2 + 1e-12)
        rms = np.sqrt((w ** 2).mean(axis=(1, 2)))
        out[CLASS_ORDER[c]] = {"coef_var_total": float(A.var(0).sum()),
                               "coef_var_top3": A.var(0)[:3].round(3).tolist(),
                               "lag1_mean": float(np.mean(num / np.maximum(den, 1e-12))),
                               "wander_edge_to_middle_increment": float(np.r_[inc[:100], inc[-100:]].mean()
                                                                        / inc[100:-100].mean()),
                               "wander_mid_kurtosis": float(np.median(kurt)),
                               "wander_rms_q10_50_90": np.quantile(rms, [0.1, 0.5, 0.9]).round(3).tolist()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-detector", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    C.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    fit, dev = C.part_index("fit"), C.part_index("dev")
    rng = np.random.default_rng(SEED)

    if not (OUT / "models.pkl").exists():
        models, W = fit_models(X, y, fit, rng)
        with open(OUT / "models.pkl", "wb") as f:
            pickle.dump({"models": models, "wander": W}, f)
        g = np.load(C.ROOT / "generator_comparison" / "tstr" / "pool_beatgen.npz")
        gen_templates = {c: g["templates"][g["y"] == c] for c in range(4)}
        for k, d in build_variants(X, y, fit, models, W, rng, gen_templates).items():
            np.savez_compressed(OUT / f"{k}.npz", **d)
    with open(OUT / "models.pkl", "rb") as f:
        mp = pickle.load(f)
    models, W = mp["models"], mp["wander"]

    if not (OUT / "calibration.json").exists():
        real_rows = np.concatenate([fit[y[fit] == c][:150] for c in range(4)])
        cal = {"real_fit": ecg_stats(np.asarray(X[real_rows], np.float32), y[real_rows], models)}
        for k in ("V1", "V2", "W1", "full", "oracle2", "gen2", "ref_C", "ref_D"):
            d = np.load(OUT / f"{k}.npz")
            cal[k] = ecg_stats(d["X"], d["y"], models)
        samples = np.stack([W.sample(np.random.default_rng(s)) for s in range(300)])
        nn = M.nearest_corr(samples, W.train_flat)
        cal["wander_memorisation"] = {"nearest_corr_q50_q99_max": np.quantile(nn, [0.5, 0.99, 1.0]).round(3).tolist(),
                                      "copy_frac_gt_0.98": float((nn > 0.98).mean())}
        cal["models"] = {CLASS_ORDER[c]: {"K": int(len(models[c]["U"])), "beta": models[c]["variation"].beta.round(3)[:3].tolist(),
                                          "n_beats": models[c]["variation"].n_beats,
                                          "scale_q10_50_90": np.quantile(models[c]["variation"].scales, [.1, .5, .9]).round(2).tolist()}
                         for c in range(4)}
        cal["wander_pca_explained"] = W.explained
        (OUT / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
        print(json.dumps(cal, indent=1), flush=True)
    if a.skip_detector:
        return

    rng2 = np.random.default_rng(42)
    sr = dev[y[dev] == 0]
    real_rows = np.sort(np.concatenate([rng2.choice(sr, 600, replace=False), dev[y[dev] != 0]]))
    xr, yr = np.asarray(X[real_rows], np.float32), y[real_rows]
    summary = json.loads((OUT / "detector.json").read_text()) if (OUT / "detector.json").exists() else {}
    for v, bands in RUNS.items():
        d = np.load(OUT / f"{v}.npz")
        semi = v not in ("oracle2", "gen2")
        if semi:
            assert not set(d["src_idx"].tolist()) & set(real_rows.tolist())
        for band in bands:
            if band in summary.get(v, {}):
                continue
            t0 = time.time()
            res = evaluate(xr, yr, d["X"], d["y"], band=band, threads=a.threads, g_real=pid[real_rows],
                           g_syn=pid[d["src_idx"]] if semi else None)
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
