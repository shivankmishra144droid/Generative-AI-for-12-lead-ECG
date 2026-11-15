"""Study 4, question U: does a more realistic residual make better training data? (generator_comparison/study4/
PLAN_usefulness.md; exploratory, validation split only, never loads the test split)

Two pools that differ ONLY in the residual: same generated templates (study 1/2 generator, pools2/pool_1.npz
`templates`), same sampled R-peaks (class rhythm models, fixes/models.pkl) and same GP wander sample.
   A  + GP residual (fixes/models.pkl gp_residual, fit part; a GP-residual contrast, not the frozen assembly, which
        also adds respiratory amplitude modulation)
   B  + learned residual (residual/fold{0,1}, alternating; condition = place(template, R))
Counts: the study 2 synthetic counts at ratio 1.0 (= the full real minority counts), index-aligned across A and B.

Stages:
  build     (GPU) residual/useful_pool_{A,B}.npz with X, y, r (true R-peaks, padded with -1)
  realism   (CPU) R-peak errors vs the same detector on the residual-free version (template + wander; saved as
            useful_pool_clean.npz), project detector and wfdb xqrs, and the feature judge,
            for A and B -> useful/realism.json
  train     (GPU) conditions real / A / B, seeds 600..609, study 2 recipe (6,250 steps, deterministic, best-val),
            val probabilities saved -> useful/results.json, useful/val_prob_<cond>_seed<s>.npy
  analyse   (CPU) macro / per-class AUPRC, plain and tuned macro-F1 (_threshold_explore.tune on patient-split
            validation halves, 50 splits), paired B - A -> useful/analysis.json
Usage: py -3.14 _study4_useful.py <stage>"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch

import _study4_common as C

OUT = C.S4 / "useful"
RES = C.S4 / "residual"
FIX = C.S4 / "fixes"
BUILD_SEED, SEEDS, STEPS, EVAL_EVERY = 6100, tuple(range(600, 610)), 6250, 250
CONDS = ("real", "A", "B", "N")              # N: residual-free (amendment U2)
MAXB = 40                                     # max beats stored per ECG


def stage_build():
    from _study4_residual import load_model
    from synthetic_generation.beats import place_beats
    from synthetic_generation.leads import reconstruct_12
    from synthetic_generation.realism import residual_diffusion as RD
    from generator_comparison.augment_experiment import synthetic_counts
    dev = torch.device("cuda")
    ytr = np.load(C.DATA / "y_train.npy").astype(np.int64)
    counts = synthetic_counts(ytr, 1.0)
    with open(FIX / "models.pkl", "rb") as f:
        models = pickle.load(f)["models"]
    pool = np.load(C.ROOT / "generator_comparison" / "pools2" / "pool_1.npz")
    nets = {k: load_model(k, dev) for k in (0, 1)}
    rng = np.random.default_rng(BUILD_SEED)
    tg = torch.Generator(device=dev).manual_seed(BUILD_SEED)
    A, B, CL, Y, Rr = [], [], [], [], []
    for c, n in counts.items():
        T = pool["templates"][pool["y"] == c]
        assert len(T) >= n, f"pool_1 has {len(T)} class-{c} templates, need {n}"
        for k0 in range(0, n, 64):
            ks = range(k0, min(n, k0 + 64))
            rs = [models[c]["rhythm"].sample_r_positions(rng) for _ in ks]
            conds = np.stack([place_beats(T[k].astype(np.float64), r, 1000) for k, r in zip(ks, rs)])
            wand = np.stack([models[c]["gp_wander"].sample(rng) for _ in ks])
            gp = np.stack([models[c]["gp_residual"].sample(rng) for _ in ks])
            fold = (k0 // 64) % 2
            net, dh, (s_res, s_cond) = nets[fold]
            lab = torch.eye(4, device=dev)[torch.full((len(ks),), c, device=dev)]
            lr = (RD.sample(net, torch.from_numpy((conds / s_cond).astype(np.float32)).to(dev), lab, dh, tg)
                  * s_res).cpu().numpy()
            A.append(reconstruct_12((conds + gp + wand).astype(np.float32)))
            B.append(reconstruct_12((conds + lr + wand).astype(np.float32)))
            CL.append(reconstruct_12((conds + wand).astype(np.float32)))         # clean: no residual
            Y.append(np.full(len(ks), c))
            for r in rs:
                Rr.append(np.pad(r[:MAXB], (0, MAXB - len(r[:MAXB])), constant_values=-1))
            print(f"class {c}: {min(n, k0 + 64)} / {n}", flush=True)
    y, r = np.concatenate(Y), np.stack(Rr)
    for name, X in (("A", A), ("B", B), ("clean", CL)):
        np.savez(RES / f"useful_pool_{name}.npz", X=np.concatenate(X), y=y, r=r)
    print("pools:", {c: int((y == c).sum()) for c in (1, 2, 3)})


def stage_realism():
    from _study4_features import cached, judge, univariate, OUT as FOUT
    from _study4_rpeak_check import peaks_project, peaks_xqrs
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    dev = C.part_index("dev")
    rng2 = np.random.default_rng(42)
    sr = dev[y[dev] == 0]
    real_rows = np.sort(np.concatenate([rng2.choice(sr, 600, replace=False), dev[y[dev] != 0]]))
    Fr = cached("real_dev", np.asarray(X[real_rows]))
    out = {}
    clean = np.load(RES / "useful_pool_clean.npz")
    for name in ("A", "B", "N"):
        d = clean if name == "N" else np.load(RES / f"useful_pool_{name}.npz")
        rp = {}
        for det_name, det in (("project", peaks_project), ("xqrs", peaks_xqrs)):
            miss = extra = tot = 0
            err = []
            for x, xc in zip(d["X"], clean["X"]):
                rt, rs = det(xc), det(x)                # truth: the same detector on the residual-free version
                m = sum(not len(rs) or np.min(np.abs(rs - q)) > 5 for q in rt)
                e = sum(not len(rt) or np.min(np.abs(rt - q)) > 5 for q in rs)
                miss, extra, tot = miss + m, extra + e, tot + len(rt)
                err.append(m + e > 0)
            rp[det_name] = {"missed_frac": miss / tot, "spurious_frac": extra / tot, "ecg_with_any_error_frac": float(np.mean(err))}
        Fs = cached(f"useful_{name}", d["X"])
        out[name] = {"rpeak_vs_true": rp, "judge": judge(Fr, y[real_rows], pid[real_rows], Fs, d["y"], np.arange(len(d["y"]))),
                     "univariate": univariate(Fr, y[real_rows], Fs, d["y"])}
        print(name, json.dumps({"rpeak": rp, "judge_auc": out[name]["judge"]["auc"]["mean"]}), flush=True)
    (OUT / "realism.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def stage_train():
    from generator_comparison.augment_experiment import metrics, predict, set_deterministic, train_steps
    set_deterministic()
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    Xtr = np.load(C.DATA / "X_train.npy").astype(np.float32)
    ytr = np.load(C.DATA / "y_train.npy").astype(np.int64)
    Xva, yva = np.load(C.DATA / "X_val.npy").astype(np.float32), np.load(C.DATA / "y_val.npy").astype(np.int64)
    pools = {k: np.load(RES / f"useful_pool_{k}.npz") for k in ("A", "B")}
    pools["N"] = np.load(RES / "useful_pool_clean.npz")
    assert (pools["A"]["y"] == pools["B"]["y"]).all() and (pools["A"]["r"] == pools["B"]["r"]).all()
    results = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    for seed in SEEDS:
        for cond in CONDS:
            key = f"{cond}/seed{seed}"
            if key in results:
                continue
            if cond == "real":
                Xa, ya = Xtr, ytr
            else:
                Xa = np.concatenate([Xtr, pools[cond]["X"].astype(np.float32)])
                ya = np.concatenate([ytr, pools[cond]["y"].astype(np.int64)])
            t0 = time.time()
            net, info = train_steps(Xa, ya, Xva, yva, seed, dev, STEPS, EVAL_EVERY)
            pva = predict(net, Xva, dev)
            np.save(OUT / f"val_prob_{cond}_seed{seed}.npy", pva)
            results[key] = {**info, "n_train": int(len(ya)), "train_counts": np.bincount(ya, minlength=4).tolist(),
                            "val": metrics(yva, pva), "sec": round(time.time() - t0, 1)}
            (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"{key}: val mF1 {results[key]['val']['macro_f1']:.3f} | step {info['best_step']} | "
                  f"{results[key]['sec']}s", flush=True)


def stage_analyse():
    from scipy.stats import wilcoxon
    from sklearn.metrics import average_precision_score
    import _threshold_explore as TE
    yv = np.load(C.DATA / "y_val.npy").astype(np.int64)
    pidv = pd.read_csv(C.DATA / "val_metadata.csv")["patient_id"].to_numpy()
    if (OUT / "analysis.json").exists():
        raise SystemExit("analysis.json exists: the U2 analysis runs once (PLAN_usefulness.md amendment U2).")
    res = json.loads((OUT / "results.json").read_text())
    assert all(f"{c}/seed{s}" in res for c in CONDS for s in SEEDS), "incomplete runs"
    splits = TE.make_splits(yv, pidv)
    m = {c: {"auprc": [], "auprc_cls": [], "f1": [], "f1_tuned": [], "afib_confusions": [], "afib_prec": [],
             "afib_rec": []} for c in CONDS}
    for seed in SEEDS:
        for c in CONDS:
            p = np.load(OUT / f"val_prob_{c}_seed{seed}.npy").astype(np.float64)
            assert abs(TE.f1_from_conf(TE.conf(yv, p.argmax(1))) - res[f"{c}/seed{seed}"]["best_val_macro_f1"]) < 1e-9
            ap = [average_precision_score(yv == j, p[:, j]) for j in range(4)]
            logp = np.log(np.clip(p, np.finfo(np.float32).tiny, None))
            un, tu = [], []
            for a in splits:
                for tr, te in ((a, ~a), (~a, a)):
                    b, _ = TE.tune(logp[tr], yv[tr])
                    tu.append(TE.f1_from_conf(TE.conf(yv[te], TE.predict(logp[te], b))))
                    un.append(TE.f1_from_conf(TE.conf(yv[te], p[te].argmax(1))))
            cm = TE.conf(yv, p.argmax(1))                       # [true, pred]; AFIB = 1, regular = 0, 2, 3
            m[c]["afib_confusions"].append(float(cm[[0, 2, 3], 1].sum() + cm[1, [0, 2, 3]].sum()))
            m[c]["afib_prec"].append(float(cm[1, 1] / max(cm[:, 1].sum(), 1)))
            m[c]["afib_rec"].append(float(cm[1, 1] / max(cm[1].sum(), 1)))
            m[c]["auprc"].append(float(np.mean(ap)))
            m[c]["auprc_cls"].append(ap)
            m[c]["f1"].append(float(np.mean(un)))
            m[c]["f1_tuned"].append(float(np.mean(tu)))
    rng = np.random.default_rng(0)
    out = {"per_condition": {c: {k: (np.mean(v, 0).round(4).tolist() if k == "auprc_cls" else
                                     [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)])
                                 for k, v in d.items()} for c, d in m.items()}, "paired": {}}
    for a_, b_ in (("N", "A"), ("B", "A"), ("N", "B"), ("A", "real"), ("B", "real"), ("N", "real")):
        for k in ("auprc", "f1", "f1_tuned"):
            dlt = np.array(m[a_][k]) - np.array(m[b_][k])
            bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(10000)]
            out["paired"][f"{a_}-{b_} {k}"] = {"mean": round(float(dlt.mean()), 4), "wins": f"{int((dlt > 0).sum())}/{len(dlt)}",
                                               "wilcoxon_p": float(wilcoxon(dlt).pvalue) if np.any(dlt != 0) else 1.0,
                                               "ci95": np.quantile(bs, [.025, .975]).round(4).tolist()}
    for a_ in ("N", "B"):
        for j, nm in ((3, "SBRAD"), (2, "STACH")):
            dd = np.array(m[a_]["auprc_cls"])[:, j] - np.array(m["A"]["auprc_cls"])[:, j]
            out["paired"][f"{a_}-A {nm} auprc"] = {"mean": round(float(dd.mean()), 4), "wins": f"{int((dd > 0).sum())}/{len(dd)}"}
    mc = {c: float(np.mean(m[c]["afib_confusions"])) for c in ("A", "B", "N")}
    out["mechanism"] = {"afib_regular_confusions_mean": mc, "ordered_A_gt_B_gt_N": bool(mc["A"] > mc["B"] > mc["N"]),
                        "afib_precision_mean": {c: round(float(np.mean(m[c]["afib_prec"])), 4) for c in CONDS},
                        "afib_recall_mean": {c: round(float(np.mean(m[c]["afib_rec"])), 4) for c in CONDS}}
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "realism", "train", "analyse"])
    a = ap.parse_args()
    C.check_frozen()
    {"build": stage_build, "realism": stage_realism, "train": stage_train, "analyse": stage_analyse}[a.stage]()
    C.check_frozen()
