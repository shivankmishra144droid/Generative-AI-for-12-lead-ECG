"""Study 4, D5 escalation: learned residual model (GPU; train split, 'fit' part only for training).

Rules fixed before training (generator_comparison/study4/NOTES_residual.md):
- target r = highpass(x8) - place(template, R); condition place(template, R) + class (residual_diffusion.py)
- cross-fitting: the fit part is split by patient into folds 0/1 (seed 4005); one model per fold, trained once,
  STEPS fixed; a 5 % patient-held-out slice of each fold only tracks the denoising loss (no early stopping)
- every semi-synthetic source gets its residual from the model of the OTHER fold (no memorised noise)
- one global scale constant per fold for the residual and one for the condition, from that fold's training rows
- class-weighted batches (weight 1 / sqrt(class count)), EMA 0.999, Adam 2e-4 with warm-up and cosine decay

Stages:  data | train --fold k | sample | detect
  data    -> generator_comparison/study4/residual/data.npz (cond, res, wander, y, idx, fold, holdout)
  train   -> residual/fold{k}/ckpt.pt (resumable; run through run_retry.sh), history.json
  sample  -> residual/variants.npz: for the V2 sources (same as _study4_fixes.py)
             LR   own template at own R + LEARNED residual + own wander     (compare ref_C 0.999, E 0.5)
             LRW  as LR with PCA wander (fixes/models.pkl)
           plus calibration.json (residual statistics vs the sources' real residuals) and memorisation.json
  detect  -> detector protocol v2 as _study4_fixes.py
Usage: py -3.14 _study4_residual.py <stage> [--fold k]"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time

import numpy as np
import pandas as pd
import torch

import _study4_common as C
from _study4_fixes import load_rec
from _study4_leftover import leftover_stats
from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.beats import highpass, place_beats
from synthetic_generation.leads import reconstruct_12
from synthetic_generation.realism import models as M
from synthetic_generation.realism import residual_diffusion as RD
from synthetic_generation.realism.detect import evaluate

OUT = C.S4 / "residual"
FIX = C.S4 / "fixes"
FOLD_SEED, HOLDOUT_SHARE = 4005, 0.05
STEPS, BATCH, LR, WARMUP, EMA, EVAL_EVERY, CKPT_EVERY = 20000, 16, 2e-4, 500, 0.999, 1000, 1000
TRAIN_SEED = {0: 5000, 1: 5001}
SAMPLE_SEED = 5100
RUNS = {"LR": ["full", "between", "low"], "LRW": ["full"]}


def stage_data():
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    fit = C.part_index("fit")
    rng = np.random.default_rng(FOLD_SEED)
    pats = np.unique(pid[fit])
    fold_of = dict(zip(pats, rng.permutation(len(pats)) % 2))
    hold_of = dict(zip(pats, rng.random(len(pats)) < HOLDOUT_SHARE))
    cond, res, wan, keep = [], [], [], []
    for i in fit:
        x8, tpl, r = load_rec(X, i)
        if tpl is None:
            continue
        hp = highpass(x8)
        c = place_beats(tpl, r, hp.shape[1])
        cond.append(c.astype(np.float32))
        res.append((hp - c).astype(np.float32))
        wan.append((x8 - hp).astype(np.float32))
        keep.append(i)
    keep = np.array(keep)
    np.savez(OUT / "data.npz", cond=np.stack(cond), res=np.stack(res), wander=np.stack(wan), idx=keep, y=y[keep],
             fold=np.array([fold_of[p] for p in pid[keep]]), holdout=np.array([hold_of[p] for p in pid[keep]]))
    f = np.array([fold_of[p] for p in pid[keep]])
    print(f"data: {len(keep)} of {len(fit)} fit recordings (no template: {len(fit) - len(keep)}); "
          f"fold 0 {np.bincount(y[keep][f == 0], minlength=4)}, fold 1 {np.bincount(y[keep][f == 1], minlength=4)}")


def scales(d, fold):
    m = (d["fold"] == fold) & ~d["holdout"]
    return float(d["res"][m].std()), float(d["cond"][m].std())


def stage_train(fold: int):
    torch.manual_seed(TRAIN_SEED[fold])
    rng = np.random.default_rng(TRAIN_SEED[fold])
    dev = torch.device("cuda")
    d = np.load(OUT / "data.npz")
    s_res, s_cond = scales(d, fold)
    tr = np.where((d["fold"] == fold) & ~d["holdout"])[0]
    ho = np.where((d["fold"] == fold) & d["holdout"])[0]
    yy = d["y"]
    w = 1 / np.sqrt(np.bincount(yy[tr], minlength=4)[yy[tr]])
    w /= w.sum()
    res = torch.from_numpy(d["res"] / s_res)
    cond = torch.from_numpy(d["cond"] / s_cond)
    lab = torch.eye(4)[torch.from_numpy(yy)]
    net, dh = RD.build(dev)
    ema, _ = RD.build(dev)
    ema.load_state_dict(net.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    fd = OUT / f"fold{fold}"
    fd.mkdir(parents=True, exist_ok=True)
    hist, start = [], 0
    if (fd / "ckpt.pt").exists():
        ck = torch.load(fd / "ckpt.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        hist, start = ck["hist"], ck["step"]
        rng = np.random.default_rng([TRAIN_SEED[fold], start])
        torch.manual_seed(TRAIN_SEED[fold] * 100000 + start)
        print(f"resumed fold {fold} at step {start}", flush=True)
    if start >= STEPS:
        print("already trained"); return
    g_eval = torch.Generator(device=dev)

    def eval_loss(model):
        model.eval()
        g_eval.manual_seed(0)
        torch.manual_seed(12345)
        tot = 0.0
        with torch.no_grad():
            for k in range(0, len(ho), 64):
                b = ho[k:k + 64]
                tot += RD.v_loss(model, res[b].to(dev), cond[b].to(dev), lab[b].to(dev), dh).item() * len(b)
        model.train()
        return tot / max(len(ho), 1)

    t0, run = time.time(), []
    for step in range(start, STEPS):
        lr = LR * min(1, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        b = rng.choice(tr, BATCH, p=w)
        loss = RD.v_loss(net, res[b].to(dev), cond[b].to(dev), lab[b].to(dev), dh)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(EMA).add_(pn.detach(), alpha=1 - EMA)
        run.append(loss.item())
        if (step + 1) % EVAL_EVERY == 0:
            torch.manual_seed(TRAIN_SEED[fold] * 100000 + step + 1)          # eval_loss reseeds; keep training reproducible
            h = {"step": step + 1, "train": float(np.mean(run)), "holdout_ema": eval_loss(ema), "lr": lr,
                 "sec": round(time.time() - t0)}
            torch.manual_seed(TRAIN_SEED[fold] * 100000 + step + 1)
            hist.append(h)
            run = []
            print(json.dumps(h), flush=True)
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == STEPS:
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "hist": hist,
                        "step": step + 1, "scales": (s_res, s_cond)}, fd / "ckpt.pt.tmp")
            (fd / "ckpt.pt.tmp").replace(fd / "ckpt.pt")
            rng = np.random.default_rng([TRAIN_SEED[fold], step + 1])         # same stream after a resume
    (fd / "history.json").write_text(json.dumps({"scales": [s_res, s_cond], "hist": hist}, indent=1), encoding="utf-8")


def load_model(fold, dev):
    ck = torch.load(OUT / f"fold{fold}" / "ckpt.pt", map_location=dev, weights_only=False)
    assert ck["step"] == STEPS, f"fold {fold} not finished ({ck['step']} / {STEPS})"
    net, dh = RD.build(dev)
    net.load_state_dict(ck["ema"])
    net.eval()
    return net, dh, ck["scales"]


def stage_sample():
    dev = torch.device("cuda")
    d = np.load(OUT / "data.npz")
    pos = {int(i): k for k, i in enumerate(d["idx"])}
    src = np.load(FIX / "V2.npz")["src_idx"]
    src = np.array([i for i in src if int(i) in pos])
    rows = np.array([pos[int(i)] for i in src])
    with open(FIX / "models.pkl", "rb") as f:
        W = pickle.load(f)["wander"]
    gen = np.zeros_like(d["res"][rows])
    for fold in (0, 1):
        net, dh, (s_res, s_cond) = load_model(1 - fold, dev)                 # the OTHER fold's model
        sel = np.where(d["fold"][rows] == fold)[0]
        g = torch.Generator(device=dev).manual_seed(SAMPLE_SEED + fold)
        for k in range(0, len(sel), 64):
            b = rows[sel[k:k + 64]]
            c = torch.from_numpy(d["cond"][b] / s_cond).to(dev)
            lab = torch.eye(4, device=dev)[torch.from_numpy(d["y"][b]).to(dev)]
            gen[sel[k:k + 64]] = (RD.sample(net, c, lab, dh, g) * s_res).cpu().numpy()
            print(f"fold {fold}: {min(k + 64, len(sel))} / {len(sel)}", flush=True)
    rng = np.random.default_rng(SAMPLE_SEED)
    LRx, LRWx = [], []
    for j, rw in enumerate(rows):
        base = d["cond"][rw].astype(np.float64) + gen[j]
        LRx.append(reconstruct_12((base + d["wander"][rw])[None].astype(np.float32))[0])
        LRWx.append(reconstruct_12((base + W.sample(rng))[None].astype(np.float32))[0])
    yv = d["y"][rows]
    np.savez_compressed(OUT / "variants.npz", src_idx=src, y=yv, LR=np.stack(LRx), LRW=np.stack(LRWx),
                        residual=gen.astype(np.float32))
    # calibration: residual statistics, real vs learned, same sources (R-peaks re-derived from the condition rows)
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    rr = [load_rec(X, int(i))[2] for i in src]
    cal = {}
    for c in range(4):
        m = yv == c
        real = d["res"][rows[m]].astype(np.float64)
        lr_ = gen[m].astype(np.float64)
        P = lambda a: (np.abs(np.fft.rfft(a, axis=-1)) ** 2).mean((0, 1))
        f = np.fft.rfftfreq(1000, 0.01)
        bands = [(0, .5), (.5, 2), (2, 5), (5, 15), (15, 25), (25, 50)]
        share = lambda p: [round(float(p[(f >= a) & (f < b)].sum() / p.sum()), 3) for a, b in bands]
        cal[CLASS_ORDER[c]] = {k: dict(leftover_stats(a, [rr[j] for j in np.where(m)[0]]),
                                       band_shares=share(P(a)), log_rms_sd=float(np.log(np.sqrt((a ** 2).mean((1, 2)))).std()))
                               for k, a in (("real", real), ("learned", lr_))}
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(json.dumps(cal, indent=1), flush=True)
    # memorisation: nearest correlation to the generating model's own training residuals, and to the source's own
    mem = {}
    for fold in (0, 1):
        sel = np.where(d["fold"][rows] == fold)[0]
        train_rows = np.where((d["fold"] == 1 - fold) & ~d["holdout"])[0]
        nn = np.concatenate([M.nearest_corr(gen[sel[k:k + 200]], d["res"][train_rows]) for k in range(0, len(sel), 200)])
        a = gen[sel].reshape(len(sel), -1)
        b = d["res"][rows[sel]].reshape(len(sel), -1)
        own = [np.corrcoef(a[k], b[k])[0, 1] for k in range(len(sel))]
        mem[f"fold{fold}_sources"] = {"nearest_train_corr_q50_q99_max": np.quantile(nn, [.5, .99, 1]).round(3).tolist(),
                                      "copy_frac_gt_0.98": float((nn > 0.98).mean()),
                                      "corr_with_own_true_residual_q50_q90": np.quantile(own, [.5, .9]).round(3).tolist()}
    (OUT / "memorisation.json").write_text(json.dumps(mem, indent=1), encoding="utf-8")
    print(json.dumps(mem, indent=1), flush=True)


def stage_pool():
    """Fully assembled ECGs with the learned residual (run only if LR moved; NOTES_residual.md):
      oracleL  real template of a fold-f recording + sampled class rhythm + residual from the fold-(1-f) model + PCA wander
      genL     generated template (study 1 pool, tstr/pool_beatgen.npz) + sampled rhythm + learned residual + PCA wander
               (alternate templates between the two fold models)
    N_POOL per class, like oracle2 / gen2 in _study4_fixes.py."""
    from synthetic_generation.beats import recording_template
    dev = torch.device("cuda")
    d = np.load(OUT / "data.npz")
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    with open(FIX / "models.pkl", "rb") as f:
        mp = pickle.load(f)
    models, W = mp["models"], mp["wander"]
    g = np.load(C.ROOT / "generator_comparison" / "tstr" / "pool_beatgen.npz")
    nets = {k: load_model(k, dev) for k in (0, 1)}
    rng = np.random.default_rng(SAMPLE_SEED + 7)
    tg = torch.Generator(device=dev).manual_seed(SAMPLE_SEED + 7)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    out = {"oracleL": ([], [], []), "genL": ([], [], [])}               # X, y, detector group
    mem = {}
    n_pool = 400
    for c in range(4):
        jobs = []                                                            # (variant, model fold, template)
        cand = {f: np.where((d["y"] == c) & (d["fold"] == f))[0] for f in (0, 1)}
        for k in range(n_pool):
            f = k % 2
            tpl = None
            while tpl is None:
                i_src = int(d["idx"][rng.choice(cand[f])])
                tpl, _, _ = recording_template(to8(X, i_src))
            jobs.append(("oracleL", 1 - f, tpl, f"p{pid[i_src]}"))          # group = source patient (review fix)
        G = g["templates"][g["y"] == c]
        for k in range(n_pool):
            jobs.append(("genL", k % 2, G[k % len(G)].astype(np.float64), f"g{c}_{k % len(G)}"))   # group = template
        for fold in (0, 1):
            net, dh, (s_res, s_cond) = nets[fold]
            sel = [j for j in jobs if j[1] == fold]
            for k in range(0, len(sel), 64):
                chunk = sel[k:k + 64]
                conds = np.stack([place_beats(t, models[c]["rhythm"].sample_r_positions(rng), 1000) for _, _, t, _ in chunk])
                lab = torch.eye(4, device=dev)[torch.full((len(chunk),), c, device=dev)]
                r = (RD.sample(net, torch.from_numpy((conds / s_cond).astype(np.float32)).to(dev), lab, dh, tg)
                     * s_res).cpu().numpy()
                tr_rows = np.where((d["fold"] == fold) & ~d["holdout"])[0]      # this model's training residuals
                nn = M.nearest_corr(r, d["res"][tr_rows])
                mem.setdefault(f"fold{fold}", []).extend(nn.tolist())
                for (v, _, _, grp), cc, rr in zip(chunk, conds, r):
                    out[v][0].append(reconstruct_12((cc + rr + W.sample(rng))[None].astype(np.float32))[0])
                    out[v][1].append(c)
                    out[v][2].append(grp)
        print(CLASS_ORDER[c], "pools done", flush=True)
    np.savez_compressed(OUT / "pools.npz", **{f"{v}_X": np.stack(a) for v, (a, _, _) in out.items()},
                        **{f"{v}_y": np.array(b) for v, (_, b, _) in out.items()},
                        **{f"{v}_g": np.array(g_) for v, (_, _, g_) in out.items()})
    mem = {k: {"nearest_train_corr_q50_q99_max": np.quantile(v, [.5, .99, 1]).round(3).tolist(),
               "copy_frac_gt_0.98": float((np.array(v) > 0.98).mean())} for k, v in mem.items()}
    (OUT / "memorisation_pools.json").write_text(json.dumps(mem, indent=1), encoding="utf-8")
    print(json.dumps(mem, indent=1), flush=True)


def to8(X, i):
    from synthetic_generation.leads import to_independent_8
    return to_independent_8(X[i:i + 1])[0].astype(np.float64)


def stage_detect_pools():
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    dev_rows = C.part_index("dev")
    p = np.load(OUT / "pools.npz")
    rng2 = np.random.default_rng(42)
    sr = dev_rows[y[dev_rows] == 0]
    real_rows = np.sort(np.concatenate([rng2.choice(sr, 600, replace=False), dev_rows[y[dev_rows] != 0]]))
    xr, yr = np.asarray(X[real_rows], np.float32), y[real_rows]
    summary = json.loads((OUT / "detector.json").read_text()) if (OUT / "detector.json").exists() else {}
    for v in ("oracleL", "genL"):
        for band in ("full", "low"):
            if band in summary.get(v, {}):
                continue
            t0 = time.time()
            res = evaluate(xr, yr, p[f"{v}_X"], p[f"{v}_y"], band=band, threads=4, g_real=pid[real_rows],
                           g_syn=p[f"{v}_g"])                            # review fix: repeated templates grouped
            res["real_rows_sha256"] = hashlib.sha256(real_rows.tobytes()).hexdigest()
            (OUT / f"detector_{v}_{band}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            summary.setdefault(v, {})[band] = {"auc": res["auc"],
                                               "per_class": {k: round(s["mean"], 3) for k, s in res["per_class"].items()}}
            (OUT / "detector.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"{v:8s} {band:8s} AUC {res['auc']['mean']:.3f} [{res['auc']['ci95'][0]:.3f}, "
                  f"{res['auc']['ci95'][1]:.3f}] {summary[v][band]['per_class']} {time.time() - t0:.0f}s", flush=True)


def stage_detect():
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    dev_rows = C.part_index("dev")
    d = np.load(OUT / "variants.npz")
    src = d["src_idx"]
    rng2 = np.random.default_rng(42)                                     # same real side as _study4_fixes.py
    sr = dev_rows[y[dev_rows] == 0]
    real_rows = np.sort(np.concatenate([rng2.choice(sr, 600, replace=False), dev_rows[y[dev_rows] != 0]]))
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
            print(f"{v:4s} {band:8s} AUC {res['auc']['mean']:.3f} [{res['auc']['ci95'][0]:.3f}, "
                  f"{res['auc']['ci95'][1]:.3f}] {summary[v][band]['per_class']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "train", "sample", "detect", "pool", "detect_pools"])
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    C.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    {"data": stage_data, "train": lambda: stage_train(a.fold), "sample": stage_sample, "detect": stage_detect,
     "pool": stage_pool, "detect_pools": stage_detect_pools}[a.stage]()
    C.check_frozen()
