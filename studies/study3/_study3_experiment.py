"""Study 3 classifier runs for one (level, draw) block: 3 conditions x 5 seeds, the study 2 training recipe
(6,250 fixed steps, deterministic CUDA, best-val checkpoint) from generator_comparison/augment_experiment.py.

  real        SR + the real minority subset at this level
  oversample  real subset; each minority class duplicated (cycled, deterministic) up to TARGET = 2 x full count
  synthetic   real subset; each minority class filled up to TARGET with ECGs drawn at random (seed 1000 + seed)
              from this (level, draw)'s own pool

Writes generator_comparison/experiment/study3/L<level>_d<draw>/ (results.json, config.json, test_prob_*.npy,
val_prob_*.npy). The log shows validation metrics only. --val-only never loads the test split.

Usage: py -3.14 _study3_experiment.py <level> <draw> [--seeds 300,301] [--conditions real,synthetic]
       [--steps 6250] [--val-only] [--out DIR]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import _study3_common as C
from generator_comparison.augment_experiment import metrics, predict, set_deterministic, train_steps

RECIPE_KEYS = ("level", "draw", "steps", "eval_every", "deterministic", "pool_sha256", "subset_sha256", "target")


def build_training_set(cond, seed, Xtr, ytr, idx, pX, py):
    """(X, y, n_synthetic) for one condition. idx: X_train rows of this level (all SR + kept minority)."""
    Xs, ys = Xtr[idx], ytr[idx]
    if cond == "real":
        return Xs, ys, 0
    add_X, add_y, n_syn = [], [], 0
    rng = np.random.default_rng(1000 + seed)
    for c in C.MINORITY:
        need = C.TARGET[c] - int((ys == c).sum())
        if cond == "oversample":
            add_X.append(Xs[np.resize(np.where(ys == c)[0], need)])
        elif cond == "synthetic":
            add_X.append(pX[rng.choice(np.where(py == c)[0], need, replace=False)])
            n_syn += need
        else:
            raise ValueError(cond)
        add_y.append(np.full(need, c, np.int64))
    return np.concatenate([Xs, *add_X]), np.concatenate([ys, *add_y]), n_syn


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("level", type=int, choices=C.LEVELS)
    ap.add_argument("draw", type=int, choices=C.DRAWS)
    ap.add_argument("--seeds", default=None, help="comma list; default: the draw's 5 registered seeds")
    ap.add_argument("--conditions", default=",".join(C.CONDS))
    ap.add_argument("--steps", type=int, default=6250)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    set_deterministic()
    if not torch.cuda.is_available():
        raise SystemExit("study 3 runs need CUDA (deterministic recipe); torch.cuda.is_available() is False.")
    C.check_frozen()
    dev = torch.device("cuda")
    out = Path(a.out) if a.out else C.EXP / f"L{a.level}_d{a.draw}"
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else C.train_seeds(a.draw)

    Xtr = np.load(C.DATA / "X_train.npy").astype(np.float32)
    ytr = np.load(C.DATA / "y_train.npy").astype(np.int64)
    Xva, yva = np.load(C.DATA / "X_val.npy").astype(np.float32), np.load(C.DATA / "y_val.npy").astype(np.int64)
    if not a.val_only:
        Xte, yte = np.load(C.DATA / "X_test.npy").astype(np.float32), np.load(C.DATA / "y_test.npy").astype(np.int64)
    idx = C.train_index(a.draw, a.level)
    pool = np.load(C.pool_path(a.draw, a.level))
    pX, py = pool["X"].astype(np.float32), pool["y"].astype(np.int64)
    for c in C.MINORITY:
        assert (py == c).sum() >= C.TARGET[c] - (ytr[idx] == c).sum(), "pool too small"

    cfg = {"level": a.level, "draw": a.draw, "seeds": seeds, "steps": a.steps, "eval_every": a.eval_every,
           "deterministic": True, "val_only": a.val_only, "target": {str(c): n for c, n in C.TARGET.items()},
           "real_counts": np.bincount(ytr[idx], minlength=4).tolist(),
           "pool": C.pool_path(a.draw, a.level).name,
           "pool_sha256": C.sha256(C.pool_path(a.draw, a.level)), "subset_sha256": C.sha256(C.subset_path(a.draw)),
           "device": str(dev)}
    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    if results and (out / "config.json").exists():
        old = json.loads((out / "config.json").read_text())
        changed = [k for k in RECIPE_KEYS if old.get(k) != cfg[k]]
        if changed:
            raise SystemExit(f"{out} has runs made with a different {changed}; resuming would mix them.")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("CONFIG", json.dumps(cfg), flush=True)

    for cond in a.conditions.split(","):
        for seed in seeds:
            key = f"{cond}/seed{seed}"
            if key in results and (a.val_only or "test" in results[key]):
                print("skip (done)", key)
                continue
            Xa, ya, n_syn = build_training_set(cond, seed, Xtr, ytr, idx, pX, py)
            t0 = time.time()
            net, info = train_steps(Xa, ya, Xva, yva, seed, dev, a.steps, a.eval_every)
            pva = predict(net, Xva, dev)
            np.save(out / f"val_prob_{cond}_seed{seed}.npy", pva)
            r = {**info, "n_train": int(len(ya)), "n_synthetic": n_syn,
                 "train_counts": np.bincount(ya, minlength=4).tolist(), "val": metrics(yva, pva)}
            if not a.val_only:
                pte = predict(net, Xte, dev)
                np.save(out / f"test_prob_{cond}_seed{seed}.npy", pte)
                r["test"] = metrics(yte, pte)
            r["sec"] = round(time.time() - t0, 1)
            results[key] = r
            (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"L{a.level} d{a.draw} {key}: val mF1 {r['val']['macro_f1']:.3f} | val SBRAD F1 "
                  f"{r['val']['f1']['SBRAD']:.3f} | step {r['best_step']} | {r['sec']}s", flush=True)
    C.check_frozen()
    print("done")


if __name__ == "__main__":
    main()
