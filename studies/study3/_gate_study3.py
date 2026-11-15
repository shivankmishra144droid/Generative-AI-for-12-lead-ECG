"""Study 3, Phase 3 validation gate (validation data only; never loads the test split).

  G1 usable generators: train on synthetic only (each pool's 440 SR + 440 per minority class, seed 0, the TSTR
     recipe of generator_comparison/tstr.py), balanced accuracy on real val > 0.35 (chance 0.25). A level below
     100 % whose generator fails is reported as "not trainable at this level"; nothing is retuned.
  G2 memorisation (flag, not a stop): copy fraction <= 0.10 for every minority class (from study3/checks.json).
     A level whose pools fail _check_study3 is flagged in the same way; only a failure at 10 % stops the study.
  G3 determinism: the study recipe (L10, draw 1, seed 300, synthetic, --val-only) run twice in fresh folders gives
     identical validation histories.
Writes generator_comparison/study3/gate.json.

Usage: py -3.14 _gate_study3.py"""
from __future__ import annotations

import json
import shutil

import numpy as np

import _study3_common as C

G1_MIN, G2_MAX, K = 0.35, 0.10, 440


def g1():
    from generator_comparison.tstr import DATA, train_eval
    Xva, yva = np.load(DATA / "X_val.npy").astype(np.float32), np.load(DATA / "y_val.npy")
    out = {}
    for level in C.LEVELS:
        for d in C.DRAWS:
            pool = np.load(C.pool_path(d, level))
            X, y = pool["X"], pool["y"]
            rng = np.random.default_rng(0)
            idx = np.concatenate([rng.choice(np.where(y == c)[0], K, replace=False) for c in range(4)])
            r = train_eval(X[idx].astype(np.float32), y[idx], Xva, yva, 0)
            out[f"L{level}_d{d}"] = {"balanced_acc": r["balanced_acc"], "recall": r["recall"],
                                     "pass": r["balanced_acc"] > G1_MIN}
            print(f"G1 L{level}_d{d}: TSTR bal acc {r['balanced_acc']:.3f}", flush=True)
    by_level = {lv: all(out[f"L{lv}_d{d}"]["pass"] for d in C.DRAWS) for lv in C.LEVELS}
    return {"per_pool": out, "level_pass": by_level, "pass": all(by_level.values())}


def g2():
    chk = json.loads((C.S3 / "checks.json").read_text())
    flags = {name: max(v["copy_frac"] for v in p["memorisation"].values()) for name, p in chk["pools"].items()}
    by_level = {lv: all(chk["pools"][f"L{lv}_d{d}"]["pass"] for d in C.DRAWS) for lv in C.LEVELS}
    return {"max_copy_frac": flags, "flagged": [n for n, v in flags.items() if v > G2_MAX],
            "checks_pass_by_level": by_level}


def g3():
    import _study3_experiment as E
    runs = {}
    for name in ("a", "b"):
        folder = C.EXP.parent / "study3_check" / name
        if folder.exists():
            shutil.rmtree(folder)
        E.main(["10", "1", "--seeds", "300", "--conditions", "synthetic", "--val-only", "--out", str(folder)])
        r = json.loads((folder / "results.json").read_text())["synthetic/seed300"]
        assert "test" not in r and not list(folder.glob("test_prob_*")), "val-only run touched test"
        runs[name] = {"val_history": r["val_history"], "best_step": r["best_step"], "val": r["val"]}
    return {"run_a": runs["a"]["val_history"], "run_b": runs["b"]["val_history"], "pass": runs["a"] == runs["b"]}


if __name__ == "__main__":
    C.check_frozen()
    res = {"G1": g1(), "G2": g2(), "G3": g3()}            # G1 (CPU) before G3 turns on deterministic kernels
    res["GO"] = bool(res["G1"]["level_pass"][10] and res["G3"]["pass"] and res["G2"]["checks_pass_by_level"][10])
    C.check_frozen()
    (C.S3 / "gate.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if k == "GO" else {kk: vv for kk, vv in v.items() if kk not in ("per_pool", "run_a", "run_b")})
                      for k, v in res.items()}, indent=2))
