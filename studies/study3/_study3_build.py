"""Study 3, Phase 2: per (draw, level < 100 %), fit the assembly models and train the template generator on that
real-data subset ONLY, then build a pool. The 100 % level uses the frozen generator and assembly models.
Resumable (for run_retry.sh): finished steps are skipped, a generator resumes from its newest checkpoint.

No-leak rule: the assembly models see only X_train[subset]; the generator sees only clean templates whose source
row is in the subset. Both are recorded in <gen_dir>/meta.json and checked by _check_study3.py.

Usage: bash run_retry.sh _study3_build.py generator_comparison/study3/build.log"""
from __future__ import annotations

import json
import pickle
import re
import subprocess
import sys

import numpy as np

import _study3_common as C

# The template generator recipe of _run_template_gen.py (checked against its saved plan.json below).
DIFF = {"T": 200, "beta_0": 5e-4, "beta_T": 0.1}
TRAIN_CFG = {
    "n_iters": 15000, "batch_size": 64, "learning_rate": 2e-4, "warmup_iters": 500, "cosine_decay": True,
    "log_every": 250, "ckpt_every": 2500, "eval_every": 2500, "use_amp": False,
    "ema_decay": 0.999, "ema_warmup": True, "diffusion_config": DIFF, "parameterization": "v",
    "balanced_sampling": True,
}
N_FIT = 600                                                  # as _beat_oracle.py


def recipe_matches_frozen():
    from synthetic_generation.config import WAVENET_4GB
    plan = json.loads((C.ROOT / "generator_comparison" / "template_gen" / "plan.json").read_text())
    return plan["train_cfg"] == TRAIN_CFG and plan["wavenet"] == {**WAVENET_4GB, "s4_lmax": 70}


def fit_assembly(draw, level, X, y):
    from synthetic_generation.assembly import fit_class_models
    out = C.models_path(draw, level)
    if out.exists():
        return
    idx = C.train_index(draw, level)
    Xs, ys = np.asarray(X[idx]), y[idx]
    rng = np.random.default_rng(C.gen_seed(draw, level))
    models = {c: fit_class_models(Xs, ys, c, N_FIT, rng) for c in range(4)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(models, f)
    print(f"L{level} d{draw} assembly fitted:", {c: m["n_used"] for c, m in models.items()}, flush=True)


def subset_templates(draw, level):
    d = np.load(C.ROOT / "generator_comparison" / "beat_templates" / "templates_train.npz")
    keep = d["clean"] & np.isin(d["src_idx"], C.train_index(draw, level))
    return d["templates"][keep].astype(np.float32), d["y"][keep].astype(np.int64), d["src_idx"][keep]


def train_generator(draw, level):
    import synthetic_generation.train as TR
    from synthetic_generation.config import WAVENET_4GB
    gd = C.gen_dir(draw, level)
    if C.ckpt_path(draw, level).exists():
        return
    T, y, src = subset_templates(draw, level)
    ck = gd / "checkpoints"
    done = sorted((int(m.group(1)), p) for p in ck.glob("tpl_*.pt") if (m := re.fullmatch(r"tpl_(\d+)\.pt", p.name)))
    resume = done[-1][1] if done else None
    meta = {"draw": draw, "level": level, "gen_seed": C.gen_seed(draw, level),
            "subset_sha256": C.sha256(C.subset_path(draw)),
            "n_templates": {c: int((y == c).sum()) for c in range(4)},
            "template_src_idx": src.tolist(), "resumed_from": str(resume.name) if resume else None}
    gd.mkdir(parents=True, exist_ok=True)
    (gd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"L{level} d{draw} generator: templates {meta['n_templates']}, resume {meta['resumed_from']}", flush=True)
    TR.SEED = C.gen_seed(draw, level)                       # train_loop seeds from the module-level SEED
    TR.train_loop(T, y, TRAIN_CFG, tag="tpl", wavenet_config={**WAVENET_4GB, "s4_lmax": 70}, resume_path=resume,
                  ckpt_dir=ck, history_path=gd / "history.json")


def build_pool(draw, level):
    out = C.pool_path(draw, level)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + "_tmp.npz")              # a crash mid-write never leaves a "finished" pool
    subprocess.run([sys.executable, "-m", "studies.beat_generator._beat_gen_pool", str(C.ckpt_path(draw, level)), C.pool_counts(), str(tmp),
                    "--seed", str(C.pool_seed(draw, level)), "--models", str(C.models_path(draw, level))],
                   check=True, cwd=C.ROOT)
    tmp.replace(out)


if __name__ == "__main__":
    C.check_frozen()
    assert recipe_matches_frozen(), "generator recipe differs from template_gen/plan.json"
    for d in C.DRAWS:
        assert C.subset_path(d).exists(), "run _make_subsets3.py first"
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    for level in C.LEVELS:
        for d in C.DRAWS:
            if level != 100:
                fit_assembly(d, level, X, y)
                train_generator(d, level)
            build_pool(d, level)
            C.check_frozen()
            print(f"L{level} d{d} done", flush=True)
    print("done")
