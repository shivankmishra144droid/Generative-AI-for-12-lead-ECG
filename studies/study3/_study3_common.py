"""Shared constants and guards for study 3 (data-scarcity curve, report/WORKFLOW_3.md, PREREGISTRATION_3.md).

Every study 3 output lives under generator_comparison/study3/ (data) or generator_comparison/experiment/study3/
(classifier runs). The study 1/2 artifacts are read-only: `check_frozen()` runs at the start and end of every
study 3 script and stops if any of them changed."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # repository root (file lives in studies/study3/)
DATA = ROOT / "processed_data"
S3 = ROOT / "generator_comparison" / "study3"
EXP = ROOT / "generator_comparison" / "experiment" / "study3"

LEVELS = (10, 25, 50, 100)                       # % of each real minority class kept
DRAWS = (1, 2, 3)
MINORITY = (1, 2, 3)                             # AFIB, STACH, SBRAD; SR (0) is never subsampled
FULL = {1: 1056, 2: 568, 3: 446}                 # real train counts
TARGET = {c: 2 * n for c, n in FULL.items()}     # D1: every augmented condition fills each class to 2x its full count
N_SR_POOL = 440                                  # synthetic SR per pool, only for the train-on-synthetic gate (G1)
CONDS = ("real", "oversample", "synthetic")
SEEDS_PER_DRAW = 5


def subset_seed(draw):
    return 1000 + draw                           # 1001 / 1002 / 1003


def gen_seed(draw, level):
    return 3000 + 100 * draw + 10 * LEVELS.index(level)


def pool_seed(draw, level):
    # _beat_gen_pool uses seed + class (0..3) as the diffusion seed, so pool seeds are >= 10 apart
    return 4000 + 100 * draw + 10 * LEVELS.index(level)


def train_seeds(draw):
    s = 300 + SEEDS_PER_DRAW * (draw - 1)
    return list(range(s, s + SEEDS_PER_DRAW))    # draw 1: 300-304, 2: 305-309, 3: 310-314


def subset_path(draw):
    return S3 / "subsets" / f"draw{draw}.npz"


def gen_dir(draw, level):
    return S3 / "generators" / f"L{level}_d{draw}"


def models_path(draw, level):
    if level == 100:
        return ROOT / "generator_comparison" / "beat_templates" / "assembly_models.pkl"
    return gen_dir(draw, level) / "assembly_models.pkl"


def ckpt_path(draw, level):
    if level == 100:
        return ROOT / "generator_comparison" / "template_gen" / "checkpoints" / "tpl_8x64_last.pt"
    return gen_dir(draw, level) / "checkpoints" / "tpl_last.pt"


def pool_path(draw, level):
    return S3 / "pools" / f"pool_L{level}_d{draw}.npz"


def pool_counts():
    """SR,AFIB,STACH,SBRAD counts for _beat_gen_pool: the full fill target, so every seed draws from more than it needs."""
    return f"{N_SR_POOL},{TARGET[1]},{TARGET[2]},{TARGET[3]}"


def train_index(draw, level):
    """Rows of X_train used at this level: all SR rows plus the kept minority rows."""
    return np.load(subset_path(draw))[f"L{level}"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# Registered in report/PREREGISTRATION_2.md (commit d8c7024). Study 3 must never change them.
FROZEN = {
    "generator_comparison/beat_templates/assembly_models.pkl": "b2b133a6b2289c611accdbdafe0b4f315ea361fb170e2fcfde415507337fa9f5",
    "generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt": "19a8e764681f0a0a897a7edf9d0f46565798d686948aa1d43ba076801ad979df",
    "baseline/cnn_lstm_baseline.pt": "51466880ac3c50a9d6957f1d90be58db72fb2f06a40bb94704ba0751fbfd327e",
    "generator_comparison/pools2/pool_1.npz": "bf7daeca6e6ef95d0c6f643885d746246af7037ec5b976560c6361e356cd2141",
    "generator_comparison/pools2/pool_2.npz": "e428a704812b61f06c30f4b52762083a0e2d575265faf9c48b4a0136c24bcbb1",
    "generator_comparison/pools2/pool_3.npz": "95892db476faacdb7855d76da946f052a604c14c923f85bd486fa5f988b8c7c8",
    "generator_comparison/pools2/pool_4.npz": "1d05494af6065e8dbf933d959c89e05450106854a231d95c38c389f108ab6b27",
}


def check_frozen():
    bad = [p for p, h in FROZEN.items() if not (ROOT / p).exists() or sha256(ROOT / p) != h]
    if bad:
        raise SystemExit(f"Frozen study 1/2 artifacts changed or missing: {bad}. Stopping.")
