"""Study 3, Phase 2 step 1: nested, patient-level real-data subsets (labels and metadata only, no waveforms).

For each draw, train patients are put in a random order (seed 1000 + draw). For each minority class, the kept
recordings at level p are those of the shortest prefix of that patient order holding at least ceil(p * n_class)
recordings of the class, so every patient's recordings of a class are kept or dropped together, and
10 % <= 25 % <= 50 % <= 100 % are nested by construction. SR rows are always kept.
Writes generator_comparison/study3/subsets/draw<d>.npz: L10, L25, L50, L100 = sorted X_train row indices.

Usage: py -3.14 _make_subsets3.py"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from _study3_common import DATA, DRAWS, LEVELS, MINORITY, S3, sha256, subset_path, subset_seed


def make_subsets(y, pid, seed):
    rng = np.random.default_rng(seed)
    upid = np.unique(pid)
    rank = dict(zip(rng.permutation(upid), range(len(upid))))
    r = np.array([rank[p] for p in pid])
    out = {}
    for lv in LEVELS:
        keep = [np.where(y == 0)[0]]
        for c in MINORITY:
            ic = np.where(y == c)[0]
            ic = ic[np.lexsort((ic, r[ic]))]                       # by patient rank, then row
            need = math.ceil(lv / 100 * len(ic))
            last_rank = r[ic[need - 1]]                            # include every row of the patient at the cut
            keep.append(ic[r[ic] <= last_rank])
        out[f"L{lv}"] = np.sort(np.concatenate(keep)).astype(np.int64)
    return out


if __name__ == "__main__":
    y = np.load(DATA / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(DATA / "train_metadata.csv")
    assert len(meta) == len(y) and np.array_equal(meta["label"].to_numpy(), y), "metadata does not match y_train"
    pid = meta["patient_id"].to_numpy()
    (S3 / "subsets").mkdir(parents=True, exist_ok=True)
    summary = {}
    for d in DRAWS:
        s = make_subsets(y, pid, subset_seed(d))
        np.savez(subset_path(d), **s)
        summary[f"draw{d}"] = {"sha256": sha256(subset_path(d)),
                               **{k: np.bincount(y[v], minlength=4).tolist() for k, v in s.items()}}
    (S3 / "subsets" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
