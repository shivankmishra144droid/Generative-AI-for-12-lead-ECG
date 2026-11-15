"""Study 4, Phase 0: patient-level split of the train split into three parts (labels and metadata only, no waveforms).

  fit    40 % of patients: source recordings for semi-synthetic variants; fitting of noise / deviation models
  dev    30 %: the real side of the development detector (Phases 0-5)
  judge  30 %: the real side of the judge detector, used once in Phase 6

Each patient is assigned to one part at random (seed 4004), so a patient's recordings never cross parts.
Writes generator_comparison/study4/split.npz (fit, dev, judge = sorted X_train rows) and split_summary.json.

Usage: py -3.14 _study4_split.py"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _study4_common as C


def make_split(pid, seed):
    rng = np.random.default_rng(seed)
    upid = np.unique(pid)
    part_of = dict(zip(upid, rng.choice(len(C.PARTS), len(upid), p=C.PART_SHARE)))
    a = np.array([part_of[p] for p in pid])
    return {name: np.where(a == k)[0].astype(np.int64) for k, name in enumerate(C.PARTS)}


if __name__ == "__main__":
    C.check_frozen()
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(C.DATA / "train_metadata.csv")
    assert len(meta) == len(y) and np.array_equal(meta["label"].to_numpy(), y), "metadata does not match y_train"
    pid = meta["patient_id"].to_numpy()
    s = make_split(pid, C.SPLIT_SEED)
    C.S4.mkdir(parents=True, exist_ok=True)
    if C.split_path().exists():
        raise SystemExit(f"{C.split_path()} exists; the split is made once.")
    np.savez(C.split_path(), **s)
    summary = {"seed": C.SPLIT_SEED, "share": dict(zip(C.PARTS, C.PART_SHARE)), "sha256": C.sha256(C.split_path()),
               **{k: {"n": int(len(v)), "patients": int(len(np.unique(pid[v]))),
                      "per_class": np.bincount(y[v], minlength=4).tolist()} for k, v in s.items()}}
    (C.S4 / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
