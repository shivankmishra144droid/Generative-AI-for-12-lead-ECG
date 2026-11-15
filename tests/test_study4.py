"""Study 4 code (Phase 0 of report/WORKFLOW_4.md). Fake data or train-split metadata only.

Run from repo root: py -3.14 -m pytest tests -q
"""
import numpy as np
import pytest

import _study4_common as C
from _study4_split import make_split
from synthetic_generation.realism import detect as D


def test_split_is_patient_disjoint_and_complete():
    rng = np.random.default_rng(0)
    pid = rng.integers(0, 500, 3000)
    s = make_split(pid, 1)
    rows = np.concatenate(list(s.values()))
    assert np.array_equal(np.sort(rows), np.arange(len(pid)))                  # every row once
    pats = [set(pid[v]) for v in s.values()]
    assert not (pats[0] & pats[1] or pats[0] & pats[2] or pats[1] & pats[2])   # no patient in two parts
    shares = np.array([len(p) for p in pats]) / len(np.unique(pid))
    assert np.allclose(shares, C.PART_SHARE, atol=0.06)


def test_real_split_file_if_present():
    if not C.split_path().exists():
        pytest.skip("split not built")
    import pandas as pd
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    d = np.load(C.split_path())
    assert not set(pid[d["fit"]]) & set(pid[d["dev"]]) and not set(pid[d["dev"]]) & set(pid[d["judge"]])
    assert not set(pid[d["fit"]]) & set(pid[d["judge"]])


def test_judge_part_is_guarded():
    with pytest.raises(SystemExit, match="reserved"):
        C.part_index("judge")


def test_group_split_keeps_groups_together():
    rng = np.random.default_rng(1)
    groups = np.array([f"r{g}" for g in rng.integers(0, 40, 300)])
    tr, te = D.group_split(groups, rng)
    assert len(tr) + len(te) == len(groups) and not set(tr) & set(te)
    assert not set(groups[tr]) & set(groups[te])
    assert 0.55 < len(tr) / len(groups) <= 0.7


def test_bands_shape_and_masks():
    rng = np.random.default_rng(2)
    t = np.arange(1000)
    beat = np.exp(-0.5 * ((t[:, None] % 80 - 40) / 2.0) ** 2)                 # spikes every 0.8 s
    x = (beat * rng.uniform(1, 2, 12) + 0.05 * rng.standard_normal((1000, 12)))[None].astype(np.float32)
    for b in D.BANDS:
        assert D.apply_band(x, b).shape == x.shape
    on, off = D.apply_band(x, "beats"), D.apply_band(x, "between")
    assert np.allclose(on + off, x, atol=1e-6)                                 # the two masks split the signal
    assert (on != 0).any() and (off != 0).any()
    low = D.apply_band(x + 3 * np.sin(2 * np.pi * 0.1 * t / 100)[None, :, None].astype(np.float32), "low")
    assert np.abs(low).max() > 2                                               # slow drift survives the low band


def test_detector_finds_an_obvious_difference_and_not_noise():
    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(4), 60)
    real = rng.standard_normal((240, 1000, 12)).astype(np.float32)
    same = rng.standard_normal((240, 1000, 12)).astype(np.float32)
    easy = same + 2.0
    r_same = D.run(real, y, same, y, seed=0, epochs=3)
    r_easy = D.run(real, y, easy, y, seed=0, epochs=3)
    assert r_easy["auc"] > 0.95 and abs(r_same["auc"] - 0.5) < 0.2


def test_frozen_code_guard(monkeypatch, tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "FROZEN_CODE", {"a.py": C.sha256(f)})
    monkeypatch.setattr(C.C3, "check_frozen", lambda: None)
    C.check_frozen()
    f.write_text("x = 2")
    with pytest.raises(SystemExit, match="Frozen generator"):
        C.check_frozen()
