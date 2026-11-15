"""Study 3 code (Phase 1 of report/WORKFLOW_3.md). Fake data, or train-split files only: no val/test array is read.

Run from repo root: py -3.14 -m pytest tests -q
"""
import json
import math

import numpy as np
import pytest
import torch

import _study3_analysis as A
import _study3_common as C
import _study3_experiment as E
from _make_subsets3 import make_subsets


def fake_labels(rng, n=2000):
    y = rng.choice(4, n, p=[0.85, 0.07, 0.045, 0.035])
    pid = rng.integers(0, 900, n)                           # many patients have several recordings
    return y.astype(np.int64), pid


def test_subsets_nested_patient_level_and_sized():
    y, pid = fake_labels(np.random.default_rng(0))
    s = make_subsets(y, pid, seed=1)
    prev = None
    for lv in C.LEVELS:
        cur = s[f"L{lv}"]
        assert np.array_equal(cur, np.unique(cur))
        assert set(np.where(y == 0)[0]) <= set(cur)                     # SR always kept
        for c in C.MINORITY:
            n = int((y[cur] == c).sum())
            assert n >= math.ceil(lv / 100 * (y == c).sum())
            kept = set(pid[cur][y[cur] == c])
            dropped = set(pid[(y == c) & ~np.isin(np.arange(len(y)), cur)])
            assert not kept & dropped                                    # a patient's class-c rows go together
        assert prev is None or set(prev) <= set(cur)                     # nested
        prev = cur
    assert len(s["L100"]) == len(y)
    assert not np.array_equal(make_subsets(y, pid, 2)["L10"], s["L10"])   # draws differ


def test_seed_ranges_are_new_and_spaced():
    ts = [s for d in C.DRAWS for s in C.train_seeds(d)]
    assert ts == list(range(300, 315))                                   # never used in studies 1-2 (0-239)
    ps = [C.pool_seed(d, lv) for d in C.DRAWS for lv in C.LEVELS]
    # _beat_gen_pool passes seed + c to generate_class_batch, which seeds with that + 1000 + c
    noise = lambda p: [p + 1000 + 2 * c for c in range(4)]
    diffusion = [s for p in ps for s in noise(p)]
    assert len(set(diffusion)) == len(diffusion)
    assert not set(diffusion) & {s for p in (7, 11, 21, 31, 41) for s in noise(p)}
    gs = [C.gen_seed(d, lv) for d in C.DRAWS for lv in C.LEVELS]
    assert len(set(gs)) == len(gs) and not set(gs) & set(ps)


def fake_block(rng, n_sr=60):
    y = np.concatenate([np.zeros(n_sr), np.full(10, 1), np.full(6, 2), np.full(4, 3)]).astype(np.int64)
    X = rng.normal(size=(len(y), 50, 12)).astype(np.float32)
    idx = np.concatenate([np.arange(n_sr), n_sr + np.array([0, 1, 2]), n_sr + 10 + np.array([0, 1]), [n_sr + 16]])
    py = np.repeat([0, 1, 2, 3], [5, 20, 12, 8])
    pX = rng.normal(size=(len(py), 50, 12)).astype(np.float32) + 100        # synthetic rows are easy to tell apart
    return X, y, idx, pX, py


@pytest.mark.parametrize("cond", C.CONDS)
def test_training_set_fill_rule(monkeypatch, cond):
    monkeypatch.setattr(C, "TARGET", {1: 20, 2: 12, 3: 8})
    rng = np.random.default_rng(3)
    X, y, idx, pX, py = fake_block(rng)
    Xa, ya, n_syn = E.build_training_set(cond, 300, X, y, idx, pX, py)
    assert np.array_equal(Xa[:len(idx)], X[idx]) and np.array_equal(ya[:len(idx)], y[idx])   # the real subset, once
    extra_X, extra_y = Xa[len(idx):], ya[len(idx):]
    if cond == "real":
        assert len(extra_y) == 0 and n_syn == 0
        return
    assert {c: int((ya == c).sum()) for c in C.MINORITY} == C.TARGET     # same totals in both augmented conditions
    if cond == "synthetic":
        assert n_syn == len(extra_y) and (extra_X.mean(axis=(1, 2)) > 50).all()      # all from the pool
        for c in C.MINORITY:                                                      # no pool row used twice
            rows = extra_X[extra_y == c].reshape((extra_y == c).sum(), -1)
            assert len(np.unique(rows, axis=0)) == len(rows)
        assert not np.array_equal(Xa, E.build_training_set(cond, 301, X, y, idx, pX, py)[0])  # seed changes the draw
    else:
        assert n_syn == 0
        subset_rows = {X[i].tobytes() for i in idx}
        assert all(r.tobytes() in subset_rows for r in extra_X)                 # only duplicates of kept real rows


def fake_setup(monkeypatch, tmp_path):
    """Fake train/val (no test files), one subset file and one pool; points study 3 at them."""
    rng = np.random.default_rng(4)
    data, s3 = tmp_path / "data", tmp_path / "s3"
    data.mkdir()
    X, y, idx, pX, py = fake_block(rng)
    for name, arr in [("X_train", X), ("y_train", y), ("X_val", X[::3]), ("y_val", y[::3])]:
        np.save(data / f"{name}.npy", arr)
    monkeypatch.setattr(C, "DATA", data)
    monkeypatch.setattr(C, "S3", s3)
    monkeypatch.setattr(C, "EXP", tmp_path / "exp")
    monkeypatch.setattr(C, "TARGET", {1: 20, 2: 12, 3: 8})
    monkeypatch.setattr(C, "check_frozen", lambda: None)
    monkeypatch.setattr(E, "set_deterministic", lambda: None)
    C.subset_path(1).parent.mkdir(parents=True)
    np.savez(C.subset_path(1), **{f"L{lv}": idx for lv in C.LEVELS})
    C.pool_path(1, 10).parent.mkdir(parents=True)
    np.savez(C.pool_path(1, 10), X=pX, y=py)
    return tmp_path / "exp" / "L10_d1"


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="study 3 runs need CUDA")


@needs_cuda
def test_val_only_never_touches_test(monkeypatch, tmp_path):
    out = fake_setup(monkeypatch, tmp_path)
    E.main(["10", "1", "--seeds", "300", "--steps", "2", "--eval-every", "1", "--val-only"])
    res = json.loads((out / "results.json").read_text())
    assert set(res) == {f"{c}/seed300" for c in C.CONDS}
    assert all("test" not in r and r["steps"] == 2 for r in res.values())
    assert not list(out.glob("test_prob_*")) and len(list(out.glob("val_prob_*"))) == 3
    assert res["real/seed300"]["train_counts"] == [60, 3, 2, 1]
    assert res["synthetic/seed300"]["train_counts"] == res["oversample/seed300"]["train_counts"] == [60, 20, 12, 8]


@needs_cuda
def test_resume_refuses_changed_recipe(monkeypatch, tmp_path):
    fake_setup(monkeypatch, tmp_path)
    base = ["10", "1", "--seeds", "300", "--conditions", "real", "--eval-every", "1", "--val-only"]
    E.main(base + ["--steps", "2"])
    with pytest.raises(SystemExit, match="different"):
        E.main(base + ["--steps", "3"])


def test_frozen_guard_stops_on_change(monkeypatch, tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "FROZEN", {"a.bin": C.sha256(f)})
    C.check_frozen()
    f.write_bytes(b"y")
    with pytest.raises(SystemExit, match="Frozen"):
        C.check_frozen()


def test_generator_templates_come_from_the_subset_only():
    import _study3_build as B
    if not C.subset_path(1).exists():
        pytest.skip("subsets not built")
    T, y, src = B.subset_templates(1, 10)
    idx = set(C.train_index(1, 10).tolist())
    assert set(src.tolist()) <= idx
    ytr = np.load(C.DATA / "y_train.npy")
    assert np.array_equal(y, ytr[src])
    assert (y == 3).sum() <= 45 and (y == 0).sum() > 10000           # SBRAD scarce, SR complete


def test_build_recipe_matches_frozen_generator():
    import _study3_build as B
    assert B.recipe_matches_frozen()


def write_fake_study(d, f1_of, rng, n_test=120):
    for lv in C.LEVELS:
        for dr in C.DRAWS:
            folder = d / f"L{lv}_d{dr}"
            folder.mkdir(parents=True)
            res = {}
            for c in C.CONDS:
                for s in C.train_seeds(dr):
                    m = f1_of(lv, c, dr, s)
                    res[f"{c}/seed{s}"] = {"test": {"macro_f1": m, "accuracy": m,
                                                    "f1": {k: m for k in ("SR", "AFIB", "STACH", "SBRAD")},
                                                    "recall": {k: m for k in ("SR", "AFIB", "STACH", "SBRAD")}}}
                    np.save(folder / f"test_prob_{c}_seed{s}.npy", rng.dirichlet(np.ones(4), n_test))
            (folder / "results.json").write_text(json.dumps(res))
    return rng.integers(0, 4, n_test), np.repeat(np.arange(n_test // 2), 2)


def test_analysis_detects_scarcity_effect_and_trend(tmp_path):
    rng = np.random.default_rng(5)
    noise = {(lv, c, dr, s): rng.normal(0, 0.003) for lv in C.LEVELS for c in C.CONDS for dr in C.DRAWS
             for s in C.train_seeds(dr)}
    gain = {10: 0.08, 25: 0.05, 50: 0.03, 100: 0.02}
    f = lambda lv, c, dr, s: 0.75 + {"real": 0, "oversample": gain[lv] / 2, "synthetic": gain[lv]}[c] + noise[lv, c, dr, s]
    y, pid = write_fake_study(tmp_path, f, rng)
    out = A.analyse(tmp_path, y, pid, boot=10)
    h = out["hypotheses"]
    assert h["H1"]["decision"] == "SUPPORTED" and h["H1"]["adjusted_p"] == h["H1"]["wilcoxon_p"]
    assert h["H1"]["n_units"] == 15 and abs(h["H1"]["mean_diff"] - 0.08) < 0.005
    assert h["H3"]["decision"] == "SUPPORTED" and all(h[f"H2-{lv}"]["decision"] == "SUPPORTED" for lv in C.LEVELS)
    assert h["H4"]["decision"] == "SUPPORTED"
    assert all(h[k]["adjusted_p"] >= h[k]["p"] for k in ("H2-10", "H2-25", "H2-50", "H2-100", "H3", "H4"))
    md = A.to_md(out)
    assert "H1" in md and "Per-class detail" in md
    A.figures(out, tmp_path)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3


def test_analysis_null(tmp_path):
    rng = np.random.default_rng(6)
    y, pid = write_fake_study(tmp_path, lambda lv, c, dr, s: 0.8 + rng.normal(0, 0.003), rng)
    h = A.analyse(tmp_path, y, pid, boot=10)["hypotheses"]
    assert h["H1"]["decision"] == "NOT SUPPORTED"


def test_analysis_refuses_incomplete(tmp_path):
    rng = np.random.default_rng(7)
    write_fake_study(tmp_path, lambda *a: 0.8, rng)
    p = tmp_path / "L25_d2" / "results.json"
    res = json.loads(p.read_text())
    del res["synthetic/seed306"]
    p.write_text(json.dumps(res))
    with pytest.raises(SystemExit, match="Incomplete"):
        A.load(tmp_path)


def test_launcher_covers_all_blocks(monkeypatch):
    import runpy
    calls = []
    monkeypatch.setattr(E, "main", lambda argv=None: calls.append(tuple(argv)))
    runpy.run_path("studies/study3/_run_study3.py", run_name="__main__")
    assert sorted(calls) == sorted((str(lv), str(d)) for lv in C.LEVELS for d in C.DRAWS)
