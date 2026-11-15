"""Study 4, Phase 0: baseline detector AUCs under protocol v2 (synthetic_generation/realism/detect.py). CPU, train only.

Source recordings and fitted models come from the 'fit' part, the detector's real side from the 'dev' part
(patient-disjoint, _study4_split.py), so no source recording can appear on the real side.

  E       real ECG rebuilt: own median template at own R-peaks + own residual + own wander   (expected ~0.5)
  C       own template + Gaussian-process residual + own wander
  D       own template + own residual + Gaussian-process wander
  oracle  real templates (fit part) + the full rule-based assembly fitted on the fit part
  gen     the study 1 generated pool (generator_comparison/tstr/pool_beatgen.npz: generated templates + assembly)

Each variant x band (full, low, mid, beats, between) x 5 seeds. Writes generator_comparison/study4/baselines/
(<variant>.npz, detector_<variant>_<band>.json, baselines.json, baselines.md).

Usage: py -3.14 _study4_baselines.py [--variants E,C,...] [--bands full,...] [--threads 4]"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

import _study4_common as C
from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.assembly import assemble, fit_class_models
from synthetic_generation.beats import highpass, place_beats, recording_template
from synthetic_generation.leads import reconstruct_12, to_independent_8
from synthetic_generation.realism.detect import BANDS, evaluate

OUT = C.S4 / "baselines"
VARIANTS = ("E", "C", "D", "oracle", "gen")
N_SRC, N_FIT, N_REAL_SR = 400, 600, 600
SEED = 41


def build_variants(X, y, fit):
    rng = np.random.default_rng(SEED)
    out = {k: ([], [], []) for k in ("E", "C", "D", "oracle")}
    for c in range(4):
        models = fit_class_models(X[fit], y[fit], c, N_FIT, rng)
        ic = fit[y[fit] == c]
        for i in rng.choice(ic, min(N_SRC, len(ic)), replace=False):
            x8 = to_independent_8(X[i:i + 1])[0].astype(np.float64)
            tpl, r, _ = recording_template(x8)
            if tpl is None:
                continue
            hp = highpass(x8)
            beats = place_beats(tpl, r)
            resid, wander = hp - beats, x8 - hp
            parts = {"E": beats + resid + wander,
                     "C": beats + models["residual"].sample(rng) + wander,
                     "D": beats + resid + models["wander"].sample(rng)}
            for k, v in parts.items():
                out[k][0].append(reconstruct_12(v[None].astype(np.float32))[0])
                out[k][1].append(c)
                out[k][2].append(i)
        T = models["templates"]
        for _ in range(N_SRC):
            out["oracle"][0].append(assemble(T[rng.integers(len(T))], models, rng))
            out["oracle"][1].append(c)
        print(CLASS_ORDER[c], "sources", len(out["E"][1]), "templates", models["n_used"], flush=True)
    res = {}
    for k, (xs, ys, src) in out.items():
        res[k] = {"X": np.stack(xs).astype(np.float32), "y": np.asarray(ys, np.int64),
                  "src_idx": np.asarray(src if src else fit, np.int64)}   # oracle: every fit row may have fed it
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--bands", default=",".join(BANDS))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    C.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    fit, dev = C.part_index("fit"), C.part_index("dev")
    assert not set(fit) & set(dev)

    variants = a.variants.split(",")
    need = [v for v in variants if v != "gen" and not (OUT / f"{v}.npz").exists()]
    if need:
        for k, d in build_variants(X, y, fit).items():
            np.savez_compressed(OUT / f"{k}.npz", **d)
    rng = np.random.default_rng(SEED + 1)
    sr = dev[y[dev] == 0]
    real_rows = np.sort(np.concatenate([rng.choice(sr, N_REAL_SR, replace=False), dev[y[dev] != 0]]))
    xr, yr = np.asarray(X[real_rows], np.float32), y[real_rows]

    summary = json.loads((OUT / "baselines.json").read_text()) if (OUT / "baselines.json").exists() else {}
    for v in variants:
        if v == "gen":
            d = np.load(C.ROOT / "generator_comparison" / "tstr" / "pool_beatgen.npz")
        else:
            d = np.load(OUT / f"{v}.npz")
            assert not set(d["src_idx"].tolist()) & set(real_rows.tolist()), "source recording on the real side"
        xs, ys = d["X"], d["y"]
        # detector split grouped by patient: E/C/D by their source recording's patient; pools ECG by ECG
        g_syn = pid[d["src_idx"]] if v in ("E", "C", "D") else None
        for band in a.bands.split(","):
            t0 = time.time()
            res = evaluate(xr, yr, xs, ys, band=band, threads=a.threads, g_real=pid[real_rows], g_syn=g_syn)
            res["real_rows_sha256"] = hashlib.sha256(real_rows.tobytes()).hexdigest()
            (OUT / f"detector_{v}_{band}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            summary.setdefault(v, {})[band] = {"auc": res["auc"],
                                               "per_class": {k: round(s["mean"], 3) for k, s in res["per_class"].items()}}
            (OUT / "baselines.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"{v:7s} {band:8s} AUC {res['auc']['mean']:.3f} [{res['auc']['ci95'][0]:.3f}, "
                  f"{res['auc']['ci95'][1]:.3f}] {summary[v][band]['per_class']} {time.time() - t0:.0f}s", flush=True)
    write_md(summary)
    C.check_frozen()


def write_md(summary):
    bands = [b for b in BANDS if any(b in s for s in summary.values())]
    lines = ["# Study 4 Phase 0: baseline detector AUCs (protocol v2, dev real side, 5 seeds, mean [95 % CI])", "",
             "| variant | " + " | ".join(bands) + " |", "|---" * (len(bands) + 1) + "|"]
    for v, s in summary.items():
        cells = [f"{s[b]['auc']['mean']:.3f} [{s[b]['auc']['ci95'][0]:.2f}, {s[b]['auc']['ci95'][1]:.2f}]"
                 if b in s else "" for b in bands]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    (OUT / "baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
