"""Study 4, Phase 3 escalation: what beat-to-beat variation does the detector accept? (CPU, train split only)

Same fit-part sources as _study4_fixes.py's V2 (own template at own R-peaks + own V2 leftover + own wander); the
template-relative coefficients p_k (per lead amplitude / shift / width) replaced by:

   VS   own real coefficients, permuted across the recording's full beats   tests order / RR coupling
   VB   real coefficient vectors of a donor recording of the same class      tests whether real variation transfers
        (donor's full beats drawn with replacement)                          to another template (diagnostic only:
                                                                             a bank of real vectors is not a model)
Calibration before the detector: deviation energy (variant vs own) and the R-peak re-detection miss rate, the
measure on which V1 failed (Phase 2 audit). Detector protocol v2, full band.
Writes generator_comparison/study4/variation/. Usage: py -3.14 _study4_variation.py"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

import _study4_common as C
from _study4_fixes import load_rec
from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.beats import place_beats, recording_template
from synthetic_generation.leads import reconstruct_12
from synthetic_generation.realism import models as M
from synthetic_generation.realism.detect import evaluate

OUT = C.S4 / "variation"
FIX = C.S4 / "fixes"
SEED = 81
VARIANTS = ("VS", "VB")


def main():
    C.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "train_metadata.csv")["patient_id"].to_numpy()
    dev = C.part_index("dev")
    src = np.load(FIX / "V2.npz")["src_idx"]
    rng = np.random.default_rng(SEED)

    if not (OUT / "variants.npz").exists():
        dec = {}
        for i in src:
            x8, tpl, r = load_rec(X, i)
            dec[i] = (x8, M.template_decompose(x8))
        out = {k: [] for k in VARIANTS}
        cal = {CLASS_ORDER[c]: {k: {"energy": [], "own_energy": [], "missed": [], "beats": 0} for k in VARIANTS}
               for c in range(4)}
        for i in src:
            x8, d2 = dec[i]
            c = int(y[i])
            full = M.full_beats(d2["r"])
            if full.sum() < 2:
                donors = [i]
            else:
                donors = [j for j in src[y[src] == c] if j != i and M.full_beats(dec[j][1]["r"]).sum() >= 2]
            j = donors[rng.integers(len(donors))]
            pj = dec[j][1]["coef"][M.full_beats(dec[j][1]["r"])]
            p_own = d2["coef"]
            new = {"VS": p_own.copy(), "VB": p_own.copy()}
            fi = np.where(full)[0]
            new["VS"][fi] = p_own[rng.permutation(fi)]
            new["VB"][fi] = pj[rng.integers(len(pj), size=len(fi))]
            tpl = d2["template"]
            for k in VARIANTS:
                beats = place_beats(tpl[None] + M.template_deviation(tpl, new[k]), d2["r"], x8.shape[1])
                x = beats + d2["leftover"] + d2["wander"]
                out[k].append(reconstruct_12(x[None].astype(np.float32))[0])
                cc = cal[CLASS_ORDER[c]][k]
                cc["energy"].append(float((M.template_deviation(tpl, new[k][fi]) ** 2).sum(axis=(1, 2)).mean()) if len(fi) else 0.0)
                cc["own_energy"].append(float((M.template_deviation(tpl, p_own[fi]) ** 2).sum(axis=(1, 2)).mean()) if len(fi) else 0.0)
                _, r2, _ = recording_template(x)
                matched = sum(np.min(np.abs(r2 - q)) <= 5 for q in d2["r"]) if r2 is not None and len(r2) else 0
                cc["missed"].append(1 - matched / max(len(d2["r"]), 1))
                cc["beats"] += len(d2["r"])
        summ = {c: {k: {"deviation_energy_variant_over_own": float(np.mean(v["energy"]) / max(np.mean(v["own_energy"]), 1e-12)),
                        "missed_beat_rate": float(np.mean(v["missed"])), "n_beats": v["beats"]} for k, v in d.items()}
                for c, d in cal.items()}
        (OUT / "calibration.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
        print(json.dumps(summ, indent=1), flush=True)
        np.savez_compressed(OUT / "variants.npz", src_idx=src, y=y[src], **{k: np.stack(v) for k, v in out.items()})

    d = np.load(OUT / "variants.npz")
    rng2 = np.random.default_rng(42)                                     # same real side as _study4_fixes.py
    sr = dev[y[dev] == 0]
    real_rows = np.sort(np.concatenate([rng2.choice(sr, 600, replace=False), dev[y[dev] != 0]]))
    assert not set(src.tolist()) & set(real_rows.tolist())
    xr, yr = np.asarray(X[real_rows], np.float32), y[real_rows]
    summary = json.loads((OUT / "detector.json").read_text()) if (OUT / "detector.json").exists() else {}
    for v in VARIANTS:
        if "full" in summary.get(v, {}):
            continue
        t0 = time.time()
        res = evaluate(xr, yr, d[v], d["y"], band="full", threads=4, g_real=pid[real_rows], g_syn=pid[src])
        res["real_rows_sha256"] = hashlib.sha256(real_rows.tobytes()).hexdigest()
        (OUT / f"detector_{v}_full.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        summary.setdefault(v, {})["full"] = {"auc": res["auc"],
                                             "per_class": {k: round(s["mean"], 3) for k, s in res["per_class"].items()}}
        (OUT / "detector.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{v:4s} full AUC {res['auc']['mean']:.3f} [{res['auc']['ci95'][0]:.3f}, {res['auc']['ci95'][1]:.3f}] "
              f"{summary[v]['full']['per_class']} {time.time() - t0:.0f}s", flush=True)
    C.check_frozen()


if __name__ == "__main__":
    main()
