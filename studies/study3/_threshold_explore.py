"""EXPLORATORY, post hoc, NOT pre-registered (report/THRESHOLD_EXPLORE.md).

Question: is study 3's synthetic > real macro-F1 gain mostly a decision-threshold effect? Every run's decision
rule argmax_c(log p_c + b_c) (b_SR = 0) is tuned on one patient-disjoint half of validation and scored on the
other, then swapped, over R = 50 stratified patient splits. Uses only saved validation probabilities, y_val,
val_metadata and the `best_val_macro_f1` field of each results.json. No training, nothing outside the
validation split is read (every load goes through `_load`, which only accepts allowlisted file names).

Outputs: generator_comparison/threshold_explore/{results.json, summary.md, fig_macro_f1.png, fig_gap.png}.
Run: py -3.14 -B _threshold_explore.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score

import _study3_common as C
from generator_comparison.eval_metrics import CLASS_ORDER

OUT = C.ROOT / "generator_comparison" / "threshold_explore"
R = 50
GRID = np.round(np.arange(-4.0, 4.0 + 1e-9, 0.05), 2)
SWEEPS = 4
N_BOOT = 10000
K = 4
PAIRS = (("synthetic", "real"), ("synthetic", "oversample"), ("oversample", "real"))

_ALLOWED = {"y_val.npy", "val_metadata.csv", "results.json"}
OPENED: list[str] = []


def _load(p: Path):
    """Only validation-split files: y_val.npy, val_metadata.csv, results.json, val_prob_*.npy."""
    name = Path(p).name
    assert name in _ALLOWED or (name.startswith("val_prob_") and name.endswith(".npy")), f"refusing to open {p}"
    OPENED.append(str(p))
    if name.endswith(".npy"):
        return np.load(p)
    if name.endswith(".csv"):
        return pd.read_csv(p)
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- scoring
def f1_from_conf(cm):
    """Macro-F1 from (..., 4, 4) confusion counts [true, pred]; sklearn semantics (0/0 -> 0)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(-2) - tp
    fn = cm.sum(-1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean(-1)


def conf(y, pred):
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(np.float64)


def per_class(y, pred):
    cm = conf(y, pred)
    tp = np.diag(cm)
    npred, ntrue = cm.sum(0), cm.sum(1)
    prec = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    rec = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0)
    den = npred + ntrue
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return f1, prec, rec


def predict(logp, b):
    return np.argmax(logp + b, axis=1)


def tune(logp, y):
    """Coordinate search over GRID for b_AFIB, b_STACH, b_SBRAD (b_SR = 0), maximising macro-F1.
    For class c with the others fixed, ECG i is predicted c iff b_c > d_i = max_{k!=c}(s_ik) - logp_ic, so the
    confusion matrix for every grid value comes from one sort + cumulative counts. Among tied best grid values
    the one closest to the current b_c is kept (so a step never moves without a strict gain)."""
    b = np.zeros(K)
    n = len(y)
    for _ in range(SWEEPS):
        moved = False
        for c in (1, 2, 3):
            s = logp + b
            oth = s.copy()
            oth[:, c] = -np.inf
            k = oth.argmax(1)
            d = oth[np.arange(n), k] - logp[:, c]
            order = np.argsort(d, kind="stable")
            pair = (y * K + k)[order]
            onehot = np.zeros((n + 1, K * K))
            onehot[np.arange(1, n + 1), pair] = 1
            cum = onehot.cumsum(0)
            moved_cnt = cum[np.searchsorted(d[order], GRID, side="left")].reshape(len(GRID), K, K)
            cm = conf(y, k)[None] - moved_cnt
            cm[:, :, c] += moved_cnt.sum(2)
            sc = f1_from_conf(cm)
            best = np.flatnonzero(sc >= sc.max() - 1e-12)
            g = GRID[best[np.argmin(np.abs(GRID[best] - b[c]))]]
            if g != b[c]:
                b[c], moved = g, True
        if not moved:
            break
    return b, float(sc.max())


# ---------------------------------------------------------------- data
def load_all():
    y = _load(C.DATA / "y_val.npy").astype(np.int64)
    meta = _load(C.DATA / "val_metadata.csv")
    assert (meta["label"].to_numpy() == y).all()
    P, ref = {}, {}
    for lv in C.LEVELS:
        for d in C.DRAWS:
            cell = C.EXP / f"L{lv}_d{d}"
            res = _load(cell / "results.json")
            for s in C.train_seeds(d):
                for c in C.CONDS:
                    P[lv, c, d, s] = _load(cell / f"val_prob_{c}_seed{s}.npy").astype(np.float64)
                    ref[lv, c, d, s] = float(res[f"{c}/seed{s}"]["best_val_macro_f1"])
            del res                                   # nothing else from results.json is used
    return y, meta["patient_id"].to_numpy(), P, ref


def make_splits(y, pid):
    """R patient-level halves, stratified by each patient's rarest class (by validation count)."""
    rank = np.argsort(np.argsort(np.bincount(y, minlength=K)))     # 0 = rarest
    df = pd.DataFrame({"pid": pid, "r": rank[y]})
    strat = df.groupby("pid")["r"].min()
    pats, st = strat.index.to_numpy(), strat.to_numpy()
    splits = []
    for r in range(R):
        rng = np.random.default_rng(r)
        inA = set()
        for v in np.unique(st):
            p = rng.permutation(pats[st == v])
            h = len(p) // 2 + (rng.random() < 0.5) * (len(p) % 2)
            inA.update(p[:h].tolist())
        a = np.isin(pid, list(inA))
        splits.append(a)
    return splits


# ---------------------------------------------------------------- analysis
def boot_ci(diff, rng):
    """diff: (3 draws, 5 seeds). Resample draws, then seeds within each drawn draw. Returns replicate means."""
    di = rng.integers(0, 3, (N_BOOT, 3))
    si = rng.integers(0, 5, (N_BOOT, 3, 5))
    return diff[di[:, :, None], si].mean((1, 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y, pid, P, ref = load_all()
    LOGP = {k: np.log(np.clip(v, np.finfo(np.float32).tiny, None)) for k, v in P.items()}
    audit = {}
    print("validation class counts", dict(zip(CLASS_ORDER, np.bincount(y).tolist())), "patients", len(np.unique(pid)))

    # ---- audit 1: loading check
    diffs, diffs_fast = [], []
    for k, p in P.items():
        pred = p.argmax(1)
        diffs.append(abs(f1_score(y, pred, average="macro") - ref[k]))
        diffs_fast.append(abs(f1_from_conf(conf(y, pred)) - ref[k]))
    audit["loading_max_abs_diff_sklearn"] = float(max(diffs))
    audit["loading_max_abs_diff_fast"] = float(max(diffs_fast))
    audit["n_runs"] = len(P)
    print(f"[audit] loading: {len(P)} runs, max |argmax mF1 - best_val_macro_f1| sklearn {max(diffs):.2e}, "
          f"fast scorer {max(diffs_fast):.2e}")
    assert max(diffs) < 1e-9 and max(diffs_fast) < 1e-9, "loading check failed"

    # ---- splits + audit 2: leak check, class counts
    splits = make_splits(y, pid)
    cnt = np.array([[np.bincount(y[a], minlength=K), np.bincount(y[~a], minlength=K)] for a in splits])
    for a in splits:
        assert not set(pid[a]) & set(pid[~a]), "patient in both halves"
        assert a.any() and (~a).any()
    assert (cnt > 0).all()
    audit["leak_check"] = "no patient in both halves, all 50 splits"
    audit["half_class_counts_min"] = dict(zip(CLASS_ORDER, cnt.min((0, 1)).tolist()))
    audit["half_class_counts_max"] = dict(zip(CLASS_ORDER, cnt.max((0, 1)).tolist()))
    audit["split0_counts"] = {"A": cnt[0, 0].tolist(), "B": cnt[0, 1].tolist()}
    print("[audit] leak: OK. per-half class counts over 50 splits: min", audit["half_class_counts_min"],
          "max", audit["half_class_counts_max"])

    # ---- audit 4: null check (all-zero offsets == plain argmax) on every run, split 0 halves
    null_ok = all(np.array_equal(predict(LOGP[k][m], np.zeros(K)), P[k][m].argmax(1))
                  for k in P for m in (splits[0], ~splits[0]))
    assert null_ok
    audit["null_check"] = "argmax(log p + 0) == argmax(p) on every run and half (split 0)"
    print("[audit] null: OK")

    # ---- main loop
    keys = list(P)
    shp = (len(keys), R, 2)
    un_mf, tu_mf, in_mf, in_un = (np.zeros(shp) for _ in range(4))
    un_pc, tu_pc = np.zeros(shp + (3, K)), np.zeros(shp + (3, K))       # (f1, prec, rec)
    offs = np.zeros(shp + (3,))
    search_vs_direct = 0.0
    for r, a in enumerate(splits):
        halves = (a, ~a)
        for h in (0, 1):
            tr, ho = halves[h], halves[1 - h]
            ytr, yho = y[tr], y[ho]
            for i, k in enumerate(keys):
                lp = LOGP[k]
                b, s_search = tune(lp[tr], ytr)
                pin = predict(lp[tr], b)
                s_in = f1_from_conf(conf(ytr, pin))
                search_vs_direct = max(search_vs_direct, abs(s_in - s_search))
                in_mf[i, r, h] = s_in
                in_un[i, r, h] = f1_from_conf(conf(ytr, lp[tr].argmax(1)))
                pu, pt = lp[ho].argmax(1), predict(lp[ho], b)
                un_mf[i, r, h] = f1_from_conf(conf(yho, pu))
                tu_mf[i, r, h] = f1_from_conf(conf(yho, pt))
                un_pc[i, r, h] = per_class(yho, pu)
                tu_pc[i, r, h] = per_class(yho, pt)
                offs[i, r, h] = b[1:]
        if r % 10 == 9:
            print(f"  split {r + 1}/{R} done")

    # ---- audit 3: optimism + search consistency
    audit["search_vs_direct_max_abs_diff"] = float(search_vs_direct)
    assert search_vs_direct < 1e-9, "incremental search score != direct argmax score"
    audit["in_sample_tuned_ge_untuned_every_fit"] = bool((in_mf >= in_un - 1e-12).all())
    assert audit["in_sample_tuned_ge_untuned_every_fit"]
    audit["optimism_mean_in_sample"] = float(in_mf.mean())
    audit["optimism_mean_held_out"] = float(tu_mf.mean())
    assert in_mf.mean() >= tu_mf.mean(), "optimism check failed"
    audit["boundary_hits_frac"] = float((np.abs(offs) >= GRID[-1] - 1e-9).mean())
    audit["boundary_hits_by_class"] = dict(zip(CLASS_ORDER[1:], (np.abs(offs) >= GRID[-1] - 1e-9)
                                               .mean((0, 1, 2)).round(4).tolist()))
    print(f"[audit] search == direct (max diff {search_vs_direct:.1e}); in-sample tuned >= untuned on every fit; "
          f"optimism: in-sample {in_mf.mean():.4f} >= held-out {tu_mf.mean():.4f}; "
          f"offsets on grid edge {audit['boundary_hits_frac']:.4f}")

    # ---- aggregate to units (level, cond) -> (3 draws, 5 seeds)
    idx = {k: i for i, k in enumerate(keys)}

    def unit(arr, lv, c):        # mean over splits and halves
        return np.array([[arr[idx[lv, c, d, s]].mean((0, 1)) for s in C.train_seeds(d)] for d in C.DRAWS])

    def per_split(arr, lv, c):   # (R,) level mean per split (halves averaged)
        return np.mean([arr[idx[lv, c, d, s]].mean(1) for d in C.DRAWS for s in C.train_seeds(d)], 0)

    rng = np.random.default_rng(12345)
    res = {"audit": audit, "levels": {}}
    for lv in C.LEVELS:
        L = {"cond": {}, "gaps": {}}
        U = {c: unit(un_mf, lv, c) for c in C.CONDS}
        T = {c: unit(tu_mf, lv, c) for c in C.CONDS}
        for c in C.CONDS:
            ks = [idx[lv, c, d, s] for d in C.DRAWS for s in C.train_seeds(d)]
            o = offs[ks].reshape(-1, 3)
            ap = np.array([[average_precision_score(y == j, P[keys[i]][:, j]) for j in range(K)] for i in ks])
            fullv = np.array([ref[keys[i]] for i in ks])
            L["cond"][c] = {
                "untuned_mean": float(U[c].mean()), "untuned_sd": float(U[c].std(ddof=1)),
                "tuned_mean": float(T[c].mean()), "tuned_sd": float(T[c].std(ddof=1)),
                "full_val_untuned_mean": float(fullv.mean()),
                "macro_auprc_mean": float(ap.mean()), "macro_auprc_sd": float(ap.mean(1).std(ddof=1)),
                "auprc": dict(zip(CLASS_ORDER, ap.mean(0).round(4).tolist())),
                "offset_median": dict(zip(CLASS_ORDER[1:], np.median(o, 0).round(3).tolist())),
                "offset_q25": dict(zip(CLASS_ORDER[1:], np.percentile(o, 25, 0).round(3).tolist())),
                "offset_q75": dict(zip(CLASS_ORDER[1:], np.percentile(o, 75, 0).round(3).tolist())),
                "per_class": {nm: {"untuned": dict(zip(CLASS_ORDER, unit(un_pc[..., j, :], lv, c).mean((0, 1))
                                                       .round(4).tolist())),
                                   "tuned": dict(zip(CLASS_ORDER, unit(tu_pc[..., j, :], lv, c).mean((0, 1))
                                                     .round(4).tolist()))}
                              for j, nm in enumerate(("f1", "precision", "recall"))},
            }
        cmp = list(PAIRS) + [("synthetic_untuned", "real_tuned")]
        for x, z in cmp:
            if x == "synthetic_untuned":
                du, dt = None, U["synthetic"] - T["real"]
            else:
                du, dt = U[x] - U[z], T[x] - T[z]
            g = {}
            bt = boot_ci(dt, rng)
            g["tuned_mean"] = float(dt.mean())
            g["tuned_unit_sd"] = float(dt.std(ddof=1))
            g["tuned_draw_means"] = dt.mean(1).tolist()
            g["tuned_ci"] = np.percentile(bt, [2.5, 97.5]).tolist()
            w = wilcoxon(dt.ravel())
            g["tuned_wilcoxon_p"] = float(w.pvalue)
            g["tuned_n_positive"] = int((dt > 0).sum())
            if du is not None:
                bu = boot_ci(du, np.random.default_rng(rng.integers(1 << 31)))
                rng2 = np.random.default_rng(int(rng.integers(1 << 31)))
                # share closed: same resample for both gaps, ratio of means per replicate
                di = rng2.integers(0, 3, (N_BOOT, 3))
                si = rng2.integers(0, 5, (N_BOOT, 3, 5))
                mu, mt = du[di[:, :, None], si].mean((1, 2)), dt[di[:, :, None], si].mean((1, 2))
                share = 1 - mt / np.where(np.abs(mu) > 1e-12, mu, np.nan)
                g["untuned_mean"] = float(du.mean())
                g["untuned_ci"] = np.percentile(bu, [2.5, 97.5]).tolist()
                g["untuned_wilcoxon_p"] = float(wilcoxon(du.ravel()).pvalue)
                g["share_closed"] = float(1 - dt.mean() / du.mean())
                g["share_closed_ci"] = np.nanpercentile(share, [2.5, 97.5]).tolist()
                ps_u = per_split(un_mf, lv, x) - per_split(un_mf, lv, z)
                ps_t = per_split(tu_mf, lv, x) - per_split(tu_mf, lv, z)
                g["tuned_gap_over_splits_p5_p50_p95"] = np.percentile(ps_t, [5, 50, 95]).tolist()
                g["untuned_gap_over_splits_p5_p50_p95"] = np.percentile(ps_u, [5, 50, 95]).tolist()
            L["gaps"][f"{x} - {z}"] = g
        res["levels"][f"L{lv}"] = L
        g = L["gaps"]["synthetic - real"]
        print(f"L{lv}: syn-real untuned {g['untuned_mean']:+.4f} tuned {g['tuned_mean']:+.4f} "
              f"share closed {g['share_closed']:.2f}  " +
              "  ".join(f"{c} {L['cond'][c]['untuned_mean']:.4f}->{L['cond'][c]['tuned_mean']:.4f}" for c in C.CONDS))

    # ---- audit 5: what was opened
    assert all(Path(p).name in _ALLOWED or Path(p).name.startswith("val_prob_") for p in OPENED)
    audit["files_opened"] = len(OPENED)
    audit["file_names_opened"] = sorted({Path(p).name.split("_seed")[0] for p in OPENED})
    with open(OUT / "results.json", "w") as f:
        json.dump(res, f, indent=1)
    figures(res)
    print("wrote", OUT)


# ---------------------------------------------------------------- figures
COL = {"real": "#2a78d6", "oversample": "#eb6834", "synthetic": "#1baf7a"}
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lv = list(C.LEVELS)
    x = np.arange(len(lv))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, title in zip(axs, ("untuned", "tuned"), ("Plain argmax", "Offsets tuned on the other half")):
        for j, c in enumerate(C.CONDS):
            m = np.array([res["levels"][f"L{l}"]["cond"][c][f"{key}_mean"] for l in lv])
            se = np.array([res["levels"][f"L{l}"]["cond"][c][f"{key}_sd"] for l in lv]) / np.sqrt(15)
            ax.errorbar(x + (j - 1) * 0.08, m, yerr=1.96 * se, color=COL[c], lw=2, marker="o", ms=7,
                        capsize=3, label=c)
        ax.set_xticks(x, [f"{l} %" for l in lv])
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("real minority data kept", color=INK2)
        _style(ax)
    axs[0].set_ylabel("held-out validation macro-F1\n(mean ± 1.96 SE, 15 runs)", color=INK2)
    axs[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Exploratory: validation macro-F1 before and after per-class offset tuning", color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_macro_f1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, (key, lab, col) in enumerate((("untuned", "before tuning", "#2a78d6"), ("tuned", "after tuning", "#eb6834"))):
        g = [res["levels"][f"L{l}"]["gaps"]["synthetic - real"] for l in lv]
        m = np.array([q[f"{key}_mean"] for q in g])
        ci = np.array([q[f"{key}_ci"] for q in g])
        ax.errorbar(x + (j - 0.5) * 0.12, m, yerr=[m - ci[:, 0], ci[:, 1] - m], color=col, lw=2, marker="o",
                    ms=7, capsize=3, label=lab, ls="none")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, [f"{l} %" for l in lv])
    ax.set_xlabel("real minority data kept", color=INK2)
    ax.set_ylabel("synthetic − real, held-out val macro-F1\n(95 % bootstrap CI over draws, seeds)", color=INK2)
    ax.set_title("Synthetic − real gap before and after tuning", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
