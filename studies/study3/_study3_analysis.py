"""Frozen analysis for study 3 (report/PREREGISTRATION_3.md). Do not edit after registration.
Reads generator_comparison/experiment/study3/L<level>_d<draw>/; writes study3_result.json / .md and figures/
next to them. Refuses to run unless all 180 runs exist, and refuses to run a second time.

Units: 15 per level = 3 real-data draws x 5 seeds, paired across conditions (and across levels, same draw + seed).
  H1 (primary)   synthetic - real at 10 %, test macro-F1, two-sided Wilcoxon signed-rank, alpha 0.05, no adjustment
  Secondary family (Holm, family-wise alpha 0.05):
    H2-10/25/50/100  synthetic - oversample at each level, two-sided Wilcoxon
    H3               trend: the synthetic - OVERSAMPLE gain increases as the real share falls (does synthetic
                     content matter more when real data is scarce?), Page's L test over the 15 matched units,
                     levels ordered 100 < 50 < 25 < 10
    H4               the same trend for synthetic - real (expected: the real-only class weights grow to ~66x for
                     SBRAD at 10 %, so this trend says little about synthetic content on its own)
95 % CIs: three-level bootstrap (test patients x draws x seeds within draw), 2000 resamples, seed 0.
Descriptive (no inference): every condition x level mean and SD; per-class F1 / recall; macro and SBRAD AUPRC;
oversample - real; per-draw mean differences with a one-sample t-test on the 3 draw means; consistency check:
the 100 % cells repeat study 2's real / oversample / random recipe, so their means should be near 0.803 / 0.820 / 0.839.

Usage: py -3.14 _study3_analysis.py"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import page_trend_test, ttest_1samp, wilcoxon

import _study3_common as C
from generator_comparison.eval_metrics import CLASS_ORDER
from generator_comparison.stats import fast_f1, holm

ALPHA, BOOT = 0.05, 2000
STUDY2_MEANS = {"real": 0.8030, "oversample": 0.8199, "synthetic": 0.8393}     # study 2 real / oversample / random
D = C.EXP
COMPARE = {"synthetic-real": ("synthetic", "real"), "synthetic-oversample": ("synthetic", "oversample"),
           "oversample-real": ("oversample", "real")}


def units():
    return [(d, s) for d in C.DRAWS for s in C.train_seeds(d)]


def load(d=D):
    """{level: {cond: {"f1": (15,), "test": [dict], "prob": [array]}}}; refuses if any of the 180 runs is missing."""
    res = {(lv, dr): json.loads((d / f"L{lv}_d{dr}" / "results.json").read_text())
           if (d / f"L{lv}_d{dr}" / "results.json").exists() else {} for lv in C.LEVELS for dr in C.DRAWS}
    missing = [f"L{lv}_d{dr}/{c}/seed{s}" for lv in C.LEVELS for dr, s in units() for c in C.CONDS
               if "test" not in res[lv, dr].get(f"{c}/seed{s}", {})]
    if missing:
        raise SystemExit(f"Incomplete: {len(missing)} runs missing, e.g. {missing[:3]}. The analysis needs all 180.")
    out = {}
    for lv in C.LEVELS:
        out[lv] = {}
        for c in C.CONDS:
            test = [res[lv, dr][f"{c}/seed{s}"]["test"] for dr, s in units()]
            out[lv][c] = {"f1": np.array([t["macro_f1"] for t in test]), "test": test,
                          "prob": [np.load(d / f"L{lv}_d{dr}" / f"test_prob_{c}_seed{s}.npy") for dr, s in units()]}
    return out


def auprc(prob, y):
    from sklearn.metrics import average_precision_score
    return [float(average_precision_score(y == j, prob[:, j])) for j in range(4)]


def bootstrap(R, y, pid, boot, seed=0):
    """{level: {comparison: (boot,)}} of resampled mean differences: patients, then draws, then seeds in each draw."""
    rng = np.random.default_rng(seed)
    upid = np.unique(pid)
    rows = {u: np.where(pid == u)[0] for u in upid}
    U = units()
    pos = {dr: [i for i, (d, _) in enumerate(U) if d == dr] for dr in C.DRAWS}
    pred = {lv: {c: [p.argmax(1) for p in R[lv][c]["prob"]] for c in C.CONDS} for lv in C.LEVELS}
    out = {lv: {k: np.zeros(boot) for k in COMPARE} for lv in C.LEVELS}
    for b in range(boot):
        idx = np.concatenate([rows[u] for u in rng.choice(upid, len(upid), replace=True)])
        draw = np.concatenate([rng.choice(pos[dr], len(pos[dr]), replace=True)
                               for dr in rng.choice(C.DRAWS, len(C.DRAWS))])
        for lv in C.LEVELS:
            f = {c: {i: fast_f1(y[idx], pred[lv][c][i][idx])[0] for i in set(draw)} for c in C.CONDS}
            for k, (a, r) in COMPARE.items():
                out[lv][k][b] = np.mean([f[a][i] - f[r][i] for i in draw])
    return out


def wil(a, b):
    return float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0


def analyse(d, y, pid, boot=BOOT):
    R = load(d)
    U = units()
    draw_of = np.array([dr for dr, _ in U])
    comp = {}
    for lv in C.LEVELS:
        for k, (a, r) in COMPARE.items():
            diff = R[lv][a]["f1"] - R[lv][r]["f1"]
            sd = diff.std(ddof=1)
            comp[f"L{lv} {k}"] = {
                "level": lv, "comparison": k, "n_units": len(diff), "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)), "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                "dz": float(diff.mean() / sd) if sd > 0 else float("nan"), "wilcoxon_p": wil(R[lv][a]["f1"], R[lv][r]["f1"]),
                "draw_mean_diffs": {f"draw{dr}": float(diff[draw_of == dr].mean()) for dr in C.DRAWS},
                "draw_level_t_p": float(ttest_1samp([diff[draw_of == dr].mean() for dr in C.DRAWS], 0).pvalue)}
    def trend(ref):                                           # columns 100, 50, 25, 10: predicted increasing
        g = np.column_stack([R[lv]["synthetic"]["f1"] - R[lv][ref]["f1"] for lv in sorted(C.LEVELS, reverse=True)])
        return page_trend_test(g)
    hyp = {"H1": {**comp["L10 synthetic-real"], "test": "Wilcoxon, two-sided", "adjusted_p": comp["L10 synthetic-real"]["wilcoxon_p"],
                  "adjustment": "none (single primary test)"}}
    sec = {f"H2-{lv}": {**comp[f"L{lv} synthetic-oversample"], "test": "Wilcoxon, two-sided"} for lv in C.LEVELS}
    for v in sec.values():
        v["p"] = v["wilcoxon_p"]
    for h, ref in (("H3", "oversample"), ("H4", "real")):
        pg = trend(ref)
        sec[h] = {"comparison": f"trend of synthetic - {ref} gain over levels 100 < 50 < 25 < 10", "test": "Page's L",
                  "L": float(pg.statistic), "p": float(pg.pvalue),
                  "mean_gain_by_level": {f"L{lv}": comp[f"L{lv} synthetic-{ref}"]["mean_diff"] for lv in C.LEVELS}}
    for h, p in zip(sec, holm([v["p"] for v in sec.values()])):
        sec[h].update(adjusted_p=float(p), adjustment="Holm over H2-10/25/50/100, H3, H4")
    hyp.update(sec)
    bd = bootstrap(R, y, pid, boot)
    for key, v in comp.items():
        x = bd[v["level"]][v["comparison"]]
        v["bootstrap_ci95"] = [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    for h, v in hyp.items():
        if "level" in v:
            v["bootstrap_ci95"] = comp[f"L{v['level']} {v['comparison']}"]["bootstrap_ci95"]
        if v["adjusted_p"] < ALPHA:
            pos = v.get("mean_diff", 1.0) > 0
            v["decision"] = "SUPPORTED" if pos else "SIGNIFICANT, OPPOSITE DIRECTION"
        else:
            v["decision"] = "NOT SUPPORTED"
    desc = {}
    for lv in C.LEVELS:
        desc[f"L{lv}"] = {}
        for c in C.CONDS:
            T, P = R[lv][c]["test"], R[lv][c]["prob"]
            ap = np.array([auprc(p, y) for p in P])
            desc[f"L{lv}"][c] = {
                "macro_f1_mean": float(R[lv][c]["f1"].mean()), "macro_f1_sd": float(R[lv][c]["f1"].std(ddof=1)),
                "accuracy": float(np.mean([t["accuracy"] for t in T])),
                "f1": {cl: float(np.mean([t["f1"][cl] for t in T])) for cl in CLASS_ORDER},
                "recall": {cl: float(np.mean([t["recall"][cl] for t in T])) for cl in CLASS_ORDER},
                "auprc": dict(zip(CLASS_ORDER, map(float, ap.mean(0)))), "macro_auprc": float(ap.mean()),
                "macro_auprc_sd": float(ap.mean(1).std(ddof=1))}
    consistency = {c: {"study3_L100": desc["L100"][c]["macro_f1_mean"], "study2": STUDY2_MEANS[c],
                       "diff": desc["L100"][c]["macro_f1_mean"] - STUDY2_MEANS[c]} for c in C.CONDS}
    return {"alpha": ALPHA, "boot": boot, "hypotheses": hyp, "comparisons": comp, "descriptive": desc,
            "consistency_with_study2": consistency}


def to_md(out):
    L = ["# Study 3 result (pre-registered): data-scarcity curve", "",
         "Test macro-F1, mean ± SD over 15 units (3 draws × 5 seeds) per cell.", "",
         "| real minority kept | real | oversample | synthetic |", "|---|---|---|---|"]
    for lv in C.LEVELS:
        v = out["descriptive"][f"L{lv}"]
        L.append(f"| {lv} % | " + " | ".join(f"{v[c]['macro_f1_mean']:.4f} ± {v[c]['macro_f1_sd']:.4f}" for c in C.CONDS) + " |")
    L += ["", "## Registered hypotheses", "",
          "| | comparison | mean diff | wins | dz | p | adjusted p | 95 % CI | decision |", "|---|---|---|---|---|---|---|---|---|"]
    for h, v in out["hypotheses"].items():
        if h in ("H3", "H4"):
            L.append(f"| {h} | {v['comparison']} | – | – | – | {v['p']:.4g} (L = {v['L']:.0f}) | {v['adjusted_p']:.4g} "
                     f"({v['adjustment']}) | – | **{v['decision']}** |")
            continue
        ci = v["bootstrap_ci95"]
        L.append(f"| {h} | L{v['level']} {v['comparison']} | {v['mean_diff']:+.4f} | {v['wins']}/{v['n_units']} | "
                 f"{v['dz']:.2f} | {v['wilcoxon_p']:.4g} | {v['adjusted_p']:.4g} ({v['adjustment']}) | "
                 f"{ci[0]:+.4f} to {ci[1]:+.4f} | **{v['decision']}** |")
    L += ["", "Consistency check (100 % level repeats study 2's recipe): "
          + "; ".join(f"{c} {v['study3_L100']:.4f} vs study 2 {v['study2']:.4f} ({v['diff']:+.4f})"
                      for c, v in out["consistency_with_study2"].items()) + "."]
    L += ["", "## All paired comparisons (descriptive except where registered above)", "",
          "| level | comparison | mean diff | median | wins | dz | Wilcoxon p (unadjusted) | 95 % CI | per-draw means | draw-level t p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v in out["comparisons"].values():
        ci = v["bootstrap_ci95"]
        L.append(f"| {v['level']} % | {v['comparison']} | {v['mean_diff']:+.4f} | {v['median_diff']:+.4f} | "
                 f"{v['wins']}/{v['n_units']} | {v['dz']:.2f} | {v['wilcoxon_p']:.4g} | {ci[0]:+.4f} to {ci[1]:+.4f} | "
                 + ", ".join(f"{x:+.4f}" for x in v["draw_mean_diffs"].values()) + f" | {v['draw_level_t_p']:.3f} |")
    L += ["", "## Per-class detail (descriptive)", "",
          "| level | condition | accuracy | F1 " + " / ".join(CLASS_ORDER) + " | recall " + " / ".join(CLASS_ORDER)
          + " | AUPRC " + " / ".join(CLASS_ORDER) + " | macro AUPRC |", "|---|---|---|---|---|---|---|"]
    for lv in C.LEVELS:
        for c in C.CONDS:
            v = out["descriptive"][f"L{lv}"][c]
            L.append(f"| {lv} % | {c} | {v['accuracy']:.3f} | " + " / ".join(f"{x:.3f}" for x in v["f1"].values())
                     + " | " + " / ".join(f"{x:.2f}" for x in v["recall"].values()) + " | "
                     + " / ".join(f"{x:.3f}" for x in v["auprc"].values()) + f" | {v['macro_auprc']:.3f} |")
    return "\n".join(L) + "\n"


def figures(out, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = d / "figures"
    fd.mkdir(exist_ok=True)
    lv = list(C.LEVELS)
    ds = out["descriptive"]
    col = {"real": "#6b7280", "oversample": "#d97706", "synthetic": "#2563eb"}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for c in C.CONDS:
        m = np.array([ds[f"L{x}"][c]["macro_f1_mean"] for x in lv])
        se = np.array([ds[f"L{x}"][c]["macro_f1_sd"] for x in lv]) / np.sqrt(15)
        ax[0].errorbar(lv, m, yerr=1.96 * se, marker="o", capsize=3, label=c, color=col[c])
        ax[1].plot(lv, [ds[f"L{x}"][c]["f1"]["SBRAD"] for x in lv], marker="o", label=c, color=col[c])
        ax[2].plot(lv, [ds[f"L{x}"][c]["macro_auprc"] for x in lv], marker="o", label=c, color=col[c])
    for a, t in zip(ax, ("test macro-F1 (mean ± 1.96 SE, 15 units)", "test SBRAD F1", "test macro AUPRC")):
        a.set_xscale("log")
        a.set_xticks(lv, [f"{x} %" for x in lv])
        a.set_xlabel("real minority data kept")
        a.set_title(t)
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(fd / "fig1_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, c in (("synthetic-real", "#2563eb"), ("synthetic-oversample", "#7c3aed"), ("oversample-real", "#d97706")):
        v = [out["comparisons"][f"L{x} {k}"] for x in lv]
        m = np.array([x["mean_diff"] for x in v])
        lo, hi = np.array([x["bootstrap_ci95"][0] for x in v]), np.array([x["bootstrap_ci95"][1] for x in v])
        # percentile CI around the observed mean; clipped at 0 in the rare case the observed mean lies outside it
        ax.errorbar(lv, m, yerr=[np.maximum(m - lo, 0), np.maximum(hi - m, 0)], marker="o", capsize=3, label=k, color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(lv, [f"{x} %" for x in lv])
    ax.set_xlabel("real minority data kept")
    ax.set_ylabel("macro-F1 difference (95 % bootstrap CI)")
    ax.set_title("Paired gains by data level")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "fig2_gains.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for a, x in zip(ax, lv):
        width = 0.27
        for j, c in enumerate(C.CONDS):
            a.bar(np.arange(4) + (j - 1) * width, list(ds[f"L{x}"][c]["recall"].values()), width, label=c, color=col[c])
        a.set_xticks(range(4), CLASS_ORDER)
        a.set_title(f"recall, {x} % real")
        a.set_ylim(0, 1)
        a.grid(alpha=0.3, axis="y")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "fig3_recall.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import pandas as pd
    if (D / "study3_result.json").exists():
        raise SystemExit("study3_result.json exists: the pre-registered analysis runs once. Not overwriting.")
    C.check_frozen()
    y = np.load(C.DATA / "y_test.npy").astype(np.int64)
    pid = pd.read_csv(C.DATA / "test_metadata.csv")["patient_id"].to_numpy()
    out = analyse(D, y, pid)
    (D / "study3_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = to_md(out)
    (D / "study3_result.md").write_text(md, encoding="utf-8")
    figures(out, D)
    print(md)
