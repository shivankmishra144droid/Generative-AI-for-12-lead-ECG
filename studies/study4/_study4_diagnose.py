"""Study 4, Phase 1: diagnose where the giveaway is (CPU, 'fit' part of the train split only).

Per class, on up to N_REC fit-part recordings (the same ones that feed every later model):
 1. Per-beat decomposition: PCA basis of tapered beat deviations; rank K = the smallest rank explaining >= 80 % of the
    deviation variance (rule fixed here, before any detector result on it).
 2. Residual variance against time from the nearest R-peak (-0.5 ... +1.0 s), for:
      real residual (d_k = 0, what the current generator models), real leftover (after the per-beat fit),
      and the current Gaussian-process residual (fitted on the same recordings).
 3. Beat-to-beat statistics of the coefficients a_k: variance per component, lag-1 / lag-2 correlation across beats,
    correlation with the preceding RR interval.
 4. Wander: RMS distribution, < 0.5 Hz spectrum, non-stationarity (RMS ratio of the two halves, kurtosis of the
    first difference), cross-lead correlation, and correlation of wander RMS with residual RMS across recordings;
    real vs the current Gaussian-process wander.
 5. Leftover flatness: max / median of its time-from-R variance profile (1 = flat, i.e. stationary is enough).
Writes generator_comparison/study4/diagnostics/ (diagnostics.json, basis_<class>.npz, fig_*.png). The calibration
targets for Phases 2-3 are the 'targets' block of diagnostics.json.

Usage: py -3.14 _study4_diagnose.py"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _study4_common as C
from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.assembly import fit_class_models
from synthetic_generation.beats import FS, highpass, recording_template
from synthetic_generation.leads import to_independent_8
from synthetic_generation.realism import decompose as Dc

OUT = C.S4 / "diagnostics"
N_REC, N_FIT, SEED, EV_TARGET, MAX_RANK = 400, 600, 51, 0.80, 48
OFFSETS = np.arange(-50, 101)                     # samples from the nearest R-peak (-0.5 ... +1.0 s)
EDGE = 50                                          # first / last 0.5 s left out of the phase profiles


def phase_profile(sig: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum of squares (over leads) and counts at each offset from the nearest preceding-or-following R-peak."""
    T = sig.shape[1]
    t = np.arange(T)
    if not len(r):
        return np.zeros(len(OFFSETS)), np.zeros(len(OFFSETS))
    k = np.clip(np.searchsorted(r, t, side="right") - 1, 0, len(r) - 1)    # preceding R
    off = t - r[k]
    nxt = np.clip(k + 1, 0, len(r) - 1)
    before_next = t - r[nxt]                                                 # negative: before the next R
    use_next = (before_next >= OFFSETS[0]) & (before_next < 0) & (nxt != k)
    off = np.where(use_next, before_next, off)
    ok = (off >= OFFSETS[0]) & (off <= OFFSETS[-1]) & (t >= EDGE) & (t < T - EDGE)   # edges: partial beats
    e = (sig ** 2).sum(0)
    s = np.bincount(off[ok] - OFFSETS[0], weights=e[ok], minlength=len(OFFSETS))
    n = np.bincount(off[ok] - OFFSETS[0], minlength=len(OFFSETS)).astype(float)
    return s, n


def wander_stats(w: np.ndarray) -> dict:
    """w: (n, 8, T)."""
    rms = np.sqrt((w ** 2).mean(axis=(1, 2)))
    half = w.shape[2] // 2
    r1 = np.sqrt((w[:, :, :half] ** 2).mean(axis=(1, 2)))
    r2 = np.sqrt((w[:, :, half:] ** 2).mean(axis=(1, 2)))
    d = np.diff(w, axis=2)
    kurt = ((d - d.mean(2, keepdims=True)) ** 4).mean(2) / (d.var(2) ** 2 + 1e-12)
    spec = (np.abs(np.fft.rfft(w, axis=2)) ** 2).mean(axis=(0, 1))
    f = np.fft.rfftfreq(w.shape[2], 1 / FS)
    cc = np.nanmean([np.corrcoef(x)[np.triu_indices(8, 1)].mean() for x in w])   # flat (zero) leads give nan
    return {"rms_quantiles": np.quantile(rms, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "half_rms_log_ratio_sd": float(np.std(np.log((r1 + 1e-9) / (r2 + 1e-9)))),
            "diff_kurtosis_median": float(np.median(kurt)),
            "spectrum_share_below_0.2Hz": float(spec[f < 0.2].sum() / spec[f < 0.5].sum()),
            "cross_lead_corr_mean": float(cc)}


def diagnose_class(X, y, fit, c, rng):
    ic = fit[y[fit] == c]
    idx = rng.choice(ic, min(N_REC, len(ic)), replace=False)
    recs = []
    for i in idx:
        x8 = to_independent_8(X[i:i + 1])[0].astype(np.float64)
        tpl, r, info = recording_template(x8)
        if tpl is None:
            continue
        recs.append((i, x8, tpl, r, highpass(x8)))
    devs = np.concatenate([Dc.beat_deviations(hp, tpl, r) for _, _, tpl, r, hp in recs])
    B = Dc.fit_basis(devs, MAX_RANK)
    K = int(np.searchsorted(B["explained"], EV_TARGET) + 1)
    U = B["U"][:K]
    np.savez_compressed(OUT / f"basis_{CLASS_ORDER[c]}.npz", U=U, explained=B["explained"], src_idx=[i for i, *_ in recs])

    models = fit_class_models(X[[i for i, *_ in recs]], y[[i for i, *_ in recs]], c, N_FIT, rng)
    prof = {k: [np.zeros(len(OFFSETS)), np.zeros(len(OFFSETS))] for k in ("residual", "leftover", "gp")}
    coefs, rr_prev, w_real, w_gp, res_rms, wan_rms = [], [], [], [], [], []
    for i, x8, tpl, r, hp in recs:
        d = Dc.decompose(x8, U)
        for k, sig in (("residual", hp - Dc.place_beats(tpl, r, hp.shape[1])), ("leftover", d["leftover"]),
                       ("gp", models["residual"].sample(rng))):
            s, n = phase_profile(sig, r)
            prof[k][0] += s
            prof[k][1] += n
        coefs.append(d["coef"])
        rr_prev.append(np.concatenate([[np.nan], np.diff(r) / FS]))
        w_real.append(d["wander"])
        w_gp.append(models["wander"].sample(rng))
        res_rms.append(np.sqrt(((hp - Dc.place_beats(tpl, r, hp.shape[1])) ** 2).mean()))
        wan_rms.append(np.sqrt((d["wander"] ** 2).mean()))
    curves = {k: (s / np.maximum(n, 1)).tolist() for k, (s, n) in prof.items()}

    def lag_corr(lag):
        num, den = [], []
        for a in coefs:
            if len(a) > lag + 2:
                a0 = a - a.mean(0)
                num.append((a0[lag:] * a0[:-lag]).sum(0))
                den.append((a0 ** 2).sum(0))
        return (np.sum(num, 0) / np.maximum(np.sum(den, 0), 1e-12)).tolist()

    rr_corr = []
    for a, rr in zip(coefs, rr_prev):
        m = ~np.isnan(rr)
        if m.sum() > 4 and np.std(rr[m]) > 1e-6:
            rr_corr.append([np.corrcoef(a[m, j], rr[m])[0, 1] for j in range(K)])
    allc = np.concatenate(coefs)
    left = np.array(curves["leftover"])
    out = {"n_recordings": len(recs), "n_beats": int(len(allc)), "rank": K,
           "explained_at_rank": float(B["explained"][K - 1]),
           "explained_curve": B["explained"][:MAX_RANK].round(4).tolist(),
           "coef_var": allc.var(0).tolist(), "coef_within_recording_var": float(np.mean([a.var(0).sum() for a in coefs])),
           "coef_total_var": float(allc.var(0).sum()),
           "lag1_corr": lag_corr(1), "lag2_corr": lag_corr(2),
           "rr_corr_mean": np.nanmean(rr_corr, 0).tolist() if rr_corr else None,
           "phase_curves": curves,
           "leftover_flatness": float(left.max() / np.median(left)),
           "residual_flatness": float(np.max(curves["residual"]) / np.median(curves["residual"])),
           "gp_flatness": float(np.max(curves["gp"]) / np.median(curves["gp"])),
           "wander_real": wander_stats(np.stack(w_real)), "wander_gp": wander_stats(np.stack(w_gp)),
           "wander_vs_residual_rms_corr": float(np.corrcoef(np.log(wan_rms), np.log(res_rms))[0, 1])}
    return out


def figures(res):
    t = OFFSETS / FS
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharex=True)
    for a, (name, r) in zip(ax, res.items()):
        for k, lab in (("residual", "real residual (d_k = 0)"), ("leftover", f"real leftover (K = {r['rank']})"),
                       ("gp", "Gaussian-process residual (current)")):
            a.plot(t, r["phase_curves"][k], label=lab)
        a.set_title(name)
        a.set_xlabel("time from nearest R-peak (s)")
    ax[0].set_ylabel("mean power (sum over 8 leads)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_residual_phase.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, r in res.items():
        ax[0].plot(np.arange(1, len(r["explained_curve"]) + 1), r["explained_curve"], label=name)
        ax[1].plot(np.arange(1, r["rank"] + 1), r["lag1_corr"], "o-", label=name)
    ax[0].axhline(EV_TARGET, color="grey", lw=0.8)
    ax[0].set(xlabel="rank", ylabel="explained deviation variance")
    ax[1].set(xlabel="component", ylabel="lag-1 correlation across beats")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_beat_variation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    C.check_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(C.DATA / "X_train.npy", mmap_mode="r")
    y = np.load(C.DATA / "y_train.npy").astype(np.int64)
    fit = C.part_index("fit")
    rng = np.random.default_rng(SEED)
    res = {}
    for c in range(4):
        res[CLASS_ORDER[c]] = diagnose_class(X, y, fit, c, rng)
        r = res[CLASS_ORDER[c]]
        print(f"{CLASS_ORDER[c]}: K {r['rank']} | flatness residual {r['residual_flatness']:.2f} leftover "
              f"{r['leftover_flatness']:.2f} gp {r['gp_flatness']:.2f} | lag1 {np.round(r['lag1_corr'][:4], 2)} | "
              f"wander rms real {np.round(r['wander_real']['rms_quantiles'][2], 3)} gp "
              f"{np.round(r['wander_gp']['rms_quantiles'][2], 3)}", flush=True)
    res["targets"] = {name: {"rank": r["rank"], "coef_var": r["coef_var"], "lag1_corr": r["lag1_corr"],
                             "wander": r["wander_real"], "leftover_phase_curve": r["phase_curves"]["leftover"]}
                      for name, r in res.items()}
    (OUT / "diagnostics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    figures({k: v for k, v in res.items() if k != "targets"})
    C.check_frozen()
