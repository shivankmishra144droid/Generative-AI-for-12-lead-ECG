"""Figures for report.tex that are not already produced by the study scripts.

Numbers are copied from the result files named next to each one, so the figures can be rebuilt without the large
experiment artifacts. fig_examples needs the sample pools, which are not part of the repository; pass the research
repo path with --pools to rebuild it.
py -3.14 report/make_figures.py [--pools PATH_TO_RESEARCH_REPO]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / 'figures'
CLASSES = ('SR', 'AFIB', 'STACH', 'SBRAD')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def realism_ladder():
    # latent_diffusion/results/v2/F_*.json and F2_*.json, feature_judges.full (value, sd); validation split
    rows = [('real vs real (floor)', 0.454, 0.040), ('AE reconstruction (ceiling)', 0.627, 0.018),
            ('latent diffusion C-M (5.9M)', 0.796, 0.016), ('latent diffusion C (0.8M)', 0.919, 0.007),
            ('beat stitching (hybrid)', 0.965, 0.005), ('R-peak-map model D', 0.969, 0.004),
            ('latent diffusion v1', 1.000, 0.0), ('VAE', 1.000, 0.0), ('ECGSYN', 1.000, 0.0)]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    names, vals, sds = zip(*rows)
    colors = ['0.6', '0.6', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:blue', '0.75', '0.75', '0.75']
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, xerr=np.array(sds) * 2, color=colors, height=0.65, capsize=2)
    for yi, v, s in zip(y, vals, sds):
        ax.text(v + 2 * s + 0.01, yi, f'{v:.3f}', va='center', fontsize=8)
    ax.set_yticks(y, names)
    ax.axvline(0.5, color='k', lw=0.6, ls=':')
    ax.set_xlim(0.4, 1.1)
    ax.set_xlabel('feature-judge AUC on validation (mean ± 2 SD; 0.5 = indistinguishable)')
    fig.tight_layout(); fig.savefig(OUT / 'fig_realism_ladder.png', dpi=200); plt.close(fig)


def cm_checkpoints():
    # latent_diffusion/results/v2/ext120k/cm_checkpoints.json (dev, 63/class); cs_step120000 = 0.856
    steps = [25, 50, 54, 75, 100]
    judge = [0.864, 0.781, 0.756, 0.754, 0.719]
    loss = [0.4193, 0.4340, 0.4370, 0.4532, 0.4733]      # results/v2/diffusion_training_cm.json (dev v-loss)
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.plot(steps, judge, 'o-', color='tab:blue', label='C-M dev judge AUC')
    ax.axhline(0.856, color='tab:green', ls='--', lw=1, label='C at 120k steps (dev judge)')
    ax.set_xlabel('training step (×1000)'); ax.set_ylabel('dev judge AUC (lower = better)')
    ax2 = ax.twinx()
    ax2.plot(steps, loss, 's:', color='tab:red', label='C-M dev diffusion loss')
    ax2.set_ylabel('dev loss', color='tab:red')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')
    fig.tight_layout(); fig.savefig(OUT / 'fig_cm_checkpoints.png', dpi=200); plt.close(fig)


def study2():
    # generator_comparison/experiment/study2/study2_result.json; report/REPORT.md section 8.3
    conds = ['real', 'oversample', 'random\nsynthetic', 'filtered\nsynthetic']
    f1 = [0.8030, 0.8199, 0.8393, 0.8352]
    auprc_sbrad = [0.761, 0.750, 0.756, 0.758]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4))
    axes[0].bar(conds, f1, color=['0.6', '0.6', 'tab:orange', 'tab:orange'])
    axes[0].set_ylim(0.78, 0.85); axes[0].set_title('test macro-F1 (mean of 40 runs)')
    axes[1].bar(conds, auprc_sbrad, color=['0.6', '0.6', 'tab:orange', 'tab:orange'])
    axes[1].set_ylim(0.70, 0.80); axes[1].set_title('SBRAD AUPRC (threshold-free)')
    for ax, vals in zip(axes, (f1, auprc_sbrad)):
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig_study2.png', dpi=200); plt.close(fig)


def threshold():
    # report/THRESHOLD_EXPLORE.md table 1 (held-out validation halves, 15 runs per cell)
    levels = ['10 %', '25 %', '50 %', '100 %']
    data = {'real': ([0.7356, 0.7905, 0.7891, 0.7837], [0.7291, 0.8046, 0.8228, 0.8399]),
            'oversample': ([0.7115, 0.8040, 0.8159, 0.7996], [0.7126, 0.7997, 0.8215, 0.8384]),
            'synthetic': ([0.7346, 0.8169, 0.8325, 0.8188], [0.7370, 0.8126, 0.8334, 0.8452])}
    colors = {'real': '0.45', 'oversample': 'tab:green', 'synthetic': 'tab:orange'}
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    x = np.arange(4)
    for ax, k, title in zip(axes, (0, 1), ('untuned (argmax)', 'tuned per-class thresholds')):
        for name, (u, t) in data.items():
            ax.plot(x, (u, t)[k], 'o-', color=colors[name], label=name)
        ax.set_xticks(x, levels); ax.set_title(title); ax.set_xlabel('real minority data kept')
    axes[0].set_ylabel('validation macro-F1'); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig_threshold.png', dpi=200); plt.close(fig)


def examples(pools):
    """Lead II, one ECG per class: real validation vs beat stitching vs C-M (validation second-look pool)."""
    real = np.load(pools / 'latent_diffusion/artifacts/v2/metric_cache/val_real.npz')
    hyb = np.load(pools / 'generator_comparison/tstr/pool_beatgen.npz')
    cm = np.load(pools / 'latent_diffusion/artifacts/v2/pools/F2_cm100k.npz')
    srcs = [('real (validation)', real['X'], real['y']), ('beat stitching', hyb['X'], hyb['y']),
            ('latent diffusion C-M', cm['X'], cm['y'])]
    rng = np.random.default_rng(3)
    t = np.arange(1000) / 100
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 5.6), sharex=True)
    for j, (name, X, y) in enumerate(srcs):
        for c in range(4):
            i = rng.choice(np.flatnonzero(y == c))
            ax = axes[c, j]
            ax.plot(t, X[i, :, 1], lw=0.6, color='k')
            ax.set_yticks([])
            if c == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(CLASSES[c], fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('time (s)')
    fig.tight_layout(); fig.savefig(OUT / 'fig_examples.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pools', type=Path, default=None)
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    realism_ladder(); cm_checkpoints(); study2(); threshold()
    if a.pools:
        examples(a.pools)
    print('figures written to', OUT)
