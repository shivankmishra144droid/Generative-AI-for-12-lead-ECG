"""Phase C-E figures from saved JSON only (CPU): rhythm_sweep.png, training_curves.png, sampler_heatmap.png."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import CLASSES
from .v2_common import RESULTS


def sweep():
    r = json.loads((RESULTS/'rhythm_control.json').read_text())
    hrs = r['hrs']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, c in zip(axes, CLASSES):
        ax.plot(hrs, hrs, 'k:', lw=1, label='identity')
        for name, col, lab in (('cs', 'tab:orange', 'v2, no R-map'), ('ds', 'tab:blue', 'v2, R-map')):
            m = r['models'][name][c]
            ax.plot(hrs, [m['median_by_hr'][str(h)] for h in hrs], 'o-', color=col,
                    label=f"{lab} (slope {m['slope']:.2f}, R² {m['r2']:.2f})")
        ax.set_title(c); ax.set_xlabel('requested HR (bpm)'); ax.legend(fontsize=7)
    axes[0].set_ylabel('measured HR, median of 32 (bpm)')
    fig.suptitle('Rhythm control sweep (dev; slope/R² from all samples, points are medians; v1 not run)', fontsize=10)
    fig.tight_layout(); fig.savefig(RESULTS/'rhythm_sweep.png', dpi=130); plt.close(fig)


def curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, col in (('cs', 'tab:orange'), ('ds', 'tab:blue')):
        h = json.loads((RESULTS/f'diffusion_training_{name}.json').read_text())['history']
        st = [e['step'] for e in h]
        axes[0].plot(st, [e['dev_loss'] for e in h], color=col, label=f'{name} dev loss')
        axes[0].plot(st, [e['dev_loss_bins'][-1] for e in h], '--', color=col, label=f'{name} dev loss, highest-t bin')
    s = json.loads((RESULTS/'snapshot_curves.json').read_text())
    for name, col in (('cs', 'tab:orange'), ('ds', 'tab:blue')):
        if name in s:
            ks = sorted(s[name], key=int)
            axes[1].plot([int(k) for k in ks], [s[name][k]['full'] for k in ks], 'o-', color=col, label=f'{name} full judge')
            axes[1].plot([int(k) for k in ks], [s[name][k]['rhythm'] for k in ks], 's--', color=col, label=f'{name} rhythm judge')
    axes[0].set_xlabel('step'); axes[0].set_ylabel('v-loss'); axes[0].legend(fontsize=7); axes[0].set_ylim(.3, 1.)
    axes[1].set_xlabel('step'); axes[1].set_ylabel('judge AUC (dev, 63/class)'); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS/'training_curves.png', dpi=130); plt.close(fig)


def heatmap():
    e = json.loads((RESULTS/'sampler_sweep.json').read_text())['rows']
    cfgs, steps = [1, 2, 3], [50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, key, title in zip(axes, ('full_judge', 'mean_coverage_task'), ('full judge AUC (lower better)', 'emb_task coverage (higher better)')):
        m = np.array([[e[f'cfg{c}_s{s}'][key] for s in steps] for c in cfgs])
        im = ax.imshow(m, cmap='viridis_r' if key == 'full_judge' else 'viridis')
        for i in range(3):
            for j in range(2):
                ax.text(j, i, f'{m[i, j]:.3f}', ha='center', va='center', color='w', fontsize=9)
        ax.set_xticks(range(2), [f'DDIM {s}' for s in steps]); ax.set_yticks(range(3), [f'CFG {c}' for c in cfgs])
        ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax, fraction=.05)
    fig.suptitle('Phase E sampler grid (dev, cs, 63/class; chosen CFG 2 / DDIM 50)', fontsize=10)
    fig.tight_layout(); fig.savefig(RESULTS/'sampler_heatmap.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    sweep(); curves(); heatmap()
