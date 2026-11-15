"""Phase G: utility with the studies 2-3 recipe (exploratory, validation scored once per run).

Classifier trains on fit patients (+ synthetic), selects the checkpoint and fits decision offsets on dev, is scored
on validation. 6,250 steps, batch 64, class-weighted CE, deterministic CUDA (augment_experiment.train_steps, import only).
Fill target per minority class = 2 x fit count. *_plus_sr arms add synthetic SR = 20 % of real fit SR.
"""
import json
import subprocess
import sys
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from .common import ROOT, CLASSES, digest, load_split
from .v2_common import ART, RESULTS, AUDITS, SEED, training_data, register, write_json, gpu_lease, now
from .sample_v2 import Generator, fit_requests
from .thresholds_v2 import tune, predict as tpredict, logp_of

SEEDS = [9100, 9101, 9102, 9103, 9104]
POOLS = ART/'pools'


def needs(y, fit):
    return {c: int((y[fit] == c).sum()) for c in (1, 2, 3)}, int(round(.2*(y[fit] == 0).sum()))


def build_pools(y, fit):
    need, nsr = needs(y, fit)
    counts = [nsr, need[1], need[2], need[3]]
    sampling = json.loads((ART/'sampling.json').read_text())
    rc = json.loads((RESULTS/'rhythm_control.json').read_text())
    specs = {'latent_v2': ('cs', 'bank')}
    if rc.get('gate_D', {}).get('passed'):
        specs['latent_v2'] = ('ds', 'bank')
    if (ART/'diffusion_ds'/'last.pt').exists():
        specs['latent_v2_rhythm_model'] = ('ds', 'model')
    for tag, (name, src) in specs.items():
        path = POOLS/f'G_{tag}.npz'
        if path.exists():
            continue
        if src == 'model':
            from .phase_f_v2 import rhythm_model_requests
            yy, v, pk = rhythm_model_requests(counts, 9810)
        else:
            yy, v, pk, _ = fit_requests(counts, 9800)
        cfg = sampling if sampling['main'] == name else dict(guidance=2., steps=50, eta=0.)
        with gpu_lease(f'phaseG_{tag}') as device:
            X = Generator(name, device)(yy, v, pk, cfg['guidance'], cfg['steps'], cfg['eta'], seed=9820)
        np.savez_compressed(path, X=X, y=yy)
    sr = POOLS/'G_hybrid_sr.npz'
    if not sr.exists():
        with gpu_lease('phaseG_hybrid_sr'):
            pass                                    # lease check only: the frozen script opens its own CUDA context
        subprocess.run([sys.executable, '-m', 'studies.beat_generator._beat_gen_pool',
                        str(ROOT/'generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt'),
                        f'{nsr},0,0,0', str(sr), '--weights', 'ema', '--seed', '9800'], cwd=ROOT, check=True)
    return specs


def arm_data(arm, x, y, fit, specs):
    need, nsr = needs(y, fit)
    Xr, yr = np.asarray(x[fit], np.float32), y[fit]
    rng = np.random.default_rng(SEED + 60)
    if arm == 'real':
        return Xr, yr, 0
    if arm == 'oversample':
        dup = np.concatenate([np.resize(np.flatnonzero(yr == c), need[c]) for c in (1, 2, 3)])
        return np.concatenate([Xr, Xr[dup]]), np.concatenate([yr, yr[dup]]), 0
    base = arm.replace('_plus_sr', '')
    if base == 'hybrid':
        pool = np.load(ROOT/'generator_comparison/pools2/pool_1.npz')
        srp = np.load(POOLS/'G_hybrid_sr.npz')
    else:
        pool = np.load(POOLS/f'G_{base}.npz')
        srp = pool
    px, py = pool['X'], pool['y']
    idx = np.concatenate([rng.choice(np.flatnonzero(py == c), need[c], replace=False) for c in (1, 2, 3)])
    ax, ay = [px[idx]], [py[idx]]
    if arm.endswith('_plus_sr'):
        s = rng.choice(np.flatnonzero(srp['y'] == 0), nsr, replace=False)
        ax.append(srp['X'][s]); ay.append(srp['y'][s])
    ax, ay = np.concatenate(ax).astype(np.float32), np.concatenate(ay).astype(np.int64)
    return np.concatenate([Xr, ax]), np.concatenate([yr, ay]), len(ay)


def evaluate(yv, pv, yd, pd):
    out = dict(untuned_macro_f1=float(f1_score(yv, pv.argmax(1), average='macro')),
               auprc={CLASSES[c]: float(average_precision_score(yv == c, pv[:, c])) for c in range(4)})
    out['macro_auprc'] = float(np.mean(list(out['auprc'].values())))
    b, _ = tune(logp_of(pd), yd)
    pred = tpredict(logp_of(pv), b)
    out['offsets'] = b.tolist()
    out['tuned_macro_f1'] = float(f1_score(yv, pred, average='macro'))
    for name, pr in (('untuned', pv.argmax(1)), ('tuned', pred)):
        out[f'{name}_precision'] = dict(zip(CLASSES, map(float, precision_score(yv, pr, average=None, labels=range(4), zero_division=0))))
        out[f'{name}_recall'] = dict(zip(CLASSES, map(float, recall_score(yv, pr, average=None, labels=range(4), zero_division=0))))
    return out


def pools():
    """Separate process: pool generation must not run under deterministic-algorithm mode."""
    x, y, patient, fit, dev = training_data()
    build_pools(y, fit)


def run():
    from generator_comparison.augment_experiment import set_deterministic
    set_deterministic()
    x, y, patient, fit, dev = training_data()
    specs = build_pools(y, fit)          # no-op when `pools` already ran (all files exist)
    arms = ['real', 'oversample', 'hybrid', 'latent_v2', 'hybrid_plus_sr', 'latent_v2_plus_sr']
    if 'latent_v2_rhythm_model' in specs:
        arms.insert(4, 'latent_v2_rhythm_model')
    register('phaseG', dict(arms=arms, seeds=SEEDS, steps=6250, batch=64, selection='dev plain-argmax macro-F1',
                            offsets='tune() on dev softmax', specs={k: list(v) for k, v in specs.items()},
                            hybrid_minority='pools2/pool_1.npz', hybrid_sr='G_hybrid_sr.npz (frozen _beat_gen_pool.py, seed 9800)'))
    from generator_comparison.augment_experiment import train_steps, predict
    Xd, yd = np.asarray(x[dev], np.float32), y[dev]
    xv, yv = load_split('val')
    Xv = np.asarray(xv, np.float32)
    out_dir = ART/'G'
    out_dir.mkdir(exist_ok=True)
    resfile = RESULTS/'utility_G.json'
    results = json.loads(resfile.read_text()) if resfile.exists() else {}
    with gpu_lease('phaseG_train') as device:
        for arm in arms:
            Xa, ya, nsyn = arm_data(arm, x, y, fit, specs)
            for s in SEEDS:
                key = f'{arm}/seed{s}'
                if key in results:
                    continue
                net, info = train_steps(Xa, ya, Xd, yd, s, device, 6250, 250)
                pd_, pv = predict(net, Xd, device), predict(net, Xv, device)
                np.save(out_dir/f'dev_prob_{arm}_{s}.npy', pd_)
                np.save(out_dir/f'val_prob_{arm}_{s}.npy', pv)
                results[key] = dict(evaluate(yv, pv, yd, pd_), best_step=info['best_step'], n_train=len(ya),
                                    n_synthetic=nsyn, train_counts=np.bincount(ya, minlength=4).tolist())
                write_json(resfile, results)
                r = results[key]
                print(key, round(r['untuned_macro_f1'], 4), round(r['macro_auprc'], 4), round(r['tuned_macro_f1'], 4), flush=True)
    summary(results, arms, yv)


def summary(results, arms, yv):
    vp = np.load(ART.parent/'val_conditions.npz')['patient']
    pats = np.unique(vp)
    rng = np.random.default_rng(SEED + 70)
    metrics = ('untuned_macro_f1', 'macro_auprc', 'tuned_macro_f1')
    probs = {a: [np.load(ART/'G'/f'val_prob_{a}_{s}.npy') for s in SEEDS] for a in arms}
    offs = {a: [np.array(results[f'{a}/seed{s}']['offsets']) for s in SEEDS] for a in arms}

    def score(a, i, rows):
        p = probs[a][i][rows]
        yy = yv[rows]
        return [f1_score(yy, p.argmax(1), average='macro'),
                np.mean([average_precision_score(yy == c, p[:, c]) for c in range(4)]),
                f1_score(yy, tpredict(logp_of(p), offs[a][i]), average='macro')]
    by_pat = {p: np.flatnonzero(vp == p) for p in pats}
    table = {}
    for a in arms:
        table[a] = {m: float(np.mean([results[f'{a}/seed{s}'][m] for s in SEEDS])) for m in metrics}
        table[a].update({f'{m}_sd': float(np.std([results[f'{a}/seed{s}'][m] for s in SEEDS], ddof=1)) for m in metrics})
    diffs = {}
    boot = []
    for _ in range(300):                            # 2-level bootstrap: val patients, then seeds
        rows = np.concatenate([by_pat[p] for p in rng.choice(pats, len(pats))])
        si = rng.integers(0, len(SEEDS), len(SEEDS))
        boot.append({a: np.mean([score(a, i, rows) for i in si], 0) for a in arms})
    for a in arms:
        if a == 'real':
            continue
        point = {m: table[a][m] - table['real'][m] for m in metrics}
        draws = np.array([b[a] - b['real'] for b in boot])
        diffs[f'{a} - real'] = {m: dict(diff=point[m], ci95=np.quantile(draws[:, j], [.025, .975]).tolist())
                                for j, m in enumerate(metrics)}
    for a, b2 in (('hybrid_plus_sr', 'hybrid'), ('latent_v2_plus_sr', 'latent_v2'), ('latent_v2', 'hybrid')):
        draws = np.array([b[a] - b[b2] for b in boot])
        diffs[f'{a} - {b2}'] = {m: dict(diff=table[a][m] - table[b2][m], ci95=np.quantile(draws[:, j], [.025, .975]).tolist())
                                for j, m in enumerate(metrics)}
    write_json(RESULTS/'utility_G_summary.json', dict(created=now(), table=table, differences=diffs,
                                                     note='exploratory; 2-level bootstrap (300 reps) over val patients and seeds'))
    lines = ['| arm | untuned macro-F1 | macro AUPRC | tuned macro-F1 |', '|---|---|---|---|']
    for a in arms:
        t = table[a]
        lines.append(f"| {a} | {t['untuned_macro_f1']:.4f} ± {t['untuned_macro_f1_sd']:.4f} | {t['macro_auprc']:.4f} ± {t['macro_auprc_sd']:.4f} | {t['tuned_macro_f1']:.4f} ± {t['tuned_macro_f1_sd']:.4f} |")
    lines += ['', '| contrast | untuned F1 diff [CI] | AUPRC diff [CI] | tuned F1 diff [CI] |', '|---|---|---|---|']
    for k, d in diffs.items():
        lines.append(f'| {k} | ' + ' | '.join(f"{d[m]['diff']:+.4f} [{d[m]['ci95'][0]:+.4f}, {d[m]['ci95'][1]:+.4f}]" for m in metrics) + ' |')
    (RESULTS/'utility_table.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    dict(pools=pools, train=run)[sys.argv[1] if len(sys.argv) > 1 else 'train']()
