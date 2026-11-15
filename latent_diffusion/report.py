"""Generate the written report and durable result files from measured artifacts."""
from __future__ import annotations

import csv
import html
import importlib.metadata
import json
import shutil

import numpy as np

from .common import ART, CLASSES, HOME, ROOT, write_json


def read(name):
    path = ART / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def number(x, digits=3):
    return f'{x:.{digits}f}' if x is not None else 'unavailable'


def figures(scorecard):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = HOME / 'results'; folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    sources = [name for name, r in scorecard['sources'].items() if r.get('status') == 'scored']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    markers = ['o', 's', '^', 'D', 'v', 'x']
    for marker, name in zip(markers, sources):
        by = scorecard['results'][name]['by_class']
        axes[0].plot(range(4), [by.get(c, {}).get('prdc', {}).get('precision', np.nan) for c in CLASSES],
                     marker=marker, label=name)
        axes[1].plot(range(4), [by.get(c, {}).get('prdc', {}).get('coverage', np.nan) for c in CLASSES],
                     marker=marker, label=name)
    for ax, metric in zip(axes, ['Precision', 'Coverage']):
        ax.set(xticks=range(4), xticklabels=CLASSES, xlabel='Conditioned rhythm class',
               ylabel=f'{metric} (fraction)', ylim=(-.03, 1.03), title=f'Frozen ECG-encoder {metric.lower()}')
        ax.grid(axis='y', alpha=.2)
    axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02,1.))
    fig.suptitle('Study 9 · class-matched exploratory benchmark · k=5')
    fig.savefig(folder / 'embedding_scorecard.png', dpi=160); plt.close(fig)
    panels = []
    x = np.load(ROOT / 'processed_data/X_val.npy', mmap_mode='r')
    y = np.load(ROOT / 'processed_data/y_val.npy')
    panels.append(('Real validation', np.asarray(x[np.flatnonzero(y == 0)[0], :, 1])))
    for name, path in [('Hybrid', ROOT / 'generator_comparison/tstr/pool_beatgen.npz'),
                       ('Conditional VAE', ART / 'pools/vae.npz'), ('Latent diffusion', ART / 'pools/latent_diffusion.npz'),
                       ('ECGSYN', ART / 'pools/ecgsyn.npz')]:
        if path.exists():
            d = np.load(path)
            row = np.flatnonzero(d['y'] == 0)[0]
            panels.append((name, d['X'][row, :, 1]))
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 1.5*len(panels)), sharex=True, constrained_layout=True)
    for ax, (name, data) in zip(np.atleast_1d(axes), panels):
        ax.plot(np.arange(1000)/100, data, color='#305d79', lw=.7)
        ax.set(ylabel='Lead II (z)', title=name); ax.grid(alpha=.15)
    np.atleast_1d(axes)[-1].set_xlabel('Time (seconds)')
    fig.suptitle('First SR example per source · fixed selection, no visual cherry-picking')
    fig.savefig(folder / 'sr_waveforms.png', dpi=160); plt.close(fig)


def run():
    folder = HOME / 'results'; folder.mkdir(exist_ok=True)
    score = read('scorecard.json')
    if score is None:
        raise RuntimeError('Run the benchmark before writing the results report')
    for name in ['scorecard.json', 'scorecard_summary.csv', 'utility.json', 'control_sweep.json', 'reconstruction.json',
                 'data_contract.json', 'published_access.json', 'gpu_profiles.json', 'ae/training.json', 'vae/training.json', 'diffusion/training.json']:
        source = ART / name
        if source.exists():
            shutil.copy2(source, folder / name.replace('/', '_'))
    for source in (ART / 'pools').glob('*.json'):
        shutil.copy2(source, folder / ('pool_'+source.name))
    # Derive the presentation table from the authoritative JSON, including older
    # scorecards produced before optional clinical summary columns were added.
    summary = []
    for name, result in score['results'].items():
        for rhythm, values in result['by_class'].items():
            p = values['prdc']
            summary.append(dict(generator=name, rhythm=rhythm, n=p.get('n_synthetic'),
                precision=p.get('precision'), recall=p.get('recall'), density=p.get('density'), coverage=p.get('coverage'),
                hr_wasserstein_bpm=values['features']['hr'].get('wasserstein'),
                sdnn_wasserstein_ms=values['features']['sdnn_ms'].get('wasserstein'),
                qrs_width_wasserstein_ms=values['features']['qrs_width_ms'].get('wasserstein'),
                peak_detectability_fraction=result['peak_detectability']['synthetic_fraction_ge3_peaks'],
                qt_proxy_median_ms=result.get('qt_synthetic',{}).get('qt_ms_median'),
                judge_auc=result['feature_judge']['auc_mean'],
                membership_auc=values.get('membership',{}).get('auc'),
                waveform_nn_ratio=values.get('waveform_privacy',{}).get('ratio_of_medians')))
    if summary:
        with (folder / 'scorecard_summary.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    versions = {p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','pandas','scikit-learn','pytest','matplotlib']}
    versions.update(neurokit2='0.2.13 (isolated optional dependency)', PyWavelets='1.10.0 (isolated optional dependency)')
    write_json(folder / 'environment.json', versions)
    figures(score)
    data, ae, vae, diff = read('data_contract.json'), read('ae/training.json'), read('vae/training.json'), read('diffusion/training.json')
    rec, utility, sweep = read('reconstruction.json'), read('utility.json'), read('control_sweep.json')
    final = json.loads((folder / 'final_audit.json').read_text()) if (folder / 'final_audit.json').exists() else None
    lines = ['# latent_diffusion — whole-ECG latent generation', '',
             'Experiments: 30 October 2026. Consolidated into one folder on 31 October 2026. Exploratory engineering and benchmark study. Prior studies and their artifacts are read-only.', '',
             'All Study 9 source code, checkpoints, results and audits are under `latent_diffusion/`. Models are in `latent/`; metrics are in `gen_eval.py`. Existing repository data, baseline weights and the SSSD backbone remain read-only dependencies. The original audit is preserved under `audits/history/`; the final audit validates the consolidated layout.', '',
             '## Outcome', '',
             'Implemented and ran a convolutional whole-ECG autoencoder, a conditional VAE baseline, and a latent SSSD diffusion model. '
             'One model generates the full ten-second ECG; class, measured HR, age and sex enter the model together. '
             'The evaluation now separates fidelity, diversity, memorisation, clinical plausibility and downstream utility. '
             'This is a working GPU-feasible research pipeline; it is not a validated clinical generator or proof that latent diffusion beats the hybrid.', '',
             'The pilot does **not** solve whole-ECG realism: the clinical-feature judge still separates latent diffusion almost perfectly, '
             'and the matched-seed control sweep is poorly calibrated (SBRAD requested at 45 bpm measures about 35 bpm). '
             'This negative result is kept visible. More training, greater backbone capacity and balanced minority conditioning are follow-up experiments, '
             'not results inferred from accepting a condition vector.', '',
             'The published SSSD dataset could not be downloaded: its endpoints returned HTTP 403 or an empty HTTP 202 response. '
             'The benchmark records that row as unavailable. No published score is invented. Its importer and access log are included.', '',
             '## Phase 1 — isolation and data', '',
             f"Read {data['train']['n']:,} training and {data['val']['n']:,} validation ECGs at 100 Hz, shape (1000,12). "
             'A seeded patient-grouped 85/15 partition inside training separates generator fit from development. '
             'No Study 9 pipeline loads the test split. Metadata label alignment, unique ECG IDs, finite signals and patient disjointness are checked.', '',
             'HR is measured using the frozen project detector, so it is a noisy proxy rather than clinical ground truth. '
             'Age outside 0–100 (including the de-identification sentinel 300), invalid HR, and unknown sex are masked. '
             'PTB-XL sex coding is retained as 0/1. Numeric scales are fixed physical scales, not fitted on evaluation data. '
             'All conditions for random generation come from internal-fit patients; explicit user overrides are supported.', '',
             f"Train HR medians, SR/AFIB/STACH/SBRAD: {', '.join(number(v,1) for v in data['train']['measured_hr_median_by_class'])} bpm. "
             f"HR/age/sex availability: {', '.join(number(v*100,2) for v in data['train']['condition_available'])}%. Audit: `audits/phase1_data.json`.", '',
             '## Phase 2 — whole-ECG compression and VAE', '',
             'The encoder uses three stride-2 convolution blocks: 12×1000 → 16×125 (6× fewer scalar values). '
             'The decoder uses three transposed-convolution blocks. Reconstruction MSE plus 0.1× first-difference L1 preserves sharp morphology; '
             'a small KL penalty regularizes the latent posterior. The decoder also receives the conditions. '
             'Diffusion uses posterior means; the separate VAE baseline samples N(0,I), with stronger KL weight 0.01 instead of 0.0001.', '',
             f"Both models trained 15 epochs on internal-fit patients. AE best development MSE: {min(r['dev_mse'] for r in ae['history']):.5f}; "
             f"VAE best development MSE: {min(r['dev_mse'] for r in vae['history']):.5f}. "
             'These are standardized-coordinate reconstruction errors, not generative fidelity scores.', '',
             f"Clinical compression audit, {rec['n']} balanced development ECGs: projected MSE {rec['projected_mse']:.5f}, "
             f"missed source R-peaks {rec['rpeak']['missed_fraction']*100:.2f}%, extra peaks per source beat {rec['rpeak']['extra_per_source_beat']*100:.2f}%, "
             '50 ms matching tolerance. This uses aligned reconstructions, where a peak correspondence exists. '
             'The same missed-beat calculation is deliberately not applied to unpaired generated ECGs.', '',
             'The decoder learns twelve leads. Sampling then enforces the four dependent limb leads in physical mV before re-normalizing. '
             'Raw neural output is retained alongside projected output so this deterministic physiological constraint is visible. '
             'Existing hybrid and converted published pools use the same physical identities.', '',
             '## Phase 3 — standard latent diffusion', '',
             'The new wrapper reuses the existing SSSD residual/S4 code without changing it. The pilot uses 4 residual layers, width 32, S4 state 16, '
             'latent length 125, batch 8. The model is configurable; this is not the published 36×256 model. '
             'Per-channel latent means and scales are fitted on internal-fit encodings only.', '',
             'Recipe: cosine ᾱ schedule, T=1000; v=√ᾱ·ε−√(1−ᾱ)·z; joint class/numeric condition dropout 0.15; '
             'AdamW 2e-4; gradient clipping 1; EMA decay 0.995; deterministic DDIM η=0 with 50 sample steps; CFG scale 2. '
             'Null dropout removes all denoiser conditions jointly. CFG acts on the latent prior; the decoder remains conditional.', '',
             f"Completed {diff['history'][-1]['step']:,} diffusion optimizer updates. Fixed-noise development loss: "
             f"{diff['history'][0]['dev_loss']:.4f} → {diff['history'][-1]['dev_loss']:.4f}. "
             f"Terminal ᾱ={diff['terminal_alpha_bar']:.3g}, corresponding to signal amplitude √ᾱ≈{np.sqrt(diff['terminal_alpha_bar']):.3g}. "
             'For comparison, the old linear 200-step schedule retains about 0.132 signal power, or 0.364 signal amplitude. '
             'Power and amplitude must not be confused.', '',
             f"Measured GPU peak: {diff.get('peak_allocated_mib',0):.1f} MiB allocated, {diff.get('peak_reserved_mib',0):.1f} MiB reserved by PyTorch. "
             'The device is an RTX 3050 Laptop with 4 GB; CUDA context, driver and desktop allocations are additional. '
             'This establishes fit for this measured configuration only. The fallback S4 kernel is used. '
             'Training samples follow the original imbalanced rhythm distribution; minority conditioning is therefore a remaining limitation. '
             'One training seed and a fixed training budget do not establish convergence.', '',
             '## Phase 4 — common scorecard', '',
             'Fidelity: precision and density in a frozen real-only CNN-LSTM encoder (64-dimensional pre-head hidden representation); '
             'per-class Wasserstein distance for all 25 existing clinical features. Diversity: recall, coverage, feature SD and IQR. '
             'The embedding scale is fit on matched real-fit references only. The encoder is task-specific and was historically selected on validation, '
             'so these are exploratory task-dependent embeddings, not an independent universal ECG representation.', '',
             'Privacy: synthetic-to-training and real-validation-to-training nearest-neighbour distances, in embeddings and full waveform RMS; '
             'copy fraction; class-matched proximity membership AUC with patient-cluster bootstrap intervals. '
             'Distances use a matched reference subset, not an exhaustive search over all training ECGs. '
             'The release-distance attack is a limited black-box attack; a near-0.5 AUC does not guarantee privacy.', '',
             'Clinical plausibility: detected-beat availability, HR/SDNN/RMSSD/RR variability, QRS-width proxy and morphology features, with invalid-feature counts. '
             'Optional NeuroKit DWT R-onset–T-offset QT proxies are reported with valid interval counts and failures, at 10 ms sampling resolution. '
             'No QT or peak result is described as expert-adjudicated clinical truth. The feature judge uses three patient-grouped splits; '
             'it is a diagnostic alongside the generative metrics, not their replacement.', '',
             'Controls passed: exact-copy detection, collapsed-generator coverage reduction, far-outlier precision reduction, '
             'missing-feature reporting, frozen encoder loading, v inverse, null conditioning, finite backward, checkpoint round trip and deterministic DDIM.', '',
             '## Phase 5 — measured benchmark', '',
             '[Complete comparison table (CSV)](results/scorecard_summary.csv) · [All class-wise metrics and 25-feature distributions](results/scorecard.json). '
             'Rows below retain class support rather than averaging incompatible rhythm sets. The real-control row uses internal development patients. '
             'Each generator is matched to the same validation and fit rows within a rhythm and count. PRDC uses k=5. '
             'Generator training sizes and compute budgets differ, so this is an engineering comparison, not a controlled architecture ablation.', '']
    for name, cfg in score['sources'].items():
        if cfg.get('status') != 'scored':
            lines.extend([f"- **{name}:** {cfg.get('status', 'unavailable')}."])
            continue
        result = score['results'][name]
        lines.append(f"- **{name}:** n={cfg['n']}, feature-judge AUC {number(result['feature_judge']['auc_mean'])}. "
                     f"At least three detected peaks: {100*result['peak_detectability']['synthetic_fraction_ge3_peaks']:.1f}% of ECGs.")
        for label, r in result['by_class'].items():
            p = r['prdc']
            lines.append(f"  - {label}: precision {number(p.get('precision'))}, recall {number(p.get('recall'))}, "
                         f"density {number(p.get('density'))}, coverage {number(p.get('coverage'))}; "
                         f"HR Wasserstein {number(r['features']['hr'].get('wasserstein'),1)} bpm.")
        if 'qt_synthetic' in result:
            qt = result['qt_synthetic']
            lines.append(f"  - QT proxy: {number(qt.get('qt_ms_median'),1)} ms; "
                         f"{qt.get('valid_intervals',0)} valid intervals in {qt.get('recordings_with_valid_intervals',0)}/{qt.get('n_recordings',cfg['n'])} recordings; "
                         f"{qt.get('failures',0)} processing failures. The median covers valid delineations only.")
    lines.extend(['', 'ECGSYN is NeuroKit2’s published multilead ECGSYN implementation. It supports SR/STACH/SBRAD here and does not simulate AFIB. '
                  'A local adapter selects NeuroKit’s supported NumPy RR interpolation, guards positive RR values and leaves package files unchanged. '
                  'The default spline path produced a zero RR endpoint and an infinite loop for seed 9161 at 105.15464 bpm; the exact case is regression-tested. '
                  'The ECGSYN equations and integration are unchanged. Its simulator amplitudes are treated as mV, then normalized by the existing training statistics and limb-projected; '
                  'there is no retrospective amplitude fitting to improve its score. Age and sex are not ECGSYN controls. '
                  'Published SSSD is excluded from utility and membership claims because its fold-trained model overlaps this repository’s random splits.', '',
                  '![Class-wise embedding precision and coverage](results/embedding_scorecard.png)', '',
                  '![Fixed-selection SR waveform examples](results/sr_waveforms.png)', '',
                  '## Conditional controllability test', ''])
    profiles = read('gpu_profiles.json')
    if profiles and profiles.get('rows'):
        profile = profiles['rows'][-1]
        lines.extend([f"A larger 8-layer, 64-channel latent backbone also passed an actual forward/backward/Adam-step capacity audit: "
                      f"{profile['peak_allocated_mib']:.1f} MiB allocated, {profile['peak_reserved_mib']:.1f} MiB reserved. "
                      'This capacity profile excludes AE/EMA and does not constitute training or a quality result. See `results/gpu_profiles.json`.', ''])
    if sweep:
        for label in (0, 2, 3):
            rows = [r for r in sweep['rows'] if r['label'] == label]
            lines.append(f"- {CLASSES[label]}, requested → measured median bpm: "+'; '.join(f"{r['requested_hr']:.0f} → {number(r['measured_median'],1)}" for r in rows)+'.')
        lines.extend(['', 'Each HR setting uses matched latent seeds, age 60 and source sex code 0. Conflicting label/HR pairs are deliberate stress tests. '
                      'These numbers measure achieved control; merely accepting an HR input does not prove the model obeys it. '
                      'Age/sex conditioning is implemented but demographic fidelity and fairness are not established by this pilot.'])
    else:
        lines.append('Control sweep unavailable; no achieved-control claim is made.')
    lines.extend(['', 'Example explicit request:', '', '```powershell',
                  'py -3.14 -m latent_diffusion.generate latent_diffusion --label SBRAD --hr 45 --age 60 --sex 0 --n 16',
                  '```', '', 'This writes a separate custom pool and preserves the benchmark pool.', '',
                  '## Utility: fresh TSTR/TRTR and augmentation', ''])
    if utility:
        lines.append(f"Three fresh classifier seeds, {utility['protocol']['updates']} optimizer updates each, batch 64, identical budget across conditions. "
                     f"{utility['protocol']['n_per_class']} real/synthetic ECGs per class; all real validation patients are evaluated. "
                     'No validation checkpoint selection. Augmentation uses balanced small-data training with 50:50 real/synthetic sampling; '
                     'TRTR is the matched balanced resampling control. This does not reproduce the larger historical scarcity experiments.')
        lines.append('')
        for name, r in utility['results'].items():
            lines.append(f"- {name}: validation macro-F1 {r['macro_f1_mean']:.3f} ± {number(r['macro_f1_std'])} (SD across seeds).")
        lines.extend(['', '[Full utility runs and exclusions](results/utility.json). No conclusion about augmentation benefit should rest on a single small-data pilot.'])
    else:
        lines.append('Utility experiment unavailable; no utility numbers are claimed.')
    lines.extend(['', '## Reframing the earlier generative findings', '',
                  'These are prior Study 4 findings, not newly measured Study 9 results:', '',
                  '1. **Time structure explains a failure of assembled ECGs.** Real leftover activity is beat-locked and non-Gaussian; '
                  'matching only spectrum, amplitude and simple envelopes did not remove detectability. This is a project-specific mechanistic finding; '
                  'a “first in the literature” claim has not been established. Source: `generator_comparison/study4/NOTES_phase2_3b.md`.',
                  '2. **An interpretable clinical realism judge.** The prior 25-feature judge achieved AUC 0.973, with real controls 0.509 and 0.512. '
                  'This names the giveaways in clinical terms. Source: `generator_comparison/study4/NOTES_features.md`.',
                  '3. **Noise can corrupt R-peak-based measurements.** GP residual replacement missed 11.2% of beats and added 20.1% in one aligned experiment; '
                  '49% of ECGs had an error. The 49% figure is an ECG-level error fraction, not a 49% missed-beat rate. '
                  'Other variants/detectors have their own rates. This supports caution for HRV-based use, without establishing clinical harm.',
                  '4. **Noise variants did not materially change macro utility in the tested setting.** The three tested variants were all unrealistic; '
                  'the study did not vary realism successfully and did not prove that realism is irrelevant. '
                  'Source: `generator_comparison/study4/NOTES_usefulness.md`.', '',
                  '## Phase 6 — final audit and limits', ''])
    if final:
        lines.append(f"Final audit: {'PASS' if final['checks_pass'] else 'FAIL / INCOMPLETE'}. "
                     f"{final['baseline_files_checked']} pre-existing source/metadata/label files checked; "
                     f"{len(final['changed_existing_files'])} changed. Test outputs and exact commands: `results/final_audit.json`. "
                     'Known warnings originate from the unchanged legacy S4 logger and PyTorch weight normalization API.')
    lines.extend(['', 'Remaining limits: one generator seed; short training budgets; minority underrepresentation; task-specific encoder; small per-class scorecard samples; '
                  'heuristic clinical measurements at 100 Hz; a limited membership attack; historically reused validation; unmatched generator training budgets; '
                  'unavailable published dataset; no full 36×256 model or fine-tuning of published weights. '
                  'The optional larger-GPU alternatives remain alternatives, not completed experiments. No existing study was retrained or overwritten.', '',
                  '## Reproduction and file map', '',
                  '- `PLAN.md`: phase plan written before training.',
                  '- `latent_diffusion/latent/vae.py`, `latent_diffusion.py`: new model implementations.',
                  '- `latent_diffusion/gen_eval.py`: new reusable generative metrics.',
                  '- `latent_diffusion/data.py`, `train.py`, `generate.py`, `benchmark.py`, `utility.py`: isolated pipeline.',
                  '- `audits/`: audit evidence after each phase; `results/`: durable JSON/CSV/figures.',
                  '- `artifacts/`: local checkpoints, pools and caches (git-ignored); `published.py`: guarded external-array importer.', '',
                  '```powershell', 'py -3.14 -m latent_diffusion.run_all --resume', '```', '',
                  'Uses the CUDA-enabled Python 3.14 environment. Plain `python` on this machine is a different CPU-only Python 3.11. '
                  'Optional NeuroKit/PyWavelets wheels are installed into Study 9 only. Exact package versions: `results/environment.json`. '
                  'Diffusion checkpoints save optimizer, EMA, latent statistics and random states; the AE fingerprint prevents decoding with a different encoder. '
                  'A Study 9 `artifacts/STOP` file stops diffusion at the next durable 50-step checkpoint.', '',
                  '## Primary references', '',
                  '- [SSSD-ECG official repository and published-data links](https://github.com/AI4HealthUOL/SSSD-ECG).',
                  '- [Improved DDPM — cosine schedule](https://arxiv.org/abs/2102.09672).',
                  '- [Progressive Distillation — v parameterization](https://arxiv.org/abs/2202.00512).',
                  '- [DDIM](https://arxiv.org/abs/2010.02502).',
                  '- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).',
                  '- [Reliable Fidelity and Diversity Metrics — density/coverage](https://arxiv.org/abs/2002.09797).',
                  '- [ECGSYN on PhysioNet](https://physionet.org/content/ecgsyn/1.0.0/) and '
                  '[NeuroKit implementation](https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_simulate.py).', ''])
    report = '\n'.join(lines)
    (HOME / 'REPORT.md').write_text(report, encoding='utf-8')
    # A directly viewable comparison table alongside the complete Markdown report.
    with (folder / 'scorecard_summary.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    table = '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>'
    table += ''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in rows[1:])+'</tbody></table>'
    (HOME / 'BENCHMARK.html').write_text('<!doctype html><meta charset="utf-8"><title>latent_diffusion benchmark</title>'
        '<style>body{font:14px system-ui;margin:32px;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;text-align:right}th{background:#eee}td:first-child{text-align:left}</style>'
        '<h1>latent_diffusion: exploratory generative scorecard</h1><p>Per-rhythm, class-matched reference subsets; k=5. '
        'Blank privacy cells mean excluded by provenance. ECGSYN does not model AFIB. Published SSSD unavailable from upstream. '
        'Read REPORT.md for protocols and limitations.</p>'+table, encoding='utf-8')
    print('Written', HOME / 'REPORT.md')


if __name__ == '__main__':
    run()
