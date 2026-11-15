"""Render the actual V2 evidence. Missing experiments are never filled with estimates."""
import html
import json
import csv
from pathlib import Path
import numpy as np
from .common import HOME,CLASSES
from .v2_common import RESULTS,AUDITS,ART,now,write_json


def read(name):
    path=RESULTS/name
    return json.loads(path.read_text()) if path.exists() else None


def fmt(x):return 'undefined' if x is None else f'{x:.4f}'


def ci(item):
    if item.get('ci95') is None:return fmt(item.get('value'))+' [unavailable]'
    return fmt(item['value'])+' ['+', '.join(fmt(v) for v in item['ci95'])+']'


def export_csv(floor,scores):
    rows=[]
    def walk(node,prefix,base):
        if not isinstance(node,dict):return
        if 'value' in node:
            bounds=node.get('ci95') or [None,None]
            rows.append(dict(**base,metric=prefix,value=node['value'],ci95_lower=bounds[0],ci95_upper=bounds[1],
                split_sd=node.get('sd'),split_min=node.get('minimum'),split_max=node.get('maximum'),
                uncertainty='500-replicate percentile cluster bootstrap' if 'ci95' in node else 'five grouped splits: SD and range, not CI'))
            return
        for key,value in node.items():walk(value,f'{prefix}/{key}' if prefix else key,base)
    candidates=[('floor','real_real',floor)]
    if scores:
        candidates += [(cohort,name,item[cohort]) for name,item in scores['expanded'].items() for cohort in ('primary','floor_matched')]
        candidates += [('original_saved_pool',name,item) for name,item in scores['original_saved_pools'].items()]
    for cohort,name,score in candidates:
        walk(score['feature_judges'],'feature_judges',dict(cohort=cohort,pool=name,rhythm='ALL',n_real=sum(r['n_real'] for r in score['by_class'].values())))
        for label,row in score['by_class'].items():walk(row,'',dict(cohort=cohort,pool=name,rhythm=label,n_real=row['n_real']))
    path=RESULTS/'phaseA_scorecard.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plots(controls,floor):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    positions=np.arange(4);width=.35
    for j,emb in enumerate(('emb_task','emb_ssl')):
        values=[controls['by_class'][c][emb]['noise']['precision']['value'] for c in CLASSES]
        axes[0].bar(positions+(j-.5)*width,values,width,label=emb,color=('#476a8a','#b26e41')[j])
        values=[controls['by_class'][c][emb]['collapse']['coverage']['value'] for c in CLASSES]
        axes[1].bar(positions+(j-.5)*width,values,width,label=emb,color=('#476a8a','#b26e41')[j])
    for ax,title,ylabel,limit in zip(axes,['Gaussian-noise acceptance','Repeated-waveform coverage'],
                                     ['Embedding precision (fraction)','Embedding coverage (fraction)'],[1.05,.16]):
        ax.axhline(.1,color='black',linestyle='--',linewidth=1,label='Required <0.10')
        ax.set(xticks=positions,xticklabels=CLASSES,ylim=(0,limit),xlabel='Rhythm class',ylabel=ylabel,title=title)
        ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Study 9 V2 • Phase A negative controls • development data')
    fig.savefig(RESULTS/'controls_A8.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for j,emb in enumerate(('emb_task','emb_ssl')):
        for k,c in enumerate(CLASSES):
            v=floor['by_class'][c][emb]['coverage'];x=k+(j-.5)*.18
            ax.plot([x,x],v['ci95'],color=('#476a8a','#b26e41')[j],linewidth=3)
            ax.scatter(x,v['value'],color=('#476a8a','#b26e41')[j],s=40,label=emb if k==0 else None)
    ax.set(xticks=np.arange(4),xticklabels=CLASSES,xlabel='Rhythm class',ylabel='Embedding coverage (fraction)',ylim=(.45,1.04),
           title='Real–real floor: point estimates and 500-resample percentile intervals')
    ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.savefig(RESULTS/'bootstrap_diagnostic.png',dpi=170);plt.close(fig)


def run():
    floor=read('scorecard_floor.json');controls=read('controls_A8.json');gate=read('phaseA_gate.json')
    diagnosis=read('measurement_diagnosis.json');scores=read('scorecard_v1_rescored.json')
    audit=read('final_audit.json');encoder=read('encoder_ssl_training.json');secondary=read('secondary_detectors.json')
    if not all((floor,controls,gate,diagnosis,encoder)):raise RuntimeError('Phase A evidence not ready')
    plots(controls,floor)
    export_csv(floor,scores)
    feature_exclusions=[]
    for label,row in floor['by_class'].items():
        for name,feature in row['features'].items():
            v=feature['wasserstein']
            if v['ci95'] and not v['ci95'][0]<=v['value']<=v['ci95'][1]:
                feature_exclusions.append(dict(rhythm=label,feature=name,**v))
    write_json(RESULTS/'interval_diagnostics_extended.json',dict(wasserstein_point_exclusions=feature_exclusions,
        count=len(feature_exclusions),total=100,note='Point exclusion is a resampling-bias warning, not a direct test of true population coverage'))
    judge=floor['feature_judges']['full'];noise=controls['noise_judges']['full']
    failures='\n'.join(f"- `{key}`: {fmt(gate['checks'][key]['value'])}; required {gate['checks'][key]['rule']}." for key in gate['failures'])
    baseline_lines=[]
    if scores:
        for name,item in scores['expanded'].items():
            j=item['primary']['feature_judges']['full']
            baseline_lines.append(f"- **{name}:** full clinical judge {j['value']:.4f} ± {j['sd']:.4f} SD; five-split range [{j['minimum']:.4f}, {j['maximum']:.4f}]. [Full evidence](results/v2/rescore_{name}_full.json); [same-n floor comparison](results/v2/rescore_{name}_floor.json).")
            top=j.get('top8_permutation_importance',[])[:3]
            if top:
                detail=', '.join(f"{v['feature']} (AUC drop {v['auc_drop']:.4f})" for v in top)
                baseline_lines.append(f"  Largest held-out permutation effects: {detail}. Correlated features can share predictive information; these are diagnostic associations, not causal effects.")
    checks='Pending final software audit.'
    if audit:
        counts=[c['stdout'].strip().splitlines()[-1] for c in audit['regression_checks']]
        checks=f"Software checks pass: **{audit['software_checks_pass']}**. {'; '.join(counts)}. Pre-existing project files checked: {audit['baseline_files_checked']}; changed: {len(audit['changed_preexisting_files'])}. V1 files changed: {len(audit['v1_changed_files'])}. Experimental Gate A remains **failed**. [Audit](results/v2/final_audit.json)."
    report=f'''# latent_diffusion — V2 implementation and Phase A report

Generated {now()}. This report covers the work actually executed. V1 remains in `REPORT.md`.

## Answer and completion status

**The whole V2 study is not complete. Phase A was implemented and run, but its mandatory acceptance gate failed.** The current evidence cannot establish that a V2 generator beats the hybrid or that rhythm control works: no new V2 generator was trained, and no V2 validation benchmark or utility run was performed. The instruction explicitly says “Do not continue to B until A passes.” B–G therefore remain blocked. This is a negative measurement result, not a negative result for a larger diffusion model. [Original instructions](STUDY9_V2_INSTRUCTIONS.md); [registered gate and outcomes](results/v2/phaseA_gate.json).

## What changed from V1

- **D1, reconstruction realism:** the proposed all-development AE judge is still pending behind Gate A. Existing reconstruction results do not mathematically bound every possible latent prior.
- **D2, capacity and training:** V1 checkpoints were preserved. Larger balanced diffusion runs remain pending; fresh V1 samples do not count as V2 training.
- **D3, rhythm control:** richer rhythm conditioning remains pending. Current HR and peak metrics are detector-derived proxies.
- **D4, measurement:** added a fit-only contrastive encoder, two embedding scorecards, 500 patient-cluster resamples, five grouped clinical-judge splits, permutation importance, four feature-group judges, full rhythm sanity metrics and privacy controls.
- **D5, utility:** the old short pilot is unchanged. It is not upgraded to a full utility result by this work.

## Phase A: what worked

The clinical-feature floor is **AUC {judge['value']:.4f} ± {judge['sd']:.4f} SD**, with five-split range [{judge['minimum']:.4f}, {judge['maximum']:.4f}]. Its full and group judges satisfy the registered floor checks. Gaussian noise is separable by the full clinical judge at **AUC {noise['value']:.4f}**. Copy controls have membership AUC 1.000 in all four classes and near-zero waveform-distance ratios. The fixed normal-data bootstrap sanity simulation passes. These controls do not validate every other score. [Floor](results/v2/scorecard_floor.json); [negative/copy controls](results/v2/controls_A8.json).

The primary development reference contains 256 SR, 170 AFIB, 82 STACH and 63 SBRAD recordings. All minority recordings are retained and repeated patients remain grouped. Two disjoint development halves cannot each contain the entire minority reference. The preregistered feasible floor therefore uses 256/79/38/29 per side, and each generator has a separate comparison at those counts. Primary and smaller-floor intervals must not be compared as if sample sizes matched. [Pre-run amendment](audits/v2/discrepancies.md); [cohorts](results/v2/cohorts.json).

## Phase A: failed controls and diagnosis

{failures}

![Negative controls](results/v2/controls_A8.png)

For SR, Gaussian noise receives embedding precision **0.9844** in the task encoder and **1.0000** in SSL, despite recall **0** in both. Two of 256 task-embedding real-neighbour balls cover some noise; nine of 256 SSL balls do. A compact noise cluster can therefore obtain high precision inside broad real-neighbour regions. Precision alone is unsafe as a fidelity claim here. This is an observation about this protocol, not a novelty claim. [Geometry and fallback diagnosis](results/v2/measurement_diagnosis.json).

The permitted frozen-AE mean/SD fallback also fails. Its noise precision is 0.1367 SR, 0.0059 AFIB, 0.3293 STACH and 0.2857 SBRAD; its real-real precision is below 0.85 in STACH and SBRAD. It was evaluated descriptively after the original result and did not replace the original gate.

**{diagnosis['prdc_bootstrap_diagnostic']['count']} of {diagnosis['prdc_bootstrap_diagnostic']['total']}** floor PRDC percentile intervals exclude their own original point estimate. Resampling duplicates changes k-neighbour radii. The requested intervals are retained; they are not widened or claimed to have validated coverage. The simple mean-bootstrap sanity test does not repair uncertainty for a non-smooth neighbour statistic. [Interval diagnosis](results/v2/measurement_diagnosis.json).

The same inclusion check flags **{len(feature_exclusions)} of 100** raw clinical-feature Wasserstein floor intervals. Point exclusion is a warning about resampling bias, not a direct measurement of population coverage. Both distance families need coverage validation before final CI-based decisions. [Extended interval check](results/v2/interval_diagnostics_extended.json).

![PRDC interval diagnostic](results/v2/bootstrap_diagnostic.png)

## Frozen V1 generators, rescored on development

{chr(10).join(baseline_lines) if baseline_lines else 'Baseline rescoring is still running; no complete comparison is claimed.'}

The expanded pools contain fresh samples from frozen V1 weights/settings. Original saved-pool rescoring is reported separately; it uses at most 64 per class and 63 SBRAD. ECGSYN excludes AFIB. The historical hybrid was fitted using the full training split, including these development patients, so its comparison is descriptive and its development membership attack is excluded. Any provisional privacy calculations are archived and marked invalid. The neural V1 pools use the fit partition. [Combined scorecards](results/v2/scorecard_v1_rescored.json); [review corrections](audits/v2/phaseA_review_corrections_registration.json).

The secondary CNN detector is retained but never used as a gate. Its five-seed intervals describe variation over training seeds, not patient uncertainty. All metric CIs are exploratory. Repeated clinical-judge split ranges are not CIs. [Detector results](results/v2/secondary_detectors.json).

## Phases B–H: exact status

- **B:** not run. No all-dev reconstruction judge, AE ladder or chosen V2 AE.
- **C:** not run. No 100,000-step capacity comparison or claim that capacity fixes realism.
- **D:** not run. No trained R-map model or V2 control sweep.
- **E:** not run. No 30-setting sampler sweep or frozen final sampler.
- **F:** not run. Validation remains unopened in V2. `BENCHMARK_V2.html` displays Phase A development evidence and explicitly states that it is not the final validation benchmark.
- **G:** not run. No five-seed, 6,250-step-per-arm utility claim.
- **H:** this partial-results report is written. It does not substitute for the requested final F/G report.

## Review, safeguards and limitations

{checks}

The phase review is [phaseA.md](audits/v2/phaseA.md); encoder review is [encoder_review.md](audits/v2/encoder_review.md). No instruction threshold was relaxed after observing a result. No study 5 output was used. No download, commit or push was performed. No external package or project source file was edited. GPU jobs were serialized and their preflight logs retained.

The task encoder was historically selected on validation by earlier work. The small SSL model was trained on one seed. Several minority floor cohorts are small. The historical hybrid includes development patients in training. HR and R-peaks depend on a heuristic detector. Feature NaNs and missing HR are counted explicitly. No QT claim, clinical reader study or privacy guarantee is made. Published SSSD remains unavailable: the document specifies a user-managed `Dataset.zip` (929.87 MB) from its figshare source; this run did not retry or bypass that endpoint.

## Execution and reproducibility

Use `py -3.14`, from the repository root. All outputs below are confined to Study 9 V2. The old commands in `README.md` retain their old behavior.

```powershell
py -3.14 -m latent_diffusion.encoder_v2
py -3.14 -m latent_diffusion.phase_a_v2
py -3.14 -m latent_diffusion.diagnostics_v2
py -3.14 -m latent_diffusion.finalize_phase_a_v2
py -3.14 -m latent_diffusion.final_audit_v2
py -3.14 -m latent_diffusion.report_v2
```

Seed 9200 selects development cohorts and controls; seed 9202 expands frozen V1 generators. The encoder completed 20 epochs in **{encoder['elapsed_seconds']:.1f} seconds**, with peak reserved GPU memory **{encoder['peak_reserved_mib']:.0f} MiB**. Selected development contrastive loss: **{min(h['dev_loss'] for h in encoder['history']):.4f}**. These are measured encoder costs, not diffusion-capacity estimates. [Training evidence](results/v2/encoder_ssl_training.json). Software versions and checkpoint/source hashes are in [final audit](results/v2/final_audit.json). Logs are under `artifacts/v2/logs/`.

Creating `artifacts/v2/STOP` stops at the next documented checkpoint/pool/scorecard boundary. Remove it explicitly before resuming. The encoder restores RNG and optimizer state. Completed scorecards and complete pools are cached; an interrupted generation pool restarts that pool. The empirical test `test_empirical_gate_a` intentionally remains failing while Gate A fails; the regression audit reports software tests separately.

## Concrete next plan

1. Keep this failed gate and its results immutable. Do not call the entire V2 plan complete.
2. Revise the measurement protocol explicitly before another run: validate an embedding or waveform distance sensitive to morphology/noise; treat the failed task-embedding precision as descriptive. Requiring this fixed task embedding to reject noise cannot be repaired by training a better generator.
3. Validate an uncertainty method for PRDC against controlled distributions and sample sizes; do not use the current percentile intervals to decide superiority. Fix the clinical-judge CI construction before Phase F.
4. Repeat Phase A controls under the revised, preregistered protocol. Only a passing measurement gate can release B, then the unchanged B→C→D→E→F→G sequence.

These are proposed next experiments, not changes silently applied to the fixed gates.
'''
    (HOME/'REPORT_V2.md').write_text(report,encoding='utf-8')
    sections=[]
    def render_card(title,score):
        rows=[]
        for c,row in score['by_class'].items():
            cells=[c,str(row['n_real'])]
            for emb in ('emb_task','emb_ssl'):
                for metric in ('precision','recall','density','coverage'):cells.append(ci(row[emb][metric]))
            cells.append(ci(row['features']['hr']['wasserstein']))
            rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
        headings=['Class','n','Task precision','Task recall','Task density','Task coverage','SSL precision','SSL recall','SSL density','SSL coverage','HR W₁ (bpm)']
        j=score['feature_judges']['full']
        return f'<section><h2>{html.escape(title)}</h2><p>Clinical judge: {j["value"]:.4f} ± {j["sd"]:.4f} SD across 5 grouped splits.</p><div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headings)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></section>'
    sections.append(render_card('Real–real floor (402 per side)',floor))
    if scores:
        for name,item in scores['expanded'].items():
            sections.append(render_card(name+' — full development reference',item['primary']))
            sections.append('<details><summary>'+html.escape(name)+' — same counts as floor</summary>'+render_card(name+' — floor-matched',item['floor_matched'])+'</details>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Study 9 V2 — Phase A evidence</title><style>body{font:15px system-ui;max-width:1600px;margin:40px auto;padding:0 24px;color:#24292f;background:#fff}h1{font-size:28px}h2{font-size:20px}.notice{border-left:5px solid #b84430;padding:12px 20px;background:#fff6f3}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{text-align:left;white-space:nowrap;padding:10px;border-bottom:1px solid #ddd}th{background:#f5f5f5}section{margin:32px 0}img{width:min(100%,1100px)}summary{cursor:pointer;font-weight:600}a{color:#245b8c}footer{margin-top:30px;color:#57606a}</style><h1>Study 9 V2: measurement gate failed</h1><div class="notice"><strong>Phase A development evidence only.</strong> Phases B–G were not run. This is not the final validation benchmark. No claim that V2 beats the hybrid is supported.</div><p>PRDC: point [500-resample percentile interval]. These neighbour-statistic intervals failed calibration diagnostics and must not support a superiority claim. Clinical-judge uncertainty is split SD, not a 95% CI. ECGSYN has no AFIB. Hybrid development scores are not patient-held-out.</p><p><a href="REPORT_V2.md">Written report</a> · <a href="results/v2/phaseA_gate.json">Gate evidence</a> · <a href="results/v2/measurement_diagnosis.json">Measurement diagnosis</a></p><img src="results/v2/controls_A8.png" alt="Gaussian noise precision and collapsed-waveform coverage, compared with the required 0.1 threshold">'''+''.join(sections)+f'<footer>Generated {html.escape(now())}. Source: results/v2/*.json. No validation data used.</footer></html>'
    (HOME/'BENCHMARK_V2.html').write_text(page,encoding='utf-8')
    write_json(RESULTS/'study_status.json',dict(updated_utc=now(),phaseA_implementation_complete=bool(scores),
        phaseA_acceptance=gate['passed'],whole_instructions_complete=False,
        phases={p:'blocked by Gate A' for p in 'BCDEFG'},report='REPORT_V2.md',validation_used=False))
    print('Wrote REPORT_V2.md and BENCHMARK_V2.html with actual Phase A evidence',flush=True)


if __name__=='__main__':run()
