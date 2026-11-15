"""Resumable Phase A: prepared caches, controls, floors and frozen baseline rescoring."""
import json
import numpy as np
import torch
from .common import HOME,ROOT,ART as V1_ART,CLASSES,digest
from .v2_common import ART,RESULTS,AUDITS,SEED,training_data,register,write_json,seed_all,snapshot_v1,gpu_lease
from .gen_eval import ECGEncoder,clinical_features
from .encoder_v2 import SSLEmbedding
from .gen_eval_v2 import (score_prepared,rhythm_rows,feature_judges,prdc_uncertainty,privacy_uncertainty,GROUPS)
from .test_metrics_v2 import bootstrap_sanity


def select(y,candidates,counts,seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.choice(candidates[y[candidates]==c],int(n),replace=False)
                           for c,n in enumerate(counts) if n])


def subset(data,indices):return {k:v[indices] for k,v in data.items()}


def prepare(name,x,y,groups,encoders,scalers,afib_cv=None):
    cache=ART/'metric_cache';cache.mkdir(exist_ok=True)
    path=cache/f'{name}.npz'
    if path.exists():
        saved=dict(np.load(path,allow_pickle=False))
        if not np.array_equal(saved['y'],y) or not np.array_equal(saved['groups'],groups):raise ValueError('Cached cohort changed')
        return saved
    print('Preparing',name,len(x),flush=True)
    features=clinical_features(x)
    if afib_cv is None:afib_cv=float(np.nanmedian(features[y==1,3]))
    result=dict(X=np.asarray(x,dtype=np.float32),y=np.asarray(y),groups=np.asarray(groups),features=features,
                rhythm=rhythm_rows(x,features,afib_cv))
    for key,encoder in encoders.items():
        emb=encoder(x);mean,sd=scalers[key]
        result[key]=(emb-mean)/sd
    np.savez_compressed(path,**result)
    return result


def cohort_setup():
    x,y,patient,fit,dev=training_data()
    part=np.load(ART/'floor_partition.npz')
    full_counts=[256,170,82,63];floor_counts=[256,79,38,29]
    ref=select(y,dev,full_counts,SEED)
    tr=select(y,fit,full_counts,SEED+1)
    ra=select(y,part['a'],floor_counts,SEED+2)
    rb=select(y,part['b'],floor_counts,SEED+3)
    np.savez_compressed(ART/'evaluation_cohorts.npz',full=ref,fit=tr,floor_a=ra,floor_b=rb)
    if set(patient[ra])&set(patient[rb]):raise ValueError('Floor patient overlap')
    encoders={'emb_task':ECGEncoder(),'emb_ssl':SSLEmbedding()}
    scale_path=ART/'embedding_scalers.npz'
    if not scale_path.exists():
        scales={}
        for key,encoder in encoders.items():
            values=encoder(np.asarray(x[tr]));scales[key+'_mean']=values.mean(0);scales[key+'_sd']=np.maximum(values.std(0),1e-6)
        np.savez_compressed(scale_path,**scales)
    scales=np.load(scale_path)
    scalers={key:(scales[key+'_mean'],scales[key+'_sd']) for key in encoders}
    # AFIB cutoff uses all development AFIB records, independently of each candidate.
    afib_cv=float(np.nanmedian(clinical_features(np.asarray(x[dev[y[dev]==1]]))[:,3]))
    datasets={}
    for name,rows in [('real_full',ref),('fit_reference',tr),('floor_a',ra),('floor_b',rb)]:
        datasets[name]=prepare(name,np.asarray(x[rows]),y[rows],np.asarray([f'p:{p}' for p in patient[rows]]),encoders,scalers,afib_cv)
    write_json(RESULTS/'cohorts.json',dict(full_counts=full_counts,floor_counts=floor_counts,afib_cv=afib_cv,
        scaler='Fixed fit-only matched reference, 571 ECGs; identical for every candidate',
        task_encoder_sha256=digest(ROOT/'baseline/cnn_lstm_baseline.pt'),
        ssl_encoder_sha256=digest(ART/'encoder_ssl/best.pt'),cohorts_sha256=digest(ART/'evaluation_cohorts.npz')))
    return datasets,encoders,scalers,afib_cv


def matched(real,syn,train):
    counts=[min(int((real['y']==c).sum()),int((syn['y']==c).sum()),int((train['y']==c).sum())) for c in range(4)]
    # Fixed ordering for real/fit; independent fixed random draw of candidate rows.
    rng=np.random.default_rng(SEED)
    idx=[]
    for data,randomize in ((real,False),(syn,True),(train,False)):
        idx.append(np.concatenate([rng.choice(np.flatnonzero(data['y']==c),n,replace=False) if randomize else np.flatnonzero(data['y']==c)[:n]
                                   for c,n in enumerate(counts) if n]))
    return tuple(subset(data,rows) for data,rows in zip((real,syn,train),idx))


def cached_score(name,real,syn,train,afib_cv,privacy=True):
    path=RESULTS/f'{name}.json'
    if path.exists():return json.loads(path.read_text())
    if (ART/'STOP').exists():raise InterruptedError('STOP')
    real,syn,train=matched(real,syn,train)
    result=score_prepared(real,syn,train,('emb_task','emb_ssl'),afib_cv,privacy=privacy)
    write_json(path,result)
    print('Scored',name,'judge',result['feature_judges']['full']['value'],flush=True)
    return result


def controls(data,encoders,scalers,afib_cv,floor):
    path=RESULTS/'controls_A8.json'
    if path.exists():return json.loads(path.read_text())
    real,train=data['real_full'],data['fit_reference']
    rng=np.random.default_rng(SEED+4)
    # Noise has fit-only class/lead SD; it never copies a real temporal trace.
    noise=np.empty_like(real['X']);collapse=np.empty_like(real['X'])
    for c in range(4):
        r=real['y']==c;t=train['y']==c
        sd=train['X'][t].std(axis=(0,1))
        noise[r]=rng.normal(size=noise[r].shape)*sd
        collapse[r]=train['X'][t][0]
    groups=np.asarray([f'control:{i}' for i in range(len(noise))])
    n=prepare('control_noise',noise,real['y'],groups,encoders,scalers,afib_cv)
    c=prepare('control_collapse',collapse,real['y'],groups,encoders,scalers,afib_cv)
    noise_judge=feature_judges(real['features'],n['features'],real['y'],n['y'],real['groups'],n['groups'])
    result=dict(noise_judges=noise_judge,by_class={},bootstrap_sanity=bootstrap_sanity(),
        collapse_definition='One fixed fit ECG per class repeated, no within-class variation',
        noise_definition='Independent zero-mean Gaussian, class/lead SD from fixed fit reference')
    for label in range(4):
        key=CLASSES[label];r=real['y']==label;t=train['y']==label
        row={}
        for emb in encoders:
            row[emb]={'noise':prdc_uncertainty(real[emb][r],n[emb][r],real['groups'][r],n['groups'][r]),
                      'collapse':prdc_uncertainty(real[emb][r],c[emb][r],real['groups'][r],c['groups'][r])}
        row['exact_copy']=privacy_uncertainty(train['X'][t].reshape(t.sum(),-1),real['X'][r].reshape(r.sum(),-1),
                    train['X'][t].reshape(t.sum(),-1),train['groups'][t],real['groups'][r],train['groups'][t])
        result['by_class'][key]=row
        print('Controls',key,flush=True)
    write_json(path,result)
    return result


def gate_a(floor,controls):
    checks={}
    for c,row in floor['by_class'].items():
        for emb in ('emb_task','emb_ssl'):
            for key in ('precision','coverage'):
                value=row[emb][key]['value'];checks[f'floor/{c}/{emb}/{key}']=dict(value=value,rule='>=0.85',passed=value>=.85)
    full=floor['feature_judges']['full']
    judge_pass=.4<=full['value']<=.6 and full['sd']<=.06
    group_pass=all(.4<=v['value']<=.6 and v['sd']<=.06 for k,v in floor['feature_judges'].items() if k!='full')
    # The document explicitly allows smaller group judges when the full judge fails.
    mode='full' if judge_pass else 'four_group_fallback'
    checks['floor/judge']=dict(value=full,rule='AUC 0.40–0.60 and SD <=0.06; fallback all four groups',
                               passed=judge_pass or group_pass,fallback_used=not judge_pass,
                               group_values={k:v for k,v in floor['feature_judges'].items() if k!='full'})
    nj=controls['noise_judges']['full']['value']
    checks['noise/judge']=dict(value=nj,rule='>0.95',passed=nj>.95)
    for c,row in controls['by_class'].items():
        cp=row['exact_copy']
        checks[f'copy/{c}/distance']=dict(value=cp['ratio_of_medians']['value'],rule='<1e-5 (approximately zero)',passed=cp['ratio_of_medians']['value']<1e-5)
        checks[f'copy/{c}/membership']=dict(value=cp['membership_auc']['value'],rule='>0.9',passed=cp['membership_auc']['value']>.9)
        for emb in ('emb_task','emb_ssl'):
            for kind,metrics in (('noise',('precision',)),('collapse',('recall','coverage'))):
                for key in metrics:
                    v=row[emb][kind][key]['value'];checks[f'{kind}/{c}/{emb}/{key}']=dict(value=v,rule='<0.1',passed=v<.1)
    checks['bootstrap/sanity']=controls['bootstrap_sanity']
    result=dict(passed=all(v['passed'] for v in checks.values()),judge_gate_mode=mode,checks=checks,
                failures=[k for k,v in checks.items() if not v['passed']],
                instruction='Do not continue to B until A passes. Gate thresholds unchanged.')
    write_json(RESULTS/'phaseA_gate.json',result)
    return result


def run():
    snapshot_v1();seed_all(SEED)
    register('phaseA',dict(seed=SEED,bootstrap_replicates=500,judge_splits=5,judge_test_fraction=.3,
        floor_counts=[256,79,38,29],full_counts=[256,170,82,63],embedding_scale='571 fixed fit rows',
        a8_scope='Every supported class and both embeddings; copy tolerance ratio <1e-5',
        bootstrap_sanity='Fixed normal simulation, means at 512 and 1024; 500 replicates; no seed search',
        gate='All A8 plus floor judge .40-.60 SD<=.06; group fallback only for failed full judge',
        v1_baselines='Expanded frozen generators plus original-pool analysis',
        secondary_detector='Frozen imported run(), seed 9200, 15 epochs, GPU; descriptive AUC without CI'))
    data,encoders,scalers,afib_cv=cohort_setup()
    floor=cached_score('scorecard_floor',data['floor_a'],data['floor_b'],data['fit_reference'],afib_cv,privacy=False)
    control=controls(data,encoders,scalers,afib_cv,floor)
    gate=gate_a(floor,control)
    print('Gate A',gate['passed'],gate['failures'],flush=True)
    from .baselines_v2 import expand
    sources={name:expand(name) for name in ('vae','latent_diffusion','ecgsyn')}
    sources['hybrid']=ROOT/'generator_comparison/tstr/pool_beatgen.npz'
    results={};original={};secondary={}
    for name,path in sources.items():
        pool=np.load(path,allow_pickle=False)
        synthetic=prepare(name+'_expanded',pool['X'],pool['y'],np.asarray([f'{name}:{i}' for i in range(len(pool['y']))]),encoders,scalers,afib_cv)
        results[name]=dict(source_sha256=digest(path),
            primary=cached_score('rescore_'+name+'_full',data['real_full'],synthetic,data['fit_reference'],afib_cv,privacy=name not in ('ecgsyn','hybrid')),
            floor_matched=cached_score('rescore_'+name+'_floor',data['floor_a'],synthetic,data['fit_reference'],afib_cv,privacy=name not in ('ecgsyn','hybrid')))
        if name=='hybrid':results[name]['provenance_caution']='Historical hybrid used full training set, including V2 dev patients; descriptive comparison is not patient-held-out for this baseline'
        if name!='hybrid':
            old=np.load(V1_ART/'pools'/f'{name}.npz')
            osyn=prepare(name+'_original64',old['X'],old['y'],np.asarray([f'old_{name}:{i}' for i in range(len(old['y']))]),encoders,scalers,afib_cv)
            original[name]=cached_score('rescore_'+name+'_original64',data['real_full'],osyn,data['fit_reference'],afib_cv,privacy=name!='ecgsyn')
        write_json(RESULTS/'scorecard_v1_rescored.json',dict(expanded=results,original_saved_pools=original,split='internal development only'))
        detector_path=RESULTS/f'detector_{name}.json'
        if detector_path.exists():secondary[name]=json.loads(detector_path.read_text())
        else:
            from synthetic_generation.realism.detect import run as detector
            rr,ss,_=matched(data['real_full'],synthetic,data['fit_reference'])
            with gpu_lease('phaseA_detector_'+name) as device:
                secondary[name]=detector(rr['X'],rr['y'],ss['X'],ss['y'],seed=SEED,epochs=15,device=str(device),g_real=rr['groups'],g_syn=ss['groups'])
            write_json(detector_path,secondary[name])
    write_json(RESULTS/'secondary_detectors.json',secondary)
    write_json(AUDITS/'phaseA.json',dict(gate=gate,completed=True,downstream_allowed=gate['passed'],
                                      results=['scorecard_floor.json','scorecard_v1_rescored.json','controls_A8.json']))
    return gate


if __name__=='__main__':run()
