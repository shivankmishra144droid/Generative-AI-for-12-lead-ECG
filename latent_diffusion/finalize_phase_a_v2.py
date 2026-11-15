"""Phase A review corrections, secondary-detector uncertainty, durable audit."""
import json
import shutil
import numpy as np
from .common import CLASSES,digest
from .v2_common import ART,RESULTS,AUDITS,register,write_json,gpu_lease,now,seed_all
from .phase_a_v2 import matched


def preregister():
    register('phaseA_review_corrections',dict(
        registered_before_secondary_detector_results=True,
        secondary_detector='Five seeds 9200–9204; frozen run(); 15 epochs; seed t interval labelled exploratory',
        hybrid_privacy='Exclude development privacy: historical hybrid trained on all train, including development patients',
        evidence_preservation='Archive original provisional outputs before correction; preserve all first-seed scores',
        gate='No numerical acceptance threshold changed; original failed A8 retained'))


def strip_hybrid_privacy():
    archive=AUDITS/'provisional_invalid_hybrid_privacy';archive.mkdir(exist_ok=True)
    affected=[]
    for filename in ['rescore_hybrid_full.json','rescore_hybrid_floor.json','scorecard_v1_rescored.json']:
        path=RESULTS/filename
        if not path.exists():raise FileNotFoundError(path)
        original=json.loads(path.read_text())
        target=original['expanded']['hybrid'] if filename.startswith('scorecard') else original
        records=[target['primary'],target['floor_matched']] if 'primary' in target else [target]
        bad=any(any(k.startswith('privacy_') for k in row) for record in records for row in record['by_class'].values())
        if bad:
            archived=archive/filename
            if archived.exists():raise ValueError('Do not overwrite archived provisional output')
            shutil.copyfile(path,archived);affected.append(dict(file=filename,original_sha256=digest(archived)))
            for record in records:
                for row in record['by_class'].values():
                    for key in list(row):
                        if key.startswith('privacy_'):del row[key]
                record['privacy_status']='Excluded: development patients also used to fit historical hybrid; no valid member/nonmember labels'
            write_json(path,original)
    if affected:
        write_json(archive/'INVALID.json',dict(timestamp_utc=now(),affected=affected,
            invalid_fields='Hybrid development privacy only; other metrics unaffected',
            reason='Development is not nonmember data for the historical full-train hybrid. Do not interpret archived hybrid privacy numbers.'))


def secondary_detectors():
    from synthetic_generation.realism.detect import run,summarise
    datasets={key:dict(np.load(ART/'metric_cache'/f'{key}.npz')) for key in ('real_full','fit_reference','floor_a','floor_b')}
    all_results={}
    for name in ('floor','vae','latent_diffusion','ecgsyn','hybrid'):
        if (ART/'STOP').exists():raise InterruptedError('STOP')
        if name=='floor':rr,ss=datasets['floor_a'],datasets['floor_b']
        else:
            syn=dict(np.load(ART/'metric_cache'/f'{name}_expanded.npz'))
            rr,ss,_=matched(datasets['real_full'],syn,datasets['fit_reference'])
        path=RESULTS/f'detector_{name}.json';runs={}
        if path.exists():
            old=json.loads(path.read_text())
            runs=old['runs'] if 'runs' in old else {'9200':old}
        with gpu_lease('phaseA_detector_review_'+name) as device:
            for seed in range(9200,9205):
                if str(seed) in runs:continue
                seed_all(seed)
                runs[str(seed)]=run(rr['X'],rr['y'],ss['X'],ss['y'],seed=seed,epochs=15,device=str(device),g_real=rr['groups'],g_syn=ss['groups'])
                # Save each seed so an interruption does not discard completed runs.
                write_json(path,dict(runs=runs,status='in progress'))
                print('Secondary detector',name,seed,runs[str(seed)]['auc'],flush=True)
        summary=dict(runs=runs,auc=summarise([r['auc'] for r in runs.values()]),per_class={},status='complete',
                     uncertainty='Exploratory Student-t interval over 5 training seeds, not patient-bootstrap CI; not used as a gate')
        for label in CLASSES:
            values=[r['per_class'][label] for r in runs.values() if label in r['per_class']]
            if values:summary['per_class'][label]=summarise(values)
        write_json(path,summary);all_results[name]=summary
    write_json(RESULTS/'secondary_detectors.json',all_results)


def run():
    preregister();strip_hybrid_privacy();secondary_detectors()
    write_json(AUDITS/'phaseA_review_corrections_completed.json',dict(timestamp_utc=now(),
        hybrid_development_privacy_excluded=True,secondary_detector_seed_count=5,
        gate_changed=False,downstream_experiments_started=False))


if __name__=='__main__':run()
