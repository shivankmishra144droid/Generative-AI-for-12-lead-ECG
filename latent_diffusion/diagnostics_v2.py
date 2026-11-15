"""Post-gate measurement diagnosis. Does not replace or relax Gate A."""
import json
import numpy as np
import torch
from scipy.spatial.distance import cdist
from .generate import load_ae
from .v2_common import ART,RESULTS,register,write_json,seed_all
from .gen_eval_v2 import prdc_uncertainty
from .common import CLASSES


class AEEmbedding:
    def __init__(self):self.model=load_ae(device='cpu')
    @torch.no_grad()
    def __call__(self,x):
        out=[]
        for i in range(0,len(x),64):
            mu=self.model.encode(torch.as_tensor(x[i:i+64]).transpose(1,2))[0]
            out.append(torch.cat((mu.mean(-1),mu.std(-1,unbiased=False)),1).numpy())
        return np.concatenate(out)


def run():
    register('phaseA_post_gate_diagnosis',dict(
        trigger='Original Gate A failed; original gate and all scores remain unchanged',
        fallback='Instruction A4 option 2: frozen V1 AE mean and population SD, 32 dimensions',
        gate_use='Descriptive only: task-embedding failure alone still prevents Gate A passing',
        diagnostics=['noise ball coverage witnesses','original bootstrap point exclusion counts','AE fallback A8 controls']))
    seed_all(9200)
    data={k:dict(np.load(ART/'metric_cache'/f'{k}.npz')) for k in ('real_full','fit_reference','control_noise','control_collapse','floor_a','floor_b')}
    result={'original_gate_preserved':True,'noise_geometry':{},'ae_fallback':{}}
    for c in range(4):
        r=data['real_full'];s=data['control_noise'];ri=r['y']==c;si=s['y']==c
        result['noise_geometry'][CLASSES[c]]={}
        for emb in ('emb_task','emb_ssl'):
            rr=cdist(r[emb][ri],r[emb][ri]);np.fill_diagonal(rr,np.inf)
            radii=np.partition(rr,4,axis=1)[:,4];rs=cdist(r[emb][ri],s[emb][si])
            covered=rs<=radii[:,None]
            result['noise_geometry'][CLASSES[c]][emb]=dict(
                real_balls_covering_any_noise=int(covered.any(1).sum()),real_ball_count=int(ri.sum()),
                noise_covered_fraction=float(covered.any(0).mean()),
                noise_mean_distance_to_centroid=float(np.linalg.norm(s[emb][si]-s[emb][si].mean(0),axis=1).mean()),
                real_mean_distance_to_centroid=float(np.linalg.norm(r[emb][ri]-r[emb][ri].mean(0),axis=1).mean()),
                interpretation='A tight noise cluster can fall inside a few large real k-neighbor balls: high precision does not imply realistic waveforms')
    encoder=AEEmbedding();encoded={k:encoder(v['X']) for k,v in data.items()}
    mean=encoded['fit_reference'].mean(0);sd=np.maximum(encoded['fit_reference'].std(0),1e-6)
    encoded={k:(v-mean)/sd for k,v in encoded.items()}
    for c in range(4):
        rows={k:v['y']==c for k,v in data.items()};out={}
        for name,left,right in [('noise','real_full','control_noise'),('collapse','real_full','control_collapse'),('floor','floor_a','floor_b')]:
            out[name]=prdc_uncertainty(encoded[left][rows[left]],encoded[right][rows[right]],
                            data[left]['groups'][rows[left]],data[right]['groups'][rows[right]])
        result['ae_fallback'][CLASSES[c]]=out
    floor=json.loads((RESULTS/'scorecard_floor.json').read_text())
    excluded=[];count=0
    for c,row in floor['by_class'].items():
        for emb in ('emb_task','emb_ssl'):
            for metric,v in row[emb].items():
                count+=1
                if not v['ci95'][0]<=v['value']<=v['ci95'][1]:
                    excluded.append(dict(rhythm=c,embedding=emb,metric=metric,**v))
    result['prdc_bootstrap_diagnostic']=dict(point_outside_own_interval=excluded,count=len(excluded),total=count,
       explanation='Ordinary resampling duplicates points and changes k-neighbor radii. Percentile intervals are reported as requested but do not establish calibrated coverage for this statistic.',
       action='Do not use these PRDC intervals for a superiority claim before a separately validated uncertainty protocol')
    write_json(RESULTS/'measurement_diagnosis.json',result)
    print('Diagnosis complete; original Gate A remains failed',flush=True)


if __name__=='__main__':run()
