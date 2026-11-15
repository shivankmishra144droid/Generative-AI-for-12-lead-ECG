"""Expand frozen V1 generators into V2 paths without calling V1 writers."""
import argparse
import json
import numpy as np
import torch
from .common import ROOT,ART as V1_ART,digest
from .v2_common import ART,gpu_lease,register,seed_all,write_json
from .generate import load_ae,training_conditions,project_leads,optional_dependencies
from .train import decoder_condition
from .latent.latent_diffusion import LatentSSSD,LatentDiffusion


@torch.no_grad()
def expand(kind,n=256,seed=9202):
    path=ART/'pools'/f'{kind}_v1_expanded.npz'
    if path.exists():return path
    register(f'phaseA_expand_{kind}',dict(n_per_class=n,seed=seed,frozen_v1=True,ddim=50,cfg=2))
    path.parent.mkdir(parents=True,exist_ok=True)
    seed_all(seed);y,cond=training_conditions(n,seed);output=[]
    if kind=='ecgsyn':
        optional_dependencies()
        from .ecgsyn_adapter import simulate
        stats=json.loads((ROOT/'processed_data/normalization_stats.json').read_text())
        keep=y!=1;y,cond=y[keep],cond[keep]
        for i,(label,cv) in enumerate(zip(y,cond)):
            if (ART/'STOP').exists():raise InterruptedError('STOP')
            hr=float(cv[0]*40+75) if cv[3] else {0:70.,2:110.,3:45.}[int(label)]
            data=simulate(duration=10,sampling_rate=100,heart_rate=float(np.clip(hr,25,200)),
                          heart_rate_std=1.,noise=.01,method='multileads',random_state=seed+i)
            output.append(((data[stats['lead_order']].to_numpy()[:1000]-stats['mean'])/stats['std'])[None])
            if (i+1)%64==0:print(kind,i+1,flush=True)
        provenance=dict(adapter='V1 ECGSYN adapter; sinus only',seed=seed)
    else:
        with gpu_lease(f'phaseA_expand_{kind}') as device:
            ae=load_ae('vae' if kind=='vae' else 'ae',device)
            if kind=='latent_diffusion':
                ck=torch.load(V1_ART/'diffusion/last.pt',map_location=device,weights_only=False)
                if ck['ae_sha256']!=digest(V1_ART/'ae/best.pt'):raise ValueError('V1 AE hash changed')
                diffusion=LatentDiffusion(LatentSSSD(**ck['config'])).to(device).eval()
                diffusion.load_state_dict(ck['ema'])
            for i in range(0,len(y),8):
                if (ART/'STOP').exists():raise InterruptedError('STOP')
                labels=torch.as_tensor(y[i:i+8],device=device);values=torch.as_tensor(cond[i:i+8],device=device)
                if kind=='latent_diffusion':z=diffusion.sample(labels,values,50,2.,seed+i)*ck['std'].to(device)+ck['mean'].to(device)
                else:z=torch.randn(len(labels),16,125,device=device)
                output.append(ae.decode(z,decoder_condition(labels,values)).transpose(1,2).cpu().numpy())
                if (i+8)%128==0:print(kind,i+8,flush=True)
            provenance=dict(seed=seed,checkpoint_sha256=digest(V1_ART/('diffusion/last.pt' if kind=='latent_diffusion' else 'vae/best.pt')))
    raw=np.concatenate(output).astype(np.float32)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite baseline')
    np.savez_compressed(path,X=project_leads(raw),y=y,conditions=cond)
    write_json(path.with_suffix('.json'),dict(**provenance,n_per_class=n,pool_sha256=digest(path),
               note='Fresh draws from frozen V1 generator; not additional training; no V1 file overwritten'))
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['vae','latent_diffusion','ecgsyn']);args=p.parse_args()
    print(expand(args.kind))
