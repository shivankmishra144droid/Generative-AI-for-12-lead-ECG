"""Fit-only SimCLR ECG encoder, separate from the frozen rhythm classifier."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .v2_common import ART, RESULTS, SEED, gpu_lease, register, save_checkpoint, seed_all, training_data, write_json


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(64,64,5,padding=2),nn.GroupNorm(8,64),nn.SiLU(),
                               nn.Conv1d(64,64,5,padding=2),nn.GroupNorm(8,64))
    def forward(self,x):
        return F.avg_pool1d(F.silu(x+self.net(x)),2)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv1d(12,64,7,padding=3)
        self.blocks=nn.Sequential(*[Block() for _ in range(4)])
        self.head=nn.Linear(64,128)
        self.projection=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
    def forward(self,x):
        return self.head(self.blocks(self.stem(x)).mean(-1))


def augment(x):
    starts=torch.randint(201,(len(x),),device=x.device)
    indices=starts[:,None]+torch.arange(800,device=x.device)[None]
    z=x.gather(2,indices[:,None].expand(-1,12,-1))
    z=F.interpolate(z,size=1000,mode='linear',align_corners=False)
    z=z*(.8+.4*torch.rand(len(x),1,1,device=x.device))
    z=z+.05*torch.randn_like(z)
    return z*(torch.rand(len(x),12,1,device=x.device)>.1)


def contrastive_loss(a,b,temperature=.2):
    z=F.normalize(torch.cat((a,b)),dim=1)
    sim=z@z.T/temperature
    sim.fill_diagonal_(-torch.inf)
    target=(torch.arange(len(z),device=z.device)+len(a))%len(z)
    return F.cross_entropy(sim,target)


def train():
    protocol=dict(epochs=20,batch=128,seed=SEED,lr=.001,temperature=.2,width=64,blocks=4,
                  output=128,fit='internal fit patients only',selection='fixed-noise development contrastive loss',
                  augmentations=['800-sample crop resized to 1000','amplitude .8–1.2','Gaussian sigma .05','lead dropout .1'])
    register('phaseA_encoder',protocol)
    seed_all(SEED)
    x,_,_,fit,dev=training_data()
    folder=ART/'encoder_ssl'
    with gpu_lease('phaseA_encoder') as device:
        model=Encoder().to(device); opt=torch.optim.AdamW(model.parameters(),lr=.001)
        rng=np.random.default_rng(SEED); history=[]; start=0; best=float('inf')
        if (folder/'last.pt').exists():
            ck=torch.load(folder/'last.pt',map_location=device,weights_only=False)
            if ck['protocol'] != protocol:
                raise ValueError('Resume protocol differs from saved encoder protocol')
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            start,history,best=ck['epoch'],ck['history'],ck['best']
            rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu())
            torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        t0=time.monotonic()
        for epoch in range(start,20):
            model.train(); losses=[]
            order=rng.permutation(fit)
            for i in range(0,len(order),128):
                batch=torch.tensor(np.asarray(x[order[i:i+128]]),device=device).transpose(1,2)
                opt.zero_grad(set_to_none=True)
                loss=contrastive_loss(model.projection(model(augment(batch))),model.projection(model(augment(batch))))
                if not torch.isfinite(loss):raise FloatingPointError('SSL loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            model.eval(); validation=[]
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(SEED+1)
                with torch.no_grad():
                    for i in range(0,len(dev),128):
                        batch=torch.tensor(np.asarray(x[dev[i:i+128]]),device=device).transpose(1,2)
                        validation.append(float(contrastive_loss(model.projection(model(augment(batch))),model.projection(model(augment(batch))))))
            value=float(np.mean(validation)); improved=value<best;best=min(best,value)
            history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),dev_loss=value))
            payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,history=history,best=best,
                         numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),protocol=protocol)
            save_checkpoint(folder/'last.pt',payload)
            if improved:save_checkpoint(folder/'best.pt',payload)
            write_json(RESULTS/'encoder_ssl_training.json',dict(protocol=protocol,history=history,
                elapsed_seconds=time.monotonic()-t0,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
            print(history[-1],flush=True)
            if (ART/'STOP').exists():return


class SSLEmbedding:
    def __init__(self):
        self.model=Encoder().eval().requires_grad_(False)
        ck=torch.load(ART/'encoder_ssl/best.pt',map_location='cpu',weights_only=False)
        self.model.load_state_dict(ck['model'])
    @torch.no_grad()
    def __call__(self,x):
        return np.concatenate([self.model(torch.tensor(np.asarray(x[i:i+64])).transpose(1,2)).numpy()
                               for i in range(0,len(x),64)])


if __name__=='__main__':train()
