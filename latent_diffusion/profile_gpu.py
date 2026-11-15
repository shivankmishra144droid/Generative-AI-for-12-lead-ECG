"""Measured larger-latent-backbone training-step fit, including Adam state allocation."""
import gc
import time

import torch

from latent_diffusion.latent.latent_diffusion import LatentDiffusion, LatentSSSD
from .common import ART, audit, seed_all, write_json


def run():
    seed_all()
    if not torch.cuda.is_available():
        write_json(ART / 'gpu_profiles.json', {'status':'CUDA unavailable'})
        return
    rows = []
    for layers, width in [(4, 32), (8, 64)]:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = LatentDiffusion(LatentSSSD(layers=layers, width=width)).cuda()
        opt = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        torch.cuda.synchronize(); start = time.monotonic()
        loss = model.loss(torch.randn(8, 16, 125, device='cuda'), torch.arange(8, device='cuda') % 4,
                          torch.zeros(8, 6, device='cuda'))
        loss.backward(); opt.step(); torch.cuda.synchronize()
        rows.append(dict(layers=layers, width=width, batch=8, latent_shape=[16, 125],
                         parameters=sum(p.numel() for p in model.parameters()), loss=float(loss.detach()),
                         training_step_seconds=time.monotonic()-start,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        del model, opt, loss
    result = dict(device=torch.cuda.get_device_name(0), total_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
                  rows=rows, note='One optimizer step including backward and Adam-state allocation, not a convergence experiment; no decoder/EMA in this profile')
    write_json(ART / 'gpu_profiles.json', result)
    audit('phase3_gpu_capacity', result)
    print(result)


if __name__ == '__main__':
    run()
