#!/usr/bin/env bash
# Phases C-G after C/D training (run from the repository root). Each step logs to artifacts/v2/logs/<step>.log.
set -e
L=latent_diffusion/artifacts/v2/logs
step() { echo "=== $1 $(date)"; py -3.14 -m "latent_diffusion.$2" $3 > "$L/$1.log" 2>&1; tail -3 "$L/$1.log"; }
step gates   dev_eval_v2 gates
step sweep   dev_eval_v2 sweep
step grid    dev_eval_v2 grid
step phaseF  phase_f_v2
step G_pools phase_g_v2 pools
step G_train phase_g_v2 train
step snaps   dev_eval_v2 snapshots
echo "=== all done $(date)"
