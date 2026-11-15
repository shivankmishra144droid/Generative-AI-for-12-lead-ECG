"""Sequential, logged Study 9 reproduction. Uses the Python that invokes it.

py -3.14 -m latent_diffusion.run_all --resume
Run from the repository root. All generated files stay in latent_diffusion.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from .common import ART, HOME, ROOT, write_json


def invoke(module, args=()):
    command = [sys.executable, '-m', module, *map(str, args)]
    logdir = ART / 'logs'; logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / (module.replace('.', '_')+'.log')
    with path.open('a', encoding='utf-8') as log:
        log.write('\nCOMMAND: '+repr(command)+'\n'); log.flush()
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as proc:
            for line in proc.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = proc.wait()
    if code:
        raise RuntimeError(f'{module} failed ({code}); inspect {path}')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--skip-download', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.steps) < 1 or args.n < 6:
        p.error('Positive training budget and at least six samples/class required')
    resume = ['--resume'] if args.resume else []
    if not (HOME / 'initial_hashes.json').exists():
        raise RuntimeError('The isolation snapshot must exist before execution')
    invoke('pytest', ['latent_diffusion/test_models.py', 'latent_diffusion/test_metrics.py', 'latent_diffusion/test_integration.py', '-q'])
    invoke('latent_diffusion.data')
    invoke('latent_diffusion.train', ['ae', '--epochs', args.epochs, '--batch', 32, *resume])
    invoke('latent_diffusion.reconstruction')
    invoke('latent_diffusion.profile_gpu')
    invoke('latent_diffusion.train', ['diffusion', '--steps', args.steps, '--batch', 8, *resume])
    invoke('latent_diffusion.train', ['ae', '--name', 'vae', '--beta', .01, '--epochs', args.epochs, '--batch', 32, *resume])
    for kind in ('vae', 'latent_diffusion'):
        invoke('latent_diffusion.generate', [kind, '--n', args.n])
    deps = ART / 'deps'
    # Optional packages are installed locally; existing Python environments are unchanged.
    install = [sys.executable, '-m', 'pip', 'install', '--target', str(deps), '--no-deps',
               'neurokit2==0.2.13', 'PyWavelets==1.10.0']
    subprocess.run(install, cwd=ROOT, check=True)
    invoke('latent_diffusion.generate', ['ecgsyn', '--n', args.n])
    invoke('latent_diffusion.generate', ['control', '--n', 16])
    if not args.skip_download:
        invoke('latent_diffusion.published', ['download'])
    invoke('latent_diffusion.benchmark', ['--n', args.n, '--qt'])
    invoke('latent_diffusion.utility', ['--n', args.n])
    invoke('latent_diffusion.final_audit')
    invoke('latent_diffusion.report')


if __name__ == '__main__':
    main()
