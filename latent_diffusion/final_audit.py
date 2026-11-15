"""Senior-review checklist, regression checks and immutable-file verification."""
import ast
import json
import platform
import subprocess
import sys

from .common import ART, HOME, ROOT, audit, digest, write_json


def run():
    commands = [
        [sys.executable, '-m', 'pytest', 'latent_diffusion/test_models.py', 'latent_diffusion/test_metrics.py', 'latent_diffusion/test_integration.py', '-q'],
        [sys.executable, '-m', 'pytest', 'tests', '-q']]
    checks = []
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)
        checks.append(dict(command=command, exit_code=proc.returncode, output=proc.stdout))
    code = list(HOME.glob('*.py')) + list((HOME / 'latent').glob('*.py'))
    for path in code:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    initial = json.loads((HOME / 'initial_hashes.json').read_text())
    changed = [p for p, h in initial.items() if not (ROOT / p).exists() or digest(ROOT / p) != h]
    required = ['data_contract.json', 'reconstruction.json', 'ae/training.json', 'vae/training.json',
                'diffusion/training.json', 'scorecard.json', 'utility.json', 'control_sweep.json', 'published_access.json', 'gpu_profiles.json']
    missing = [name for name in required if not (ART / name).exists()]
    scorecard = json.loads((ART / 'scorecard.json').read_text()) if (ART / 'scorecard.json').exists() else {}
    unscored_local = [name for name in ('real_control','hybrid','vae','latent_diffusion','ecgsyn')
                     if scorecard.get('sources', {}).get(name, {}).get('status') != 'scored']
    utility = json.loads((ART / 'utility.json').read_text()) if (ART / 'utility.json').exists() else {}
    missing_utility = [name for name in ('TRTR', 'TSTR_hybrid', 'augment_hybrid', 'TSTR_vae', 'augment_vae',
                                        'TSTR_latent_diffusion', 'augment_latent_diffusion')
                       if name not in utility.get('results', {})]
    result = dict(test_commands=checks, syntax_files=len(code), baseline_files_checked=len(initial),
                  changed_existing_files=changed, missing_deliverables=missing,
                  unscored_local_baselines=unscored_local, missing_utility_conditions=missing_utility,
                  checks_pass=all(c['exit_code'] == 0 for c in checks) and not changed and not missing and not unscored_local and not missing_utility,
                  environment=dict(python=sys.version, platform=platform.platform()),
                  code_sha256={str(p.relative_to(ROOT)).replace('\\', '/'):digest(p) for p in code},
                  review=[
                      'New files only; no existing modules patched or frozen weights retrained.',
                      'Training and latent normalization use internal fit; internal development is patient-disjoint.',
                      'All output paths are Study 9; test split has no loader.',
                      'Cosine/v inverse/null CFG/DDIM/checkpoint/PRDC/copy controls tested.',
                      'Missing clinical features retain denominators; no silent imputation for Wasserstein.',
                      'Unpaired generated ECGs have peak availability, never claimed beat sensitivity.',
                      'Privacy uses class-matched members/nonmembers and disjoint reference patients; limited release attack only.',
                      'Published SSSD overlap excluded from utility/privacy; ECGSYN AFIB unsupported.',
                      'Synthetic benchmarks are exploratory, training budgets are not matched, no clinical efficacy claim.',
                      'Existing source hashes checked after each phase, no concurrent changes reverted.'
                  ])
    write_json(HOME / 'results/final_audit.json', result)
    audit('phase6_final', result)
    if not result['checks_pass']:
        raise RuntimeError('Final audit has failures or incomplete deliverables; see results/final_audit.json')


if __name__ == '__main__':
    run()
