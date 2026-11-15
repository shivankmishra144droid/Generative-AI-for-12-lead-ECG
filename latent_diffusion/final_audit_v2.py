"""Read-only V1 verification and V2-only audit outputs. Never invokes V1 writers."""
import ast
import csv
import json
import importlib.metadata
import os
import platform
import subprocess
import sys
from .common import HOME,ROOT,digest,write_json
from .v2_common import RESULTS,AUDITS,ART,check_v1,now,gpu_lease


def run():
    commands=[
        [sys.executable,'-m','pytest','latent_diffusion','-q','-k','not empirical_gate_a','-o',f'cache_dir={ART / "pytest_cache"}'],
        [sys.executable,'-m','pytest','tests/test_study2.py','tests/test_study3.py',
         'tests/test_study4.py','tests/test_study4_features.py','tests/test_selection_and_rhythm.py',
         '-q','-o',f'cache_dir={ART / "pytest_cache"}']]
    checks=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    with gpu_lease('phaseA_regression_tests'):
        for command in commands:
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
            checks.append(dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
            print(result.stdout,flush=True)
    initial=json.loads((HOME/'initial_hashes.json').read_text())
    changed=[p for p,h in initial.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]
    v1_changed=check_v1()
    files=list(HOME.glob('*v2*.py'))+list((HOME/'latent').glob('*rhythm*.py'))
    prohibited=[]
    # Build forbidden strings here to keep the final static scan itself unambiguous.
    forbidden=['X_'+'test','y_'+'test']
    for path in files:
        source=path.read_text(encoding='utf-8');ast.parse(source)
        if any(s in source for s in forbidden):prohibited.append(str(path.relative_to(HOME)))
    gate_path=RESULTS/'phaseA_gate.json'
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else {'passed':False,'failures':['not yet evaluated']}
    # Stored V1 consistency check only: rerunning its benchmark would reopen validation.
    v1=json.loads((HOME/'artifacts/scorecard.json').read_text())['results']
    with (HOME/'results/scorecard_summary.csv').open(newline='') as stream:summary=list(csv.DictReader(stream))
    spotchecks=[]
    for name in ('hybrid','latent_diffusion'):
        row=next(r for r in summary if r['generator']==name and r['rhythm']=='SR')
        stored=v1[name]
        ok=abs(float(row['judge_auc'])-stored['feature_judge']['auc_mean'])<1e-12
        ok=ok and abs(float(row['coverage'])-stored['by_class']['SR']['prdc']['coverage'])<1e-12
        spotchecks.append(dict(generator=name,rhythm='SR',stored_csv_matches_stored_json=ok,
            note='Historical V1 benchmark not rerun because it loads validation; hashes preserve reproducibility'))
    from .generate import optional_dependencies
    optional_dependencies()
    versions={}
    for package in ('torch','numpy','scipy','scikit-learn','matplotlib','neurokit2','pytest'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not installed'
    result=dict(timestamp_utc=now(),regression_checks=checks,baseline_files_checked=len(initial),
        changed_preexisting_files=changed,v1_changed_files=v1_changed,prohibited_split_references=prohibited,
        syntax_files=len(files),code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},
        v1_spotchecks=spotchecks,
        software_checks_pass=all(c['exit_code']==0 for c in checks) and all(s['stored_csv_matches_stored_json'] for s in spotchecks) and not changed and not v1_changed and not prohibited,
        experimental_gate_a_pass=gate['passed'],experimental_gate_a_failures=gate.get('failures',[]),
        distinction='Software regression success does not override a failed empirical acceptance gate',
        environment=dict(python=sys.version,platform=platform.platform(),packages=versions))
    write_json(RESULTS/'final_audit.json',result)
    return result


if __name__=='__main__':run()
