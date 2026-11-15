"""V2-only paths, preregistration, isolation and exclusive GPU lease."""
import contextlib
import datetime
import json
import os
import subprocess
import csv
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import ROOT, HOME, ART as V1_ART, digest, write_json, seed_all

ART = V1_ART / 'v2'
RESULTS = HOME / 'results/v2'
AUDITS = HOME / 'audits/v2'
SEED = 9200


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def setup():
    for p in (ART, RESULTS, AUDITS):
        p.mkdir(parents=True, exist_ok=True)


def training_data():
    x = np.load(ROOT / 'processed_data/X_train.npy', mmap_mode='r')
    y = np.load(ROOT / 'processed_data/y_train.npy')
    patient = pd.read_csv(ROOT / 'processed_data/train_metadata.csv').patient_id.to_numpy()
    part = np.load(V1_ART / 'partition.npz')
    assert not set(patient[part['fit']]) & set(patient[part['dev']])
    return x, y, patient, part['fit'], part['dev']


def register(phase, protocol):
    setup()
    p = AUDITS / f'{phase}_registration.json'
    if p.exists():
        old = json.loads(p.read_text())
        if old['protocol'] != protocol:
            raise ValueError('Protocol changed; write a numbered amendment before rerunning')
        return
    write_json(p, dict(registered_utc=now(), protocol=protocol,
                       instructions_sha256=digest(HOME/'STUDY9_V2_INSTRUCTIONS.md')))


def snapshot_v1():
    setup()
    target = AUDITS/'v1_snapshot.json'
    if target.exists():
        return
    files = list(HOME.glob('*.py')) + list((HOME/'latent').glob('*.py'))
    files = [p for p in files if 'v2' not in p.stem]
    files += [HOME/'REPORT.md', HOME/'README.md', HOME/'PLAN.md']
    for folder in (HOME/'results', HOME/'audits', V1_ART):
        files += [p for p in folder.rglob('*') if p.is_file() and 'v2' not in p.relative_to(folder).parts
                  and 'deps' not in p.parts and '__pycache__' not in p.parts
                  and p.suffix in ('.json','.csv','.md','.pt','.npz','.png')]
    write_json(target, {p.relative_to(ROOT).as_posix():digest(p) for p in files})


def check_v1():
    saved=json.loads((AUDITS/'v1_snapshot.json').read_text())
    return [p for p,h in saved.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]


@contextlib.contextmanager
def gpu_lease(phase):
    setup()
    if (ART/'STOP').exists():raise InterruptedError('STOP before GPU phase')
    lock=ART/'gpu.lock'
    while True:
        output=subprocess.run(['nvidia-smi'], capture_output=True,text=True,check=True).stdout
        with (AUDITS/f'{phase}_nvidia_smi.txt').open('a') as log:
            log.write(now()+'\n'+output)
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if os.name=='nt':
            inventory=subprocess.run(['tasklist','/FO','CSV','/NH'],capture_output=True,text=True,check=True).stdout
            names={int(row[1]):row[0].lower() for row in csv.reader(io.StringIO(inventory)) if len(row)>1 and row[1].isdigit()}
        else:   # Linux (Colab): same pid -> name map from ps
            inventory=subprocess.run(['ps','-eo','pid=,comm='],capture_output=True,text=True,check=True).stdout
            names={int(p):n.lower() for p,n in (l.split(None,1) for l in inventory.splitlines() if l.strip()) if p.isdigit()}
        # WDDM lists desktop composition/browser graphics as compute applications.
        # Only known UI processes are exempt; unknown GPU workloads still block.
        ui={'dwm.exe','brave.exe','chrome.exe','msedge.exe'}
        others=[int(s.strip()) for s in query.stdout.splitlines() if s.strip().isdigit()
                and int(s.strip())!=os.getpid() and names.get(int(s.strip())) not in ui]
        if os.environ.get('V2_SINGLE_GPU_HOST')=='1':others=[]   # dedicated VM (Colab): skip foreign-process check, keep the lock
        if not others and lock.exists():
            owner=int(lock.read_text().strip())
            if owner in names:
                others=[owner]
            else:
                # A crashed worker cannot release its lock. Never remove a live owner's lock.
                with (AUDITS/f'{phase}_nvidia_smi.txt').open('a') as log:
                    log.write(f'Recovered stale V2 GPU lease from exited PID {owner}\n')
                lock.unlink()
        if not others:
            try:
                with lock.open('x') as f:f.write(str(os.getpid()))
                break
            except FileExistsError:
                continue
        if (ART/'STOP').exists():raise InterruptedError('STOP while waiting for GPU')
        print(f'Waiting for GPU processes {others}',flush=True)
        time.sleep(20)
    if not torch.cuda.is_available():
        lock.unlink()
        raise RuntimeError('CUDA unavailable; training stopped, no CPU fallback')
    try:
        torch.cuda.reset_peak_memory_stats()
        yield torch.device('cuda')
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


def save_checkpoint(path, payload):
    if not path.resolve().is_relative_to(ART.resolve()):
        raise ValueError('V2 checkpoint outside V2')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)
