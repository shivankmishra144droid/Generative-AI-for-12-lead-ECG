"""Study-owned I/O, reproducibility and audit helpers; never writes elsewhere."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'latent_diffusion'
ART = HOME / 'artifacts'
DATA = ROOT / 'processed_data'
CLASSES = ('SR', 'AFIB', 'STACH', 'SBRAD')
SEED = 9009


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(HOME.resolve()):
        raise ValueError('Study 9 outputs must stay under latent_diffusion')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(phase, evidence):
    initial = json.loads((HOME / 'initial_hashes.json').read_text())
    changed = [p for p, h in initial.items() if not (ROOT / p).exists() or digest(ROOT / p) != h]
    result = dict(phase=phase, existing_files_changed_since_snapshot=changed,
                  evidence=evidence, versions=dict(torch=torch.__version__, numpy=np.__version__),
                  note='Concurrent changes are reported, never reverted. Study 9 writes new paths only.')
    write_json(HOME / 'audits' / f'{phase}.json', result)
    return result


def load_split(split):
    if split not in ('train', 'val'):
        raise ValueError('Study 9 deliberately has no test-split loader')
    x = np.load(DATA / f'X_{split}.npy', mmap_mode='r')
    y = np.load(DATA / f'y_{split}.npy').astype(np.int64)
    if x.shape != (len(y), 1000, 12) or not np.isin(y, range(4)).all():
        raise ValueError('Unexpected waveform shape/label mapping')
    return x, y


def save_checkpoint(path, payload):
    path = Path(path).resolve()
    if not path.is_relative_to(ART.resolve()):
        raise ValueError('Checkpoint outside Study 9 artifacts')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def balanced_rows(y, n, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == c), min(n, int((y == c).sum())), replace=False)
                           for c in range(4)])
