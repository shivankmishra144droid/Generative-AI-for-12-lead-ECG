"""Download/import the published SSSD arrays without executing external code.

The upstream Figshare share link currently responds 403/202 on this machine.
This command logs the exact response, rejects HTML/empty responses, and never
turns an unavailable external baseline into a fabricated result.

Import requires the documented statement order, record alignment and mV scale:
python -m latent_diffusion.published import --waveforms <npy> --labels <npy>
    --statements <json-list> --lead-order <json-list> --verified-mv
    --provenance <json-file>
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .common import ART, CLASSES, DATA, HOME, audit, digest, write_json

URL = 'https://figshare.com/ndownloader/files/38890965?private_link=43df16e4a50e4dd0a0c5'


def download():
    folder = HOME / 'downloads'
    folder.mkdir(exist_ok=True)
    target = folder / 'Dataset.zip'
    attempts = []
    urls = [URL, 'https://ndownloader.figshare.com/files/38890965',
            'https://api.figshare.com/v2/file/download/38890965']
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'latent_diffusion research benchmark'})
            with urllib.request.urlopen(req, timeout=30) as response:
                record = dict(url=url, status=response.status, content_type=response.headers.get('Content-Type'),
                               content_length=response.headers.get('Content-Length'))
                attempts.append(record)
                prefix = response.read(4)
                if response.status != 200 or prefix != b'PK\x03\x04':
                    record['accepted'] = False
                    continue
                tmp = target.with_suffix('.partial')
                with tmp.open('wb') as f:
                    f.write(prefix)
                    for chunk in iter(lambda: response.read(2**20), b''):
                        f.write(chunk)
                if not zipfile.is_zipfile(tmp):
                    raise ValueError('Download is not a ZIP archive')
                tmp.replace(target)
                record.update(accepted=True, sha256=digest(target), bytes=target.stat().st_size)
                # Deliberately no extraction or downloaded Python execution.
                break
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            attempts.append(dict(url=url, error=str(exc)))
    result = dict(attempts=attempts, downloaded=target.exists(),
                  upstream='https://github.com/AI4HealthUOL/SSSD-ECG',
                  license='MIT as shown on linked Figshare dataset page; inspect archive metadata after retrieval',
                  status='requires format/provenance verification before scoring' if target.exists() else 'unavailable from upstream endpoints')
    write_json(ART / 'published_access.json', result)
    audit('phase5_published_access', result)
    print(json.dumps(result, indent=2))


def convert_arrays(x, labels, statements, lead_order, stats):
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError('Expected 3D waveform array')
    if x.shape[1] == 1000:
        pass
    elif x.shape[2] == 1000:
        x = x.transpose(0, 2, 1)
    else:
        raise ValueError('Expected 1000 samples at 100 Hz; no implicit resampling')
    if len(set(lead_order)) != len(lead_order) or x.shape[2] != len(lead_order):
        raise ValueError('Ambiguous lead order')
    labels = np.asarray(labels)
    if labels.shape != (len(x), len(statements)) or len(set(statements)) != len(statements):
        raise ValueError('Label shape/statement list mismatch')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(x).all():
        raise ValueError('Expected finite waveforms and binary multi-label statements')
    if any(c not in statements for c in CLASSES):
        raise ValueError('Missing rhythm statements; do not infer their column order')
    target = labels[:, [statements.index(c) for c in CLASSES]]
    keep = target.sum(1) == 1
    y = target[keep].argmax(1).astype(np.int64)
    x = x[keep]
    def lead(name):
        return x[:, :, lead_order.index(name)]
    i = lead('I')
    ii = lead('II') if 'II' in lead_order else lead('aVF') + i/2
    independent = np.stack([i, ii] + [lead(f'V{j}') for j in range(1, 7)], axis=1)
    indices = [0, 1, 6, 7, 8, 9, 10, 11]
    mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
    independent = (independent-mean[indices][None, :, None])/std[indices][None, :, None]
    from synthetic_generation.leads import reconstruct_12
    return reconstruct_12(independent.astype(np.float32)), y, np.flatnonzero(keep)


def import_pool(args):
    if not args.verified_mv:
        raise ValueError('Explicit physical-unit verification is required; do not guess scaling')
    provenance = json.loads(Path(args.provenance).read_text())
    for field in ('units_evidence', 'lead_order_evidence', 'statement_order_evidence', 'row_alignment_evidence', 'model_training_folds'):
        if not provenance.get(field):
            raise ValueError(f'Missing provenance: {field}')
    stats = json.loads((DATA / 'normalization_stats.json').read_text())
    x, y, indices = convert_arrays(np.load(args.waveforms, allow_pickle=False), np.load(args.labels, allow_pickle=False),
                                  json.loads(Path(args.statements).read_text()), json.loads(Path(args.lead_order).read_text()), stats)
    from .data import measure_hr
    hr = {}
    for c in range(4):
        rows = np.flatnonzero(y == c)[:64]
        if not len(rows):
            raise ValueError('Missing mapped class')
        hr[CLASSES[c]] = float(np.nanmedian([measure_hr(x[i]) for i in rows]))
    if not hr['SBRAD'] < hr['SR'] < hr['STACH']:
        raise ValueError(f'HR sanity check failed: {hr}')
    out = ART / 'pools/published_sssd.npz'; out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=x, y=y, external_row=indices)
    write_json(out.with_suffix('.json'), dict(provenance=provenance, hr_sanity=hr, n_by_class=np.bincount(y, minlength=4),
                                             inputs={f:digest(getattr(args, f)) for f in ('waveforms','labels','statements','lead_order','provenance')},
                                             utility_allowed=False, privacy_allowed=False))
    audit('phase5_published_import', dict(n=len(y), hr=hr, provenance=provenance))


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('download')
    imp = sub.add_parser('import')
    for field in ('waveforms', 'labels', 'statements', 'lead-order', 'provenance'):
        imp.add_argument('--'+field, required=True)
    imp.add_argument('--verified-mv', action='store_true')
    args = p.parse_args()
    download() if args.command == 'download' else import_pool(args)


if __name__ == '__main__':
    main()
