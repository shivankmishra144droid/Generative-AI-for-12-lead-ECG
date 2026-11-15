"""Patient-safe preparation and continuous conditioning. HR is a measured proxy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .common import ART, DATA, SEED, audit, load_split, write_json


def condition_values(hr, age, sex):
    """Six columns: centered HR/age, PTB-XL sex code, and three availability flags.

    Fixed physical scales avoid validation fitting. Age >100 (including PTB-XL's
    de-identification sentinel 300) is masked. Sex 0/1 remains the source coding.
    """
    values = np.asarray([hr, age, sex], dtype=np.float32).T.copy()
    valid = np.isfinite(values)
    valid[:, 0] &= (values[:, 0] >= 20) & (values[:, 0] <= 250)
    valid[:, 1] &= (values[:, 1] >= 0) & (values[:, 1] <= 100)
    valid[:, 2] &= np.isin(values[:, 2], [0, 1])
    values = (values - [75., 50., .5]) / [40., 25., .5]
    values[~valid] = 0
    return np.c_[values, valid].astype(np.float32)


def measure_hr(x):
    from synthetic_generation.beats import detect_r_peaks
    from synthetic_generation.leads import to_independent_8
    peaks = detect_r_peaks(to_independent_8(np.asarray(x, np.float32)[None])[0])
    return 6000. / np.diff(peaks).mean() if len(peaks) >= 3 else np.nan


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    summaries = {}
    patients = {}
    for split in ('train', 'val'):
        x, y = load_split(split)
        meta = pd.read_csv(DATA / f'{split}_metadata.csv')
        if len(meta) != len(y) or not np.array_equal(meta.label.to_numpy(), y):
            raise ValueError('Metadata not aligned with waveform labels')
        if meta.patient_id.isna().any() or meta.ecg_id.duplicated().any():
            raise ValueError('Missing patient identity or duplicate ECG identity')
        patients[split] = set(meta.patient_id)
        hr = []
        for start in range(0, len(x), 256):
            block = np.asarray(x[start:start+256])
            if not np.isfinite(block).all():
                raise ValueError('Non-finite ECG')
            hr.extend(measure_hr(a) for a in block)
        hr = np.asarray(hr)
        cond = condition_values(hr, meta.age.to_numpy(), meta.sex.to_numpy())
        np.savez_compressed(ART / f'{split}_conditions.npz', values=cond, hr=hr,
                            patient=meta.patient_id.to_numpy(), ecg_id=meta.ecg_id.to_numpy())
        if split == 'train':
            fit, dev = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED)
                            .split(y, y, groups=meta.patient_id))
            assert not set(meta.patient_id.iloc[fit]) & set(meta.patient_id.iloc[dev])
            if any(not np.any(y[rows] == c) for rows in (fit, dev) for c in range(4)):
                raise ValueError('An internal partition has a missing rhythm class')
            np.savez_compressed(ART / 'partition.npz', fit=fit, dev=dev)
        summaries[split] = dict(n=len(y), patients=len(patients[split]),
                                class_counts=np.bincount(y, minlength=4),
                                condition_available=cond[:, 3:].mean(0),
                                measured_hr_median_by_class=[np.nanmedian(hr[y == c]) for c in range(4)])
        print(split, summaries[split], flush=True)
    if patients['train'] & patients['val']:
        raise ValueError('Patient leakage between train and validation')
    write_json(ART / 'data_contract.json', summaries)
    audit('phase1_data', summaries)


if __name__ == '__main__':
    prepare()
