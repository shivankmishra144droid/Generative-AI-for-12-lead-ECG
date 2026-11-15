"""Isolated NeuroKit ECGSYN adapter: use its supported NumPy RR interpolation.

With scipy 1.17.1 the default ndimage.zoom path can put a zero at the last
upsampled RR sample. The upstream RR accumulation loop then never advances.
Clone the two functions with local globals; never patch a module or package file.
The ECGSYN equations, twelve lead coefficients and integration are unchanged.
"""
from __future__ import annotations

import importlib
import types

import numpy as np


def positive_rr_resample(signal, sampling_rate, desired_sampling_rate, **kwargs):
    from neurokit2.signal.signal_resample import signal_resample
    output = signal_resample(signal, sampling_rate=sampling_rate,
                             desired_sampling_rate=desired_sampling_rate, method='numpy')
    if not np.isfinite(output).all() or np.min(output) < .02:
        raise ValueError('Invalid RR interpolation; refusing an unbounded ECGSYN loop')
    return output


def _clone(fn, replacements):
    clone = types.FunctionType(fn.__code__, {**fn.__globals__, **replacements}, fn.__name__,
                                fn.__defaults__, fn.__closure__)
    clone.__kwdefaults__ = fn.__kwdefaults__
    return clone


def simulate(**kwargs):
    module = importlib.import_module('neurokit2.ecg.ecg_simulate')
    generator = _clone(module._ecg_simulate_ecgsyn, {'signal_resample':positive_rr_resample})
    public = _clone(module.ecg_simulate, {'_ecg_simulate_ecgsyn':generator})
    return public(**kwargs)
