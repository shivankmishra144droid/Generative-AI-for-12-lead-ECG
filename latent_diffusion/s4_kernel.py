"""S4 Cauchy-kernel selection without editing the frozen synthetic_generation/s4.py (amendment 6).

s4.py's fast paths (CUDA extension, pykeops) sum over conjugate pairs; its fallback `cauchy_slow` sums only the
stored half. A checkpoint is only valid with the kernel it was trained with:
  laptop-trained (cs, ds, earlier runs): fallback as-is        -> no patch
  Colab/pykeops-trained (cm):            conjugate-pair sum    -> `with conj_kernel():`
"""
import contextlib
import torch


@contextlib.contextmanager
def conj_kernel():
    import synthetic_generation.s4 as s4
    if getattr(s4, 'has_pykeops', False) or getattr(s4, 'has_cauchy_extension', False):
        yield                                    # the fast path already sums conjugate pairs
        return
    slow = s4.cauchy_slow
    s4.cauchy_slow = lambda v, z, w: slow(torch.cat([v, v.conj()], -1), z, torch.cat([w, w.conj()], -1))
    try:
        yield
    finally:
        s4.cauchy_slow = slow


def assert_fallback_kernel():
    """Laptop-trained checkpoints must run on the fallback; refuse if a conjugate-pair fast path is active."""
    import synthetic_generation.s4 as s4
    if getattr(s4, 'has_pykeops', False) or getattr(s4, 'has_cauchy_extension', False):
        raise RuntimeError('pykeops/CUDA Cauchy kernel active: laptop-trained checkpoints (cs, ds) would be wrong here')
