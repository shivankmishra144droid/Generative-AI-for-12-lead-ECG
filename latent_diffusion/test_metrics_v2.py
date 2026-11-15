"""Algorithm tests plus preregistered empirical A8 acceptance tests.

Unit tests are not substitutes for acceptance tests on development ECGs.
"""
import json
import numpy as np
import pytest
from .gen_eval_v2 import (prdc_uncertainty,privacy_uncertainty,resamples,interval,
                          feature_judges,draw_clusters,clusters)
from .gen_eval import prdc
from .v2_common import RESULTS


def test_distance_bootstrap_point_matches_frozen_prdc():
    rng=np.random.default_rng(72);a=rng.normal(size=(35,8));b=rng.normal(size=(35,8))
    out=prdc_uncertainty(a,b,np.arange(35),np.arange(35),replicates=20)
    base=prdc(a,b)
    for k in out:assert out[k]['value']==pytest.approx(base[k])


def test_patient_clusters_and_paired_indices():
    groups=np.repeat(np.arange(7),[1,2,3,1,2,3,1])
    for a,b in resamples(groups,groups,20,paired=True):
        assert np.array_equal(a,b)
        for g in np.unique(groups):
            counts=[(a==i).sum() for i in np.flatnonzero(groups==g)]
            assert len(set(counts))==1
    with pytest.raises(ValueError):list(resamples(groups,groups[::-1],2,paired=True))


def test_exact_copies_and_collapse():
    rng=np.random.default_rng(3);fit=rng.normal(size=(50,8));real=rng.normal(size=(50,8))
    out=privacy_uncertainty(fit,real,fit,np.arange(50),np.arange(50),np.arange(50),replicates=20)
    assert out['ratio_of_medians']['value']<1e-6
    assert out['membership_auc']['value']>.9
    collapse=np.repeat(fit[:1],50,axis=0)
    p=prdc(real,collapse)
    assert p['recall']<.1


def bootstrap_sanity():
    # Fixed synthetic distribution checks CI code, not a claim of universal coverage.
    rng=np.random.default_rng(9200);a=rng.normal(size=2048)
    outcomes={}
    for n in (512,1024):
        sample=a[:n];brng=np.random.default_rng(9201)
        draws=[sample[brng.integers(n,size=n)].mean() for _ in range(500)]
        outcomes[str(n)]=interval(sample.mean(),draws)
    small,large=outcomes['512'],outcomes['1024']
    inside=all(v['ci95'][0]<=v['value']<=v['ci95'][1] for v in outcomes.values())
    shrinks=np.ptp(large['ci95'])<np.ptp(small['ci95'])
    return dict(passed=bool(inside and shrinks),point_inclusion=inside,width_shrinks=bool(shrinks),results=outcomes)


def test_bootstrap_sanity():assert bootstrap_sanity()['passed']


def test_misaligned_inputs_rejected():
    a=np.zeros((12,3))
    with pytest.raises(ValueError,match='alignment'):
        prdc_uncertainty(a,a,np.arange(11),np.arange(12),replicates=5)
    with pytest.raises(ValueError,match='alignment'):
        feature_judges(np.zeros((12,25)),np.zeros((12,25)),np.zeros(12,int),np.zeros(12,int),np.arange(11),np.arange(13))


@pytest.mark.xfail(strict=True, reason='Records that the ORIGINAL Gate A failed (results/v2/phaseA_gate.json); '
                   'it was recalibrated in audits/v2/amendment1.md (amendment 1, results/v2/phaseA_gate_amended.json).')
def test_empirical_gate_a():
    path=RESULTS/'phaseA_gate.json'
    if not path.exists():pytest.skip('Empirical development controls have not run')
    gate=json.loads(path.read_text())
    assert gate['passed'], 'Empirical Gate A failed; inspect phaseA_gate.json. Do not start Phase B.'
