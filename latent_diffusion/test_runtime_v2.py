"""GPU lease and STOP regressions, with no actual GPU allocation."""
import os
from types import SimpleNamespace
import pytest
from . import v2_common as runtime


def environment(monkeypatch,tmp_path):
    art=tmp_path/'art';audits=tmp_path/'audit';results=tmp_path/'results'
    monkeypatch.setattr(runtime,'ART',art);monkeypatch.setattr(runtime,'AUDITS',audits);monkeypatch.setattr(runtime,'RESULTS',results)
    monkeypatch.setattr(runtime.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(runtime.torch.cuda,'reset_peak_memory_stats',lambda:None)
    runtime.setup()
    return art,audits


def test_stop_prevents_acquiring_gpu(monkeypatch,tmp_path):
    art,_=environment(monkeypatch,tmp_path);(art/'STOP').touch()
    def unexpected(*a,**k):raise AssertionError('GPU query must not start after STOP')
    monkeypatch.setattr(runtime.subprocess,'run',unexpected)
    with pytest.raises(InterruptedError):
        with runtime.gpu_lease('stop'):pass
    assert not (art/'gpu.lock').exists()


def test_stale_lease_recovers_and_releases_on_exception(monkeypatch,tmp_path):
    art,audits=environment(monkeypatch,tmp_path);(art/'gpu.lock').write_text('999999')
    def fake(command,**kwargs):
        if command[0]=='tasklist':return SimpleNamespace(stdout=f'"python.exe","{os.getpid()}"\n')
        return SimpleNamespace(stdout='' if '--query-compute-apps=pid' in command else 'GPU preflight')
    monkeypatch.setattr(runtime.subprocess,'run',fake)
    with pytest.raises(ValueError,match='worker failed'):
        with runtime.gpu_lease('stale'):
            assert (art/'gpu.lock').read_text()==str(os.getpid())
            raise ValueError('worker failed')
    assert not (art/'gpu.lock').exists()
    assert 'Recovered stale' in (audits/'stale_nvidia_smi.txt').read_text()


def test_other_worker_is_waited_for_not_terminated(monkeypatch,tmp_path):
    art,_=environment(monkeypatch,tmp_path);waits=[]
    def fake(command,**kwargs):
        if command[0]=='tasklist':return SimpleNamespace(stdout=f'"python.exe","{os.getpid()}"\n"python.exe","12345"\n')
        if '--query-compute-apps=pid' in command:return SimpleNamespace(stdout='12345\n' if not waits else '')
        return SimpleNamespace(stdout='GPU preflight')
    monkeypatch.setattr(runtime.subprocess,'run',fake);monkeypatch.setattr(runtime.time,'sleep',lambda t:waits.append(t))
    with runtime.gpu_lease('wait'):assert waits==[20]
    assert not (art/'gpu.lock').exists()
