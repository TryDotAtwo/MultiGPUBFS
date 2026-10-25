import json
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import pytest
from multigpubfs import recovery_registry as reg
from multigpubfs import recovery_diagnostics as rd
from multigpubfs.cayleypy_backend import NativeOptions

IDENTITY={'server':{'host':'node','boot':'a','environment_sha256':'x'},'code_sha256':'code','runtime':{'sha':'native'}}
@pytest.fixture(autouse=True)
def isolated(tmp_path,monkeypatch):
    monkeypatch.setenv('MGBFS_RECOVERY_CACHE',str(tmp_path/'cache'))
    monkeypatch.delenv('NCCL_P2P_DISABLE',raising=False)
    monkeypatch.setattr(rd,'fingerprint',lambda:IDENTITY)
    yield
    reg.restore()

@pytest.fixture
def bundle(tmp_path,monkeypatch):
    p=tmp_path/'source';p.mkdir()
    proof={'status':'PASS','cleanup_complete':True,'policy':{'NCCL_P2P_DISABLE':'1'},'result':{'status':'PASS','data':{'devices':[0,1],'cpu_oracles_verified':True,'measured_autotune':True,'prefix_verified':True}}}
    (p/'recovery-proof.json').write_text(json.dumps(proof))
    import hashlib
    profile={'schema':1,'status':'VERIFIED','devices':[0,1],'policy':proof['policy'],'identity':IDENTITY,'proof':{'path':'recovery-proof.json','sha256':hashlib.sha256((p/'recovery-proof.json').read_bytes()).hexdigest()}}
    (p/'recovery-profile.json').write_text(json.dumps(profile))
    api={'status':'PASS','cleanup_complete':True,'result':{'status':'PASS','data':{'devices':[0,1],'api_verified':True,'persistent_reuse':True}}}
    return p/'recovery-profile.json',api

def test_missing_cache_does_not_probe(monkeypatch):
    monkeypatch.setattr(rd,'fingerprint',lambda:pytest.fail('unexpected hardware query'))
    assert reg.activate(NativeOptions()).devices is None

def test_verified_roundtrip_and_restore(bundle):
    import os
    path,api=bundle
    cached=reg.install(path,api)
    assert cached.is_file()
    options=reg.activate(NativeOptions(output_root='mine',max_seconds=123))
    assert options.devices==(0,1) and options.max_seconds==123 and options.output_root=='mine'
    assert os.environ['NCCL_P2P_DISABLE']=='1'
    assert reg.activate(options)==options
    reg.restore();assert 'NCCL_P2P_DISABLE' not in os.environ

def test_corrupt_proof_rejected(bundle):
    path,api=bundle;cached=reg.install(path,api)
    (cached.parent/'recovery-proof.json').write_text('{}')
    with pytest.warns(RuntimeWarning,match='recovery'):
        assert reg.activate(NativeOptions()).devices is None

def test_unverified_api_never_installed(bundle):
    path,api=bundle;api['result']['data']['persistent_reuse']=False
    with pytest.raises(RuntimeError):reg.install(path,api)

def test_changed_node_does_not_apply(bundle,monkeypatch):
    path,api=bundle;reg.install(path,api)
    monkeypatch.setattr(rd,'fingerprint',lambda:dict(IDENTITY,server={'host':'other'}))
    assert reg.activate(NativeOptions()).devices is None

def test_explicit_gpu_selection_not_overridden(bundle):
    path,api=bundle;reg.install(path,api)
    with pytest.warns(RuntimeWarning,match='devices'):
        assert reg.activate(NativeOptions(devices=(0,))).devices==(0,)

def test_live_workers_rejected(bundle,monkeypatch):
    path,api=bundle;reg.install(path,api)
    from multigpubfs import generic_session
    monkeypatch.setattr(generic_session,'_active',object())
    with pytest.raises(RuntimeError,match='WORKERS'):reg.activate(NativeOptions())

def test_restore_preserves_user_change(bundle,monkeypatch):
    import os
    path,api=bundle;reg.install(path,api);reg.activate(NativeOptions())
    monkeypatch.setenv('NCCL_P2P_DISABLE','0');reg.restore()
    assert os.environ['NCCL_P2P_DISABLE']=='0'


def test_public_enable_hook_and_optout(monkeypatch):
    import multigpubfs.cayleypy_backend as backend
    from multigpubfs import native_distribution
    monkeypatch.setattr(native_distribution,'prepare_for_activation',lambda:None)
    seen=[]
    monkeypatch.setattr(reg,'activate',lambda options:(seen.append(options) or replace(options,devices=(0,1))))
    try:
        backend.enable_native();assert len(seen)==1
        backend.disable_native()
        backend.enable_native(recovery=False);assert len(seen)==1
    finally:backend.disable_native()

def test_activation_failure_rolls_back(bundle,monkeypatch):
    import os
    import multigpubfs.cayleypy_backend as backend
    from multigpubfs import native_distribution
    path,api=bundle;reg.install(path,api)
    monkeypatch.setattr(native_distribution,'prepare_for_activation',lambda:(_ for _ in ()).throw(RuntimeError('loader failed')))
    with pytest.raises(RuntimeError,match='loader failed'):backend.enable_native()
    assert reg._applied is None and 'NCCL_P2P_DISABLE' not in os.environ


def test_per_call_device_override_rejected(bundle):
    path,api=bundle;reg.install(path,api);reg.activate(NativeOptions())
    with pytest.raises(RuntimeError,match='DEVICES_CHANGED'):reg.bind_options(NativeOptions(devices=(0,)))

def test_concurrent_publication_is_complete(bundle):
    from concurrent.futures import ThreadPoolExecutor
    path,api=bundle
    with ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(lambda _:reg.install(path,api),range(8)))
    assert len(set(paths))==8
    assert reg.activate(NativeOptions()).devices==(0,1)


def test_optout_restores_applied_environment(bundle):
    import os
    path,api=bundle;reg.install(path,api);reg.activate(NativeOptions())
    reg.opt_out();assert reg._applied is None and 'NCCL_P2P_DISABLE' not in os.environ
