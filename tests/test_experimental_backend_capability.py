import json, subprocess
import pytest
from multigpubfs import launch

def query_response(data, code=0):
    return subprocess.CompletedProcess(['mgbfs','key-info'],code,json.dumps(data),'old runtime' if code else '')

@pytest.mark.parametrize('backend',['segmented24','permutation_bitmap'])
def test_old_native_must_not_silently_run_legacy(tmp_path,monkeypatch,backend):
    binary=tmp_path/'mgbfs';binary.write_bytes(b'old')
    monkeypatch.setattr(launch,'native_query',lambda *a,**kw:query_response({'schema':1}))
    with pytest.raises(RuntimeError,match='NATIVE_MEMORY_BACKEND_UNAVAILABLE'):
        launch._require_memory_backend(str(binary),{},backend)

@pytest.mark.parametrize('backend,key',[('segmented24','segmented24_v1'),('permutation_bitmap','permutation_bitmap_v1')])
def test_matching_native_capability_can_be_cached(tmp_path,monkeypatch,backend,key):
    binary=tmp_path/'mgbfs';binary.write_bytes(b'new')
    monkeypatch.setattr('multigpubfs.autotune._dependency_identity', lambda *a,**kw: {
        'cuda_library_sha256':'verified-cuda', 'resolved_dependencies':(('libcuda','verified'),)})
    calls=[]
    def query(*a,**kw):
        calls.append((a,kw));return query_response({'schema':1,key:True,'segmented32_v2':True,'segmented_memory_presets_v1':True})
    monkeypatch.setattr(launch,'native_query',query)
    launch._require_memory_backend(str(binary),{'LD_LIBRARY_PATH':'/one'},backend)
    launch._require_memory_backend(str(binary),{'LD_LIBRARY_PATH':'/one'},backend)
    assert len(calls)==1
    launch._require_memory_backend(str(binary),{'LD_LIBRARY_PATH':'/other'},backend)
    assert len(calls)==2
    binary.write_bytes(b'newer binary')
    launch._require_memory_backend(str(binary),{'LD_LIBRARY_PATH':'/other'},backend)
    assert len(calls)==3

def test_bad_json_is_not_accepted(tmp_path,monkeypatch):
    binary=tmp_path/'mgbfs';binary.write_bytes(b'broken')
    monkeypatch.setattr(launch,'native_query',lambda *a,**kw:subprocess.CompletedProcess([],0,'bad',''))
    with pytest.raises(RuntimeError,match='NATIVE_MEMORY_BACKEND_UNAVAILABLE'):
        launch._require_memory_backend(str(binary),{},'segmented24')

@pytest.mark.parametrize('backend,receipt',[
    ('segmented24',{'backend':'GENERIC_DISTRIBUTED_EXACT'}),
    ('permutation_bitmap',{'backend':'EXPERIMENTAL_SEGMENTED24'}),
    ('segmented24',{'backend':'EXPERIMENTAL_PERMUTATION_BITMAP'}),
])
def test_receipt_must_match_requested_backend(backend,receipt):
    with pytest.raises(RuntimeError,match='NATIVE_MEMORY_BACKEND_RECEIPT_MISMATCH'):
        launch._check_memory_backend_receipt(receipt,backend)

def test_matching_receipts_are_accepted():
    launch._check_memory_backend_receipt({'backend':'EXPERIMENTAL_SEGMENTED24'},'segmented24')
    launch._check_memory_backend_receipt({'backend':'EXPERIMENTAL_PERMUTATION_BITMAP'},'permutation_bitmap')
    launch._check_memory_backend_receipt({'backend':'GENERIC'},'legacy')

@pytest.mark.parametrize('backend,key',[('segmented24','segmented24_v1'),('permutation_bitmap','permutation_bitmap_v1')])
def test_pre_32bit_native_is_rejected(tmp_path,monkeypatch,backend,key):
    binary=tmp_path/'mgbfs';binary.write_bytes(b'old ABI')
    monkeypatch.setattr(launch,'native_query',lambda *a,**kw:query_response({'schema':1,key:True}))
    with pytest.raises(RuntimeError,match='NATIVE_MEMORY_BACKEND_UNAVAILABLE'):
        launch._require_memory_backend(str(binary),{},backend)

@pytest.mark.parametrize('name',['segmented','segmented32','segmented24'])
def test_segmented_names_select_same_memory_backend(name):
    assert launch._normalize_memory_backend(name)=='segmented24'

def test_normalize_other_memory_backends():
    assert launch._normalize_memory_backend('legacy')=='legacy'
    assert launch._normalize_memory_backend('permutation_bitmap')=='permutation_bitmap'
    with pytest.raises(ValueError, match='INVALID_MEMORY_BACKEND'):
        launch._normalize_memory_backend('unknown')


def test_capability_cache_retains_exact_exchange_feature(tmp_path, monkeypatch):
    binary=tmp_path/'mgbfs';binary.write_bytes(b'exact')
    monkeypatch.setattr('multigpubfs.autotune._dependency_identity', lambda *a,**kw: {
        'cuda_library_sha256':'verified-cuda', 'resolved_dependencies':(('libcuda','verified'),)})
    calls=[]
    def query(*args,**kwargs):
        calls.append(1);return query_response({'schema':1,'segmented24_v1':True,'segmented32_v2':True,'segmented_memory_presets_v1':True,'segmented_exact_exchange_v1':True})
    monkeypatch.setattr(launch,'native_query',query)
    assert launch._require_memory_backend(str(binary),{},'segmented24')['segmented_exact_exchange_v1'] is True
    assert launch._require_memory_backend(str(binary),{},'segmented24')['segmented_exact_exchange_v1'] is True
    assert len(calls)==1

@pytest.mark.parametrize('backend,key',[('segmented24','segmented24_v1'),('permutation_bitmap','permutation_bitmap_v1')])
def test_pre_vram_preset_binary_cannot_silently_ignore_new_options(tmp_path,monkeypatch,backend,key):
    binary=tmp_path/'mgbfs';binary.write_bytes(b'abi2')
    monkeypatch.setattr(launch,'native_query',lambda *a,**kw:query_response({'schema':1,key:True,'segmented32_v2':True}))
    with pytest.raises(RuntimeError,match='NATIVE_MEMORY_BACKEND_UNAVAILABLE'):
        launch._require_memory_backend(str(binary),{},backend)


def test_capabilities_are_not_cached_when_dependencies_cannot_be_identified(tmp_path,monkeypatch):
    binary=tmp_path/'mgbfs';binary.write_bytes(b'unidentified')
    monkeypatch.setattr('multigpubfs.autotune._dependency_identity',lambda *a,**kw:None)
    calls=[]
    def query(*a,**kw):
        calls.append(1)
        return query_response({'schema':1,'segmented24_v1':True,'segmented32_v2':True,'segmented_memory_presets_v1':True})
    monkeypatch.setattr(launch,'native_query',query)
    launch._require_memory_backend(str(binary),{},'segmented24')
    launch._require_memory_backend(str(binary),{},'segmented24')
    assert len(calls)==2


def test_changed_cuda_library_invalidates_capability_cache(tmp_path,monkeypatch):
    binary=tmp_path/'mgbfs';binary.write_bytes(b'identified')
    identity={'cuda_library_sha256':'first', 'resolved_dependencies':(('libcuda','first'),)}
    monkeypatch.setattr('multigpubfs.autotune._dependency_identity',lambda *a,**kw:dict(identity))
    calls=[]
    def query(*a,**kw):
        calls.append(1)
        return query_response({'schema':1,'segmented24_v1':True,'segmented32_v2':True,'segmented_memory_presets_v1':True})
    monkeypatch.setattr(launch,'native_query',query)
    launch._require_memory_backend(str(binary),{},'segmented24')
    launch._require_memory_backend(str(binary),{},'segmented24')
    assert len(calls)==1
    identity['cuda_library_sha256']='second'
    launch._require_memory_backend(str(binary),{},'segmented24')
    assert len(calls)==2
