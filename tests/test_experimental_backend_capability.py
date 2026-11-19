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
    calls=[]
    def query(*a,**kw):
        calls.append((a,kw));return query_response({'schema':1,key:True})
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
