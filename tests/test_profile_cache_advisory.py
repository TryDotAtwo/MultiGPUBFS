"""Cache failures are advisory; native/logic failures must still propagate."""
import errno
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from multigpubfs.profile_cache import read_mapping, write_mapping

@pytest.mark.parametrize('content',['null','[]','"x"','{bad'])
def test_invalid_mapping_is_a_miss(tmp_path, content):
    p=tmp_path/'cache.json'
    p.write_text(content)
    data,status=read_mapping(p)
    assert data is None and status['status']=='INVALID'

def test_missing_or_inaccessible_cache(tmp_path):
    p=tmp_path/'cache.json'
    assert read_mapping(p)[1]['status']=='MISS'
    with patch.object(Path,'read_text',side_effect=PermissionError(errno.EACCES,'denied')):
        data,status=read_mapping(p)
    assert data is None and status['status']=='UNAVAILABLE' and status['stage']=='read' and status['errno']==errno.EACCES

def test_no_separate_stat_before_read(tmp_path):
    p=tmp_path/'cache.json'
    p.write_text('{"profile": 1}')
    with patch.object(Path,'stat',side_effect=OSError(errno.ESTALE,'stale')):
        data,status=read_mapping(p)
    assert data == {'profile':1} and status['status']=='HIT'

def test_success_atomic_replace_and_no_temp(tmp_path):
    p=tmp_path/'c'/'cache.json'
    p.parent.mkdir()
    p.write_text('{"old": 1}')
    status=write_mapping(p,{'profile':2},prefix='cache-')
    assert status['status']=='WRITTEN'
    assert json.loads(p.read_text())=={'profile':2}
    assert not list(p.parent.glob('*.tmp'))

@pytest.mark.parametrize('stage,err', [('mkdir',errno.EACCES),('mkstemp',errno.ENOSPC),('write',errno.EDQUOT),('close',errno.EIO),('replace',errno.ESTALE)])
def test_write_failures_report_stage_and_clean_owned_tmp(tmp_path, stage, err):
    p=tmp_path/'cache'/'profile.json'
    old_fdopen=__import__('os').fdopen
    class FailClose:
        def __init__(self, fd, *args, **kwargs): self.file=old_fdopen(fd,*args,**kwargs)
        def write(self, *args, **kwargs): return self.file.write(*args,**kwargs)
        def close(self):
            self.file.close()
            raise OSError(err,'injected')
    targets={'mkdir':'pathlib.Path.mkdir','mkstemp':'tempfile.mkstemp','write':'json.dump','close':'os.fdopen','replace':'os.replace'}
    replacement=(lambda *a,**k:FailClose(*a,**k)) if stage=='close' else None
    with patch(targets[stage], side_effect=replacement if replacement else OSError(err,'injected')):
        status=write_mapping(p,{'x':1},prefix='cache-')
    assert status['status']=='UNAVAILABLE' and status['stage']==stage and status['errno']==err
    if p.parent.exists(): assert not list(p.parent.glob('*.tmp'))
    assert not p.exists()

def test_programming_error_is_not_swallowed(tmp_path):
    p=tmp_path/'cache'/'profile.json'
    with patch('json.dump',side_effect=TypeError('bad object')):
        with pytest.raises(TypeError,match='bad object'):write_mapping(p,{'x':1},prefix='cache-')
    assert not list(p.parent.glob('*.tmp'))


def test_family_cache_invalid_type_and_failed_replace(tmp_path, monkeypatch):
    import os
    from multigpubfs import profile_families as family
    identity={'fixture':'family'}
    p=family.path(tmp_path,identity)
    p.write_text('[]')
    assert family.load(tmp_path,identity) is None
    def failed_replace(*args):raise OSError(errno.ESTALE,'injected stale file handle')
    monkeypatch.setattr(os,'replace',failed_replace)
    profile={'status':'MEASURED_EQUAL_PREFIX_GPU_PROFILE','measured':True}
    status=family.store(tmp_path,identity,profile)
    assert status=={'status':'UNAVAILABLE','stage':'replace','errno':errno.ESTALE}
    assert p.read_text()=='[]'
    assert not list(tmp_path.glob('*.tmp'))
