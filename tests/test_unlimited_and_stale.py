import errno, json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from multigpubfs.work_limits import normalize, wire, remaining, phase_expired
from multigpubfs.generic_session import Job
from multigpubfs.cayleypy_backend import NativeOptions

def test_unlimited_default_and_wire():
 assert NativeOptions().max_seconds is None
 assert normalize(None) is None
 assert normalize(0) is None
 assert wire(None)==0
 assert remaining(None,999999) is None
 assert remaining(10,3)==7
 for x in (-1,True,1.5,"0"):
  with pytest.raises(ValueError): normalize(x)

def test_phase_deadlines():
 assert not phase_expired({"phase":"bfs","entered":{"bfs":0}},100000,None)
 assert phase_expired({"phase":"bfs","entered":{"bfs":0}},100000,20)
 assert phase_expired({"phase":"preparing","entered":{"preparing":0}},601,None)
 assert phase_expired({"phase":"retiring","entered":{"retiring":0}},601,None)

def job(tmp_path):
 return Job(SimpleNamespace(root=tmp_path,process=SimpleNamespace(pid=123,poll=lambda:None)),7)

def test_phase_recovers_stale_once(tmp_path):
 j=job(tmp_path); value={"schema":1,"pid":123,"sequence":"7"}
 with patch.object(Path,"read_text",side_effect=[OSError(errno.ESTALE,"stale"),json.dumps(value)]) as read:
  assert j.phase()==value
  assert read.call_count==2

def test_phase_identity_still_enforced(tmp_path):
 with patch.object(Path,"read_text",return_value='{"schema":1,"pid":999,"sequence":"7"}'):
  with pytest.raises(RuntimeError,match="IDENTITY"):job(tmp_path).phase()

def test_stale_bounded_and_permission_not_retried(tmp_path):
 for code,count in [(errno.ESTALE,3),(errno.EACCES,1)]:
  with patch.object(Path,"read_text",side_effect=OSError(code,"failure")) as read:
   with pytest.raises(OSError):job(tmp_path).phase()
   assert read.call_count==count

def test_response_recovers_stale(tmp_path):
 j=job(tmp_path)
 with patch.object(Path,"read_text",side_effect=[OSError(errno.ESTALE,"stale"),'{"schema":1,"sequence":7,"ok":true,"result":{"exit_code":0}}']):
  assert j.poll()==0

def test_missing_phase(tmp_path):
 assert job(tmp_path).phase() is None


def test_missing_standalone_phase(tmp_path):
 from multigpubfs.launch import _job_phase
 assert _job_phase(SimpleNamespace(pid=123),tmp_path,0) is None
