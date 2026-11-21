import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from multigpubfs.launch import _native_log_faults, _stop_rank_jobs
from multigpubfs.generic_session import Pool
class FailureSupervisorTests(unittest.TestCase):
 def test_log_is_independent_of_phase_and_ignores_previous_job(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);log=root/'worker.log';old=b'MGBFS_NATIVE_ERROR {"code":17}\n';log.write_bytes(old)
   job=SimpleNamespace(worker=SimpleNamespace(root=root),log_offset=len(old));cursors={}
   self.assertEqual(_native_log_faults([job],root,cursors),[])
   with log.open('ab') as f:f.write(b'prefix MGBFS_NATIVE_ERROR {"code":999,"rank":0')
   self.assertEqual(_native_log_faults([job],root,cursors),[])
   with log.open('ab') as f:f.write(b',"call":"cudaMemcpy"}\n')
   self.assertEqual(_native_log_faults([job],root,cursors),[(0,{'code':999,'rank':0,'call':'cudaMemcpy'})])
   self.assertEqual(_native_log_faults([job],root,cursors),[])
 def test_teardown_deadline_is_for_whole_group(self):
  import subprocess,time
  class Job:
   def __init__(self,pid,unkillable=False):self.pid=pid;self.code=None;self.unkillable=unkillable
   def poll(self):return self.code
   def terminate(self):pass
   def kill(self):
    if not self.unkillable:self.code=-9
   def wait(self,timeout=None):
    if self.code is not None:return self.code
    time.sleep(timeout or 0);raise subprocess.TimeoutExpired('fixture',timeout)
  jobs=[Job(i) for i in range(8)];jobs[-1].unkillable=True
  start=time.monotonic();survivors=_stop_rank_jobs(jobs,grace=.03,kill_grace=.03);elapsed=time.monotonic()-start
  self.assertEqual(survivors,[7]);self.assertLess(elapsed,.2)
 def test_backend_family_transition_retires_legacy_communicator(self):
  from unittest.mock import Mock
  pool=Pool()
  try:
   pool.identity=(None,None,None,None,'segmented');pool.close_workers=Mock()
   pool.prepare_family({'MGBFS_EXPERIMENTAL_SEGMENTED24':'1'})
   pool.close_workers.assert_not_called()
   pool.prepare_family({'MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_EXPERIMENTAL_BITMAP':'1'})
   pool.close_workers.assert_not_called()
   pool.prepare_family({});pool.close_workers.assert_called_once()
  finally:pool.temporary.cleanup()
 def test_poisoned_pool_cannot_silently_restart(self):
  pool=Pool();pool.poisoned=True
  try:
   with self.assertRaisesRegex(RuntimeError,'FAILED_GROUP_NO_RESTART'):pool.select('unused',[0],{})
  finally:pool.close()
if __name__=='__main__':unittest.main()
