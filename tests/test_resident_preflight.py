"""Communication rehearsal must run on an existing healthy resident pool."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from multigpubfs import GraphDefinition
from multigpubfs.nccl_preflight import _cached_prefix

class ResidentPreflightTests(unittest.TestCase):
 def test_cached_transport_runs_real_graph_prefix_in_resident_pool(self):
  graph=GraphDefinition.permutation([[1,0,2]],[0,1,2]);env={'NCCL_IB_DISABLE':'1','MGBFS_SEGMENT_STRATEGY':'warp'}
  pool=Mock();pool.workers=[object(),object()]
  report={'policy':{'NCCL_IB_DISABLE':'1'}}
  # Fake the old disposable subprocess to return a valid receipt. The test
  # fails on old code because the real graph was never sent to the live pool.
  def disposable(command,env,stdout,stderr,start_new_session):
   prefix=Path(command[-2])/'prefix';prefix.mkdir()
   (prefix/'report.json').write_text(json.dumps({'status':'COMPLETE','layer_sizes':[1,1]}))
   proc=Mock();proc.wait.return_value=0;proc.poll.return_value=0;return proc
  with tempfile.TemporaryDirectory() as tmp,patch('multigpubfs.generic_session._active',pool),patch('multigpubfs.launch.run_graph',return_value={'status':'COMPLETE','layer_sizes':[1,1]}) as run,patch('multigpubfs.nccl_preflight.subprocess.Popen',side_effect=disposable) as spawn:
   receipt=_cached_prefix(Path(tmp),'native',[0,1],env,{},report,graph,'permutation_bitmap')
   self.assertEqual(receipt['layer_sizes'],[1,1]);run.assert_called_once();spawn.assert_not_called()
   self.assertEqual(run.call_args.kwargs['_profile_layers'],4)
   self.assertEqual(run.call_args.kwargs['_native_env']['MGBFS_SEGMENT_STRATEGY'],'warp')
   self.assertEqual(run.call_args.kwargs['_native_env']['NCCL_IB_DISABLE'],'1')
   pool.close_workers.assert_not_called()


class ResidentPreflightFailureTests(unittest.TestCase):
 def setUp(self):
  self.graph=GraphDefinition.permutation([[1,0,2]],[0,1,2])
  self.pool=Mock();self.pool.workers=[object(),object()];self.pool.communication_profiles={'previous':'receipt'}
 def test_failed_prefix_discards_workers_and_invalidates_evidence(self):
  with tempfile.TemporaryDirectory() as tmp,patch('multigpubfs.generic_session._active',self.pool),patch('multigpubfs.launch.run_graph',side_effect=RuntimeError('KERNEL_FAILED')):
   with self.assertRaisesRegex(RuntimeError,'KERNEL_FAILED'):
    _cached_prefix(Path(tmp),'native',[0,1],{}, {},{'policy':{}},self.graph,'permutation_bitmap')
   self.pool.close_workers.assert_called_once();self.assertEqual(self.pool.communication_profiles,{})
   detail=json.loads((Path(tmp)/'cached-transport'/'prefix-execution.json').read_text())
   self.assertEqual(detail['status'],'FAILED')
 def test_too_short_incomplete_prefix_is_not_accepted(self):
  with tempfile.TemporaryDirectory() as tmp,patch('multigpubfs.generic_session._active',self.pool),patch('multigpubfs.launch.run_graph',return_value={'status':'INCOMPLETE','layer_sizes':[1,2]}):
   with self.assertRaisesRegex(RuntimeError,'NCCL_PREFLIGHT_INSUFFICIENT_BFS_PREFIX'):
    _cached_prefix(Path(tmp),'native',[0,1],{}, {},{'policy':{}},self.graph,'permutation_bitmap')
   self.pool.close_workers.assert_called_once()
 def test_cleanup_failure_aborts_without_allowing_fallback(self):
  from multigpubfs.nccl_preflight import ResidentPreflightCleanupError
  self.pool.close_workers.side_effect=OSError('unable to close')
  with tempfile.TemporaryDirectory() as tmp,patch('multigpubfs.generic_session._active',self.pool),patch('multigpubfs.launch.run_graph',side_effect=RuntimeError('KERNEL_FAILED')):
   with self.assertRaises(ResidentPreflightCleanupError):
    _cached_prefix(Path(tmp),'native',[0,1],{}, {},{'policy':{}},self.graph,'permutation_bitmap')

class ResidentPreflightTimeoutTests(unittest.TestCase):
 def test_timeout_is_normalized_for_existing_transport_fallback(self):
  import subprocess
  pool=Mock();pool.workers=[object(),object()];pool.communication_profiles={'previous':'receipt'}
  graph=GraphDefinition.permutation([[1,0,2]],[0,1,2])
  with tempfile.TemporaryDirectory() as tmp,patch('multigpubfs.generic_session._active',pool),patch('multigpubfs.launch.run_graph',side_effect=subprocess.TimeoutExpired('native',20)):
   try:_cached_prefix(Path(tmp),'native',[0,1],{}, {},{'policy':{}},graph,'permutation_bitmap')
   except Exception as error:
    self.assertIsInstance(error,TimeoutError)
    self.assertIn('NCCL_PREFLIGHT_PREFIX_STARTUP_TIMEOUT',str(error))
   else:self.fail('timed-out prefix was accepted')
   pool.close_workers.assert_called_once();self.assertEqual(pool.communication_profiles,{})



class ResidentCacheEpochTests(unittest.TestCase):
 def test_cache_is_bound_to_live_worker_generation(self):
  from multigpubfs.nccl_preflight import _bound_cache,bind_communication_group
  def worker(pid,alive=True):
   process=Mock();process.pid=pid;process.poll.return_value=None if alive else 1
   w=Mock();w.process=process;w.job=None;return w
  pool=Mock();pool.workers=[worker(11),worker(12)];pool.generation=3;pool.identity=('gpu-group',);pool.communication_profiles={'entry':'receipt'};pool.communication_bound_token=None
  self.assertEqual(_bound_cache(pool),{})
  pool.communication_profiles={'entry':'receipt'};bind_communication_group(pool)
  self.assertEqual(_bound_cache(pool),{'entry':'receipt'})
  pool.workers[0].process.poll.return_value=1
  self.assertEqual(_bound_cache(pool),{})
  pool.workers=[worker(21),worker(22)];pool.generation=4;pool.communication_profiles={'stale':'receipt'}
  self.assertEqual(_bound_cache(pool),{})
 def test_dependency_change_alters_transport_identity(self):
  from multigpubfs.nccl_preflight import _identity
  first={'cuda_library_sha256':'aaa','resolved_dependencies':[['libc.so','/a',1,2]]}
  second={'cuda_library_sha256':'bbb','resolved_dependencies':[['libc.so','/a',1,2]]}
  with patch('multigpubfs.autotune._stat_identity',return_value=('native-stat',)),patch('multigpubfs.autotune._dependency_identity',side_effect=[first,second]):
   self.assertNotEqual(_identity('/native',[0,1],{}),_identity('/native',[0,1],{}))

if __name__=='__main__':unittest.main()
