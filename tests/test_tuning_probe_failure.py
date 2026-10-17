import json,subprocess,tempfile,unittest,warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from multigpubfs import autotune
from multigpubfs import generic_session
from multigpubfs.graph_definition import GraphDefinition

class ProbeFailureContainment(unittest.TestCase):
 def fixture(self,root):
  evidence=root/'saved/pilot-0/failure-evidence';(evidence/'rank-1').mkdir(parents=True)
  fault={'code':999,'call':'cudaMemcpyAsync','rank':1,'device':1,'layer':2,'last_completed_layer':1}
  (evidence/'rank-1/phase.json').write_text(json.dumps({'native_error':fault}))
  (evidence/'summary.json').write_text(json.dumps({'captures':[],'error':'NATIVE_DISTRIBUTED_RANK_FAILED [(1, 1)]'}))
  error=RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED [(1, 1)]; evidence='+str(evidence))
  error.tuning_evidence_root=str(root/'saved')
  pool=SimpleNamespace(generation=3,workers=['failed rank'],identity='failed',closes=0)
  def close():
   pool.closes+=1;pool.workers=[];pool.identity=None
  pool.close_workers=close
  return error,pool,evidence
 def test_failed_optional_probe_closes_every_old_rank_before_fresh_admission(self):
  graph=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
  with tempfile.TemporaryDirectory() as temp:
   error,pool,evidence=self.fixture(Path(temp))
   def admit(*args,**kwargs):
    self.assertEqual(pool.workers,[],'Failed CUDA ranks must not serve fresh admission')
    return {'devices':[0,1],'plan':{'batch':32,'generator_backend':'cuda','history_algorithm':'HASH'}}
   with patch.object(generic_session,'_active',pool),patch.object(autotune,'_session_profile_key',return_value=None),patch.object(autotune,'_choose_size_profile_uncached',side_effect=error) as probe,patch.object(autotune,'_admit',side_effect=admit),warnings.catch_warnings(record=True) as messages:
    warnings.simplefilter('always')
    try:result=autotune.choose_size_profile(graph,[0,1],None,30,'native',{},allow_specialized=False)
    except RuntimeError as e:self.fail('An optional probe aborted the main calculation: '+str(e))
    self.assertEqual(result['status'],'TUNING_PROBE_FAILED_CONSERVATIVE_PROFILE')
    self.assertFalse(result['measured']);self.assertFalse(result['online_size_tuning'])
    self.assertEqual(result['devices'],[0,1]);self.assertEqual(result['shards'],1)
    self.assertEqual(result['tuning_failure']['native_error']['code'],999)
    self.assertEqual(Path(result['tuning_failure']['evidence']),evidence)
    self.assertEqual(pool.closes,1);self.assertEqual(probe.call_count,1)
    self.assertTrue(any('autotun' in str(m.message).lower() for m in messages))
    # The same persistent session must not repeat the poisoned optional probe.
    again=autotune.choose_size_profile(graph,[0,1],None,30,'native',{},allow_specialized=False)
    self.assertEqual(again['status'],result['status']);self.assertEqual(probe.call_count,1)
    self.assertEqual(pool.closes,1)
 def test_non_native_and_unproven_errors_are_not_hidden(self):
  graph=GraphDefinition.permutation([[1,0]],[0,1])
  for error in [RuntimeError('BAD_GRAPH'),RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED [(0, 1)]'),subprocess.TimeoutExpired('native probe',5)]:
   with self.subTest(error=str(error)),patch.object(generic_session,'_active',None),patch.object(autotune,'_choose_size_profile_uncached',side_effect=error),patch.object(autotune,'_admit') as admit:
    with self.assertRaises(type(error)):autotune.choose_size_profile(graph,[0,1],None,30,'native',{},allow_specialized=False)
    admit.assert_not_called()
 def test_a_failed_fresh_admission_is_not_retried(self):
  graph=GraphDefinition.permutation([[1,0]],[0,1])
  with tempfile.TemporaryDirectory() as temp:
   error,pool,evidence=self.fixture(Path(temp))
   with patch.object(generic_session,'_active',pool),patch.object(autotune,'_session_profile_key',return_value=None),patch.object(autotune,'_choose_size_profile_uncached',side_effect=error),patch.object(autotune,'_admit',side_effect=RuntimeError('FRESH_CUDA_CONTEXT_FAILED')) as admit,warnings.catch_warnings(record=True):
    with self.assertRaisesRegex(RuntimeError,'FRESH_CUDA_CONTEXT_FAILED'):autotune.choose_size_profile(graph,[0,1],None,30,'native',{},allow_specialized=False)
    self.assertEqual(admit.call_count,1);self.assertEqual(pool.closes,1)
