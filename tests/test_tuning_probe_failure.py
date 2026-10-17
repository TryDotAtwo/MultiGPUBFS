import json,tempfile,unittest
from pathlib import Path
from multigpubfs.autotune import _native_probe_failure

class ProbeFaultEvidence(unittest.TestCase):
 def error(self,root,code=999,last=901):
  ev=root/'saved/pilot/failure-evidence';(ev/'rank-1').mkdir(parents=True)
  (ev/'rank-1/phase.json').write_text(json.dumps({'native_error':{'code':code,'call':'cudaMemcpyAsync','rank':1,'layer':2},'native_last_error':{'code':last}}))
  (ev/'summary.json').write_text('{}')
  error=RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED [(1, 1)]; evidence='+str(ev));error.tuning_evidence_root=str(root/'saved');return error,ev
 def test_primary_cuda999_is_recognized_despite_secondary_cleanup_error(self):
  with tempfile.TemporaryDirectory() as tmp:
   error,ev=self.error(Path(tmp));found=_native_probe_failure(error)
   self.assertEqual(found['native_error']['code'],999);self.assertEqual(Path(found['evidence']),ev);self.assertFalse(found['probe_retried'])
 def test_secondary999_does_not_turn_another_primary_error_into_a_recoverable_trial(self):
  with tempfile.TemporaryDirectory() as tmp:
   error,_=self.error(Path(tmp),700,999);self.assertIsNone(_native_probe_failure(error))
 def test_unproven_rank_failure_and_completion_timeout_remain_errors(self):
  for message in ['NATIVE_DISTRIBUTED_RANK_FAILED [(0, 1)]','DISTRIBUTED_COMPLETION_TIMEOUT','BAD_GRAPH']:
   self.assertIsNone(_native_probe_failure(RuntimeError(message)))
