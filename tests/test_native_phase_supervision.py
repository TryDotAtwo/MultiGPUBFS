import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import multigpubfs.launch as launcher

class NativePhaseSupervision(unittest.TestCase):
 def test_recorded_native_error_is_a_failure_before_process_exit(self):
  detect=getattr(launcher,'_phase_failures',None)
  self.assertTrue(callable(detect),'Supervisor cannot detect a recorded native error before worker exit')
  phase={'phase':'bfs','native_error':{'code':999,'call':'cudaMemcpyAsync','layer':2}}
  self.assertEqual(detect([None,phase]),[(1,1)])
  self.assertEqual(detect([{'native_error':None},{'native_error':{'code':0}}]),[])
 def test_one_shot_phase_is_read_only_for_its_actual_pid(self):
  read=getattr(launcher,'_job_phase',None)
  self.assertTrue(callable(read),'One-shot processes have no native phase supervision')
  with tempfile.TemporaryDirectory() as temporary:
   out=Path(temporary);path=out/'rank-0.phase.json';job=SimpleNamespace(pid=12345)
   phase={'schema':1,'pid':12345,'phase':'bfs','entered':{'bfs':1},'native_error':{'code':999}}
   path.write_text(json.dumps(phase));self.assertEqual(read(job,out,0),phase)
   path.write_text(json.dumps(dict(phase,pid=7)));self.assertIsNone(read(job,out,0))
   path.write_text('interrupted-json');self.assertIsNone(read(job,out,0))
