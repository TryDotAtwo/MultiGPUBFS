import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import deep_diagnostics as deep
class DeepDiagnostics(unittest.TestCase):
 def test_matrix_modes_orders_visibility(self):
  cases=deep.startup_cases([0,1])
  self.assertEqual(len({x['name'] for x in cases}),len(cases))
  self.assertEqual({x['config']['module'] for x in cases},{'inherited','EAGER','LAZY'})
  self.assertEqual({x['config']['order'] for x in cases},{'engine-first','context-first'})
  self.assertTrue(all('CUDA_VISIBLE_DEVICES' not in x['config'] for x in cases))
 def test_api_cases_have_bounded_memory(self):
  self.assertLessEqual(deep.PROBE_BYTES,65536)
 def test_snapshot_only_owned_group(self):
  with tempfile.TemporaryDirectory() as d:
   value=deep.process_snapshot(os.getpid(),Path(d)/'process-snapshot.json')
   self.assertIn('processes',value)
   self.assertNotIn('environment',json.dumps(value))
 def test_cpu_oracles_finite(self):
  for name in ('permutation','repeated','matrix','matrix-gemm','wide'):
   g=deep.fixture(name);sizes,terminal=deep.oracle(g)
   self.assertEqual(sum(sizes),len(set(terminal)) if len(sizes)==1 else sum(sizes))
   self.assertGreater(sum(sizes),0);self.assertLessEqual(sum(sizes),64)
 def test_summary_does_not_claim_root_cause(self):
  value=deep.summarize([{'name':'x','status':'TIMEOUT','last_native_phase':'hardware-get-properties'}])
  self.assertFalse(value['root_cause_confirmed']);self.assertEqual(value['failures'][0]['last_native_phase'],'hardware-get-properties')
 def test_budget_does_not_start_next_process(self):
  from multigpubfs.diagnostics import Runner
  with tempfile.TemporaryDirectory() as d:
   r=Runner(Path(d)/'out',1)
   with patch.object(r,'run',side_effect=AssertionError('Must not launch')):
    deep.run_case(r,{'name':'expired','config':{'kind':'api'}},deadline=0)
   self.assertEqual(r.report['steps'][-1]['status'],'SKIPPED')
 def test_deadline_capture_cleanup_next_case(self):
  from multigpubfs.diagnostics import Runner
  import time
  with tempfile.TemporaryDirectory() as d:
   r=Runner(Path(d)/'out',.2);r.deep_capture=True
   result=r.run('hung',[sys.executable,'-c','import time;time.sleep(5)'])
   self.assertEqual(result['status'],'TIMEOUT');self.assertTrue((r.root/'hung'/'process-snapshot.json').exists())
   self.assertEqual(r.run('next',[sys.executable,'-c','pass'])['status'],'PASS')
