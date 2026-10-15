"""Regressions for owned-child cleanup and the SM120 production default."""
import os,subprocess,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import deep_diagnostics as deep,runtime_variants as variants

class CleanupAndProduction(unittest.TestCase):
 @unittest.skipUnless(os.name=='posix','Linux process-group contract')
 def test_group_cleanup_waits_for_descendant_after_leader_exit(self):
  started=time.monotonic()
  class Child:
   name='492123'
   def __truediv__(self,name):return self
   def read_text(self):return '492123 (mgbfs) R 1 492122 492122 0'
  class Proc:
   def is_dir(self):return True
   def iterdir(self):return [Child()] if time.monotonic()-started<.08 else []
  class Leader:
   pid=492122
   def poll(self):return 0
   def wait(self,timeout=None):return 0
  with patch.object(deep,'Path',return_value=Proc()),patch.object(os,'getpgid',return_value=492122),patch.object(os,'killpg'):
   result=deep.bounded_stop(Leader())
  self.assertTrue(result['complete'],result)
  self.assertEqual(result['surviving_group_pids'],[])

 @unittest.skipUnless(os.name=='posix','Linux process-group contract')
 def test_timeout_cleanup_runs_once_and_records_group_evidence(self):
  from multigpubfs.diagnostics import Runner
  import sys,json
  with tempfile.TemporaryDirectory() as d:
   runner=Runner(Path(d)/'out',.15);runner.deep_capture=True
   result=runner.run('hung',[sys.executable,'-c','import time;time.sleep(5)'])
   self.assertEqual(result['status'],'TIMEOUT')
   self.assertEqual(len(runner.report['diagnostic_cleanup']),1)
   self.assertTrue(runner.report['diagnostic_cleanup'][0]['complete'])
   self.assertTrue((runner.root/'hung/cleanup.json').is_file())
   self.assertTrue(json.loads((runner.root/'hung/cleanup.json').read_text())['complete'])

 def test_sm120_production_bfs_precedes_legacy_contrasts(self):
  cases=variants.cases([0,1]);first_legacy=next(i for i,x in enumerate(cases) if 120 not in variants.SPECS[x['config']['variant']][1])
  for device in (0,1):
   primary=[(i,x) for i,x in enumerate(cases) if x['config']['variant']=='shared-sm120' and (x['config'].get('device')==device or x['config'].get('devices')==[device])]
   self.assertEqual(len(primary),3)
   self.assertTrue(all(i<first_legacy for i,x in primary))
  self.assertEqual(getattr(variants,'PRODUCTION',None),'shared-sm120')

 def test_coverage_prefers_current_shared_sm120_production(self):
  steps=[dict(name=c['name'],status='PASS',diagnostic_case=c['config']) for c in variants.cases([0,1])]
  self.assertEqual(variants.choose_coverage(steps,[0,1])['id'],'shared-sm120')

 def test_primary_and_contrast_partitions_retain_complete_matrix(self):
  primary=variants.cases([0,1],production_only=True)
  contrast=variants.cases([0,1],contrasts_only=True)
  self.assertEqual(len(primary),6);self.assertEqual(len(contrast),18)
  self.assertEqual({c['name'] for c in primary+contrast},{c['name'] for c in variants.cases([0,1])})
  self.assertTrue(all(c['config']['variant']=='shared-sm120' for c in primary))

 @unittest.skipUnless(os.name=='posix','Linux process-group contract')
 def test_real_owned_child_and_leader_exit_are_both_observed(self):
  import sys,json
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'child.json'
   script="import subprocess,sys,time,json;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid}));time.sleep(30)"
   leader=subprocess.Popen([sys.executable,'-c',script,str(path)],start_new_session=True)
   try:
    deadline=time.monotonic()+3
    while not path.is_file() and time.monotonic()<deadline:time.sleep(.01)
    self.assertTrue(path.is_file())
    result=deep.bounded_stop(leader)
    self.assertTrue(result['complete'],result)
    self.assertIsNotNone(leader.poll())
    child=int(json.loads(path.read_text())['pid']);stat=Path('/proc')/str(child)/'stat'
    self.assertTrue(not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z')
   finally:
    if leader.poll() is None:os.killpg(leader.pid,9);leader.wait(timeout=3)

 def test_unreaped_owned_child_is_a_failure_and_stops_following_probes(self):
  from multigpubfs.diagnostics import Runner
  import sys
  with tempfile.TemporaryDirectory() as d:
   runner=Runner(Path(d)/'out',1)
   failed={'pid':1,'signals_sent':[15,9],'process_exit_observed':True,'surviving_group_pids':[12345],'complete':False}
   with patch.object(deep,'bounded_stop',return_value=failed),patch.object(deep,'process_snapshot',return_value={}):
    result=runner.run('orphan',[sys.executable,'-c','pass'])
   self.assertEqual(result['status'],'FAIL')
   self.assertIn('OWNED_DIAGNOSTIC_PROCESS_COULD_NOT_BE_REAPED',result['error'])
   with patch('multigpubfs.diagnostics.subprocess.Popen',side_effect=AssertionError('Must not start another probe')):
    following=runner.run('following',[sys.executable,'-c','pass'])
   self.assertEqual(following['status'],'SKIPPED')
