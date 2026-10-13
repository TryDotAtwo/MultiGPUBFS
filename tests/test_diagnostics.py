import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from multigpubfs.diagnostics import Runner
class DiagnosticFramework(unittest.TestCase):
 def test_bfs_component_dispatch_does_not_start_persistent_worker(self):
  from unittest.mock import patch
  from multigpubfs.diagnostics import _component
  with tempfile.TemporaryDirectory() as tmp,patch('multigpubfs.launch.run_graph',return_value={'status':'COMPLETE','layer_sizes':[1,2,2,1],'bfs_seconds':.01}) as run:
   result=_component('bfs',Path(tmp),[0],None,60)
   self.assertTrue(result['exact_oracle_verified']);run.assert_called_once()
 def test_unknown_component_rejected(self):
  from multigpubfs.diagnostics import _component
  with self.assertRaisesRegex(ValueError,'Unknown diagnostic component'):_component('invalid',Path('.'),[],None,60)
 def runner(self,timeout=3):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);return Runner(Path(tmp.name)/'report',timeout)
 def test_success_and_manifest(self):
  r=self.runner();e=r.run('ok',[sys.executable,'-c','print("PHASE check")']);self.assertEqual(e['status'],'PASS');self.assertEqual(r.finish()['status'],'PASS')
  self.assertTrue((r.root/'diagnostics.zip').is_file());self.assertEqual(json.loads((r.root/'report.json').read_text())['status'],'PASS')
 def test_failure_preserves_log_and_continues(self):
  r=self.runner();e=r.run('bad',[sys.executable,'-c','import sys;print("original failure");sys.exit(7)']);self.assertEqual(e['exit_code'],7)
  self.assertEqual(r.run('next',[sys.executable,'-c','pass'])['status'],'PASS');self.assertEqual(r.finish()['failed_components'],['bad']);self.assertIn('original failure',(r.root/'bad.log').read_text())
 def test_timeout_preserves_partial_report(self):
  r=self.runner(.2);e=r.run('slow',[sys.executable,'-c','import time;print("PHASE sleep",flush=True);time.sleep(30)']);self.assertEqual(e['status'],'TIMEOUT');r.finish();self.assertIn('sleep',(r.root/'slow.log').read_text())
 def test_existing_output_not_overwritten(self):
  r=self.runner()
  with self.assertRaises(FileExistsError):Runner(r.root,1)
 def test_metadata_only_archive(self):
  import zipfile
  r=self.runner();(r.root/'states.json').write_text('private payload');(r.root/'job-000.json').write_text('private environment');r.finish()
  with zipfile.ZipFile(r.root/'diagnostics.zip') as z:self.assertNotIn('states.json',z.namelist());self.assertNotIn('job-000.json',z.namelist())
 @unittest.skipUnless(os.name=='posix','POSIX group termination')
 def test_timeout_terminates_child_group(self):
  r=self.runner(.5);pid=r.root/'grandchild.pid'
  code='import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)'
  r.run('tree',[sys.executable,'-c',code,str(pid)]);child=int(pid.read_text());status=Path('/proc')/str(child)/'stat'
  self.assertTrue(not status.exists() or status.read_text().split()[2]=='Z');r.finish()
