"""Real child termination on cold-query timeout; no CUDA required."""
import os,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.autotune import _admit
from multigpubfs.process_control import query,run

class StartupTimeoutTests(unittest.TestCase):
 def test_public_and_profile_queries_terminate_real_child_without_completion(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pidfile=root/'pid';native=root/'native'
   native.write_text('#!/usr/bin/env python3\nimport os,time\nfrom pathlib import Path\nPath('+repr(str(pidfile))+').write_text(str(os.getpid()))\ntime.sleep(120)\n');native.chmod(0o755)
   g=GraphDefinition.permutation([[1,0]],[0,1])
   with patch.dict(os.environ,{'MGBFS_NATIVE_QUERY_SECONDS':'0.3'}):
    for index,action in enumerate([lambda:run_graph(g,root/'public',devices=[0],shards=1,autotune=False,backend='generic',executable=str(native)),lambda:_admit(g,[0],None,1,str(native),dict(os.environ),root)]):
     with self.assertRaisesRegex(RuntimeError,'NATIVE_STARTUP_QUERY_TIMEOUT'):action()
     pid=int(pidfile.read_text());self.assertFalse((Path('/proc')/str(pid)).exists())
     self.assertFalse((root/'public/report.json').exists())
 def test_execution_timeout_terminates_child_without_completion(self):
  with tempfile.TemporaryDirectory() as d:
   pidfile=Path(d)/'pid'
   import sys
   code='import os,time;from pathlib import Path;Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(120)'
   with self.assertRaisesRegex(RuntimeError,'NATIVE_EXECUTION_TIMEOUT'):run([sys.executable,'-c',code],timeout=.3)
   self.assertFalse((Path('/proc')/pidfile.read_text()).exists())
 def test_invalid_limit_is_rejected_before_spawn(self):
  for value in ['0','-1','nan','inf','invalid']:
   with patch.dict(os.environ,{'MGBFS_NATIVE_QUERY_SECONDS':value}):
    with self.assertRaisesRegex(ValueError,'INVALID_NATIVE_QUERY_TIMEOUT'):query(['unused'])

if __name__=='__main__':unittest.main()
