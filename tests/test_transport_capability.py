import unittest,subprocess
from unittest.mock import patch
from multigpubfs.transport import library_capabilities
class Capabilities(unittest.TestCase):
 def test_compiled_on_and_off_are_distinct(self):
  for value in ('true','false'):
   with patch('multigpubfs.transport.subprocess.run',return_value=subprocess.CompletedProcess([],0,'{"lsa_compiled":'+value+'}','')):
    self.assertEqual(library_capabilities({})['lsa_compiled'],value=='true')
 def test_old_library_is_unknown_not_hardware_failure(self):
  with patch('multigpubfs.transport.subprocess.run',return_value=subprocess.CompletedProcess([],0,'{"lsa_compiled":null}','')):self.assertIsNone(library_capabilities({})['lsa_compiled'])
 def test_timeout_is_not_device_rejection(self):
  with patch('multigpubfs.transport.subprocess.run',side_effect=subprocess.TimeoutExpired('query',10)):
   self.assertEqual(library_capabilities({})['status'],'CAPABILITY_QUERY_UNAVAILABLE')
 def test_known_device_rejection_falls_back_but_explicit_lsa_errors(self):
  import tempfile
  from pathlib import Path
  from multigpubfs.transport import select_peer_transport
  def fail(graph,output,**kwargs):
   output.mkdir(parents=True)
   for rank in range(2):(output/f'rank-{rank}.log').write_text('{"status":"ERROR","error":"LSA_PREPARE_GROUP: CUDA_STATUS_4"}')
   raise RuntimeError('SPECIALIZED_COMMITTED_SNAPSHOT_MISSING')
  with tempfile.TemporaryDirectory() as d,patch('multigpubfs.transport.library_capabilities',return_value={'lsa_compiled':True}),patch('multigpubfs.specialized.run_specialized',side_effect=fail):
   decision=select_peer_transport('native',{},[0,1],Path(d)/'auto');self.assertEqual(decision['status'],'LSA_DEVICE_TOPOLOGY_UNSUPPORTED')
   with self.assertRaisesRegex(RuntimeError,'REQUESTED_LSA_NOT_VERIFIED'):select_peer_transport('native',{},[0,1],Path(d)/'explicit',request='lsa')
 def test_unrecognized_native_failure_is_never_hidden(self):
  import tempfile
  from pathlib import Path
  from multigpubfs.transport import select_peer_transport
  def fail(graph,output,**kwargs):
   output.mkdir(parents=True)
   for rank in range(2):(output/f'rank-{rank}.log').write_text('CUDA_STATUS_700')
   raise RuntimeError('illegal access')
  with tempfile.TemporaryDirectory() as d,patch('multigpubfs.transport.library_capabilities',return_value={'lsa_compiled':True}),patch('multigpubfs.specialized.run_specialized',side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,'illegal access'):select_peer_transport('native',{},[0,1],Path(d)/'unknown')
if __name__=='__main__':unittest.main()
