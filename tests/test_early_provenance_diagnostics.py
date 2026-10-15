import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import system_diagnostics as system
class EarlyBuildProvenance(unittest.TestCase):
 def test_even_a_pre_bfs_failure_bundle_identifies_cli_cuda_and_ptx(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'_native').mkdir();m={'native_binary_source_commit':'cli-source','cuda_library_source_commit':'cuda-source','ptx_architectures':[80],'architectures':[75,100,120],'sha256':{'bin/mgbfs':'cli-sha','lib/libmgbfs_cuda.so':'cuda-sha'}}
   (root/'_native/manifest.json').write_text(json.dumps(m))
   with patch.object(system,'__file__',str(root/'system_diagnostics.py')),patch.object(system,'library_candidates',return_value=[]):data=system.survey(root)
  declared=data['native_build_declaration']
  self.assertEqual(declared['native_binary_source_commit'],'cli-source');self.assertEqual(declared['cuda_library_source_commit'],'cuda-source');self.assertEqual(declared['ptx_architectures'],[80])
  self.assertEqual(declared['declared_artifact_sha256']['bin/mgbfs'],'cli-sha')
