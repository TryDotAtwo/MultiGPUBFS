import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import native_distribution as native
class ComponentProvenance(unittest.TestCase):
 def test_distinct_cli_and_cuda_source_revisions_follow_resolved_artifacts(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'bin').mkdir();(root/'lib').mkdir();binary=root/'bin/mgbfs';binary.write_bytes(b'new-cli');library=root/'lib/libmgbfs_cuda.so';library.write_bytes(b'new-cuda')
   sha=hashlib.sha256(library.read_bytes()).hexdigest()
   manifest={'schema':1,'source_commit':'legacy-cuda','native_binary_source_commit':'trace-cli','cuda_library_source_commit':'universal-cuda','architectures':[75,100,120],'ptx_architectures':[80],'sha256':{'bin/mgbfs':hashlib.sha256(binary.read_bytes()).hexdigest(),'lib/libmgbfs_cuda.so':sha}}
   (root/'manifest.json').write_text(json.dumps(manifest))
   with patch('multigpubfs.autotune._dependency_identity',return_value={'cuda_library_sha256':sha}):r=native._runtime_identity_uncached(binary,{})
  self.assertEqual(r['source_commit'],'trace-cli');self.assertEqual(r['cuda_library_source_commit'],'universal-cuda');self.assertEqual(r['ptx_architectures'],[80])
