import unittest,tempfile,json,hashlib
from pathlib import Path
from multigpubfs.native_distribution import runtime_identity
class RuntimeIdentityTests(unittest.TestCase):
 def test_verified_manifest_identity_and_library_tamper(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'bin').mkdir();(root/'lib').mkdir();binary=root/'bin/mgbfs';binary.write_bytes(b'binary');library=root/'lib/libmgbfs_cuda.so';library.write_bytes(b'cuda')
   manifest={'schema':1,'source_commit':'source','cuda_toolkit':'13.2','architectures':[86],'sha256':{'bin/mgbfs':hashlib.sha256(binary.read_bytes()).hexdigest(),'lib/libmgbfs_cuda.so':hashlib.sha256(library.read_bytes()).hexdigest()}}
   (root/'manifest.json').write_text(json.dumps(manifest));v=runtime_identity(binary,{})
   self.assertEqual(v['source_commit'],'source');self.assertEqual(v['native_sha256'],manifest['sha256']['bin/mgbfs']);self.assertEqual(len(v['python_runtime_sha256']),64)
   library.write_bytes(b'tamper')
   with self.assertRaisesRegex(RuntimeError,'LIBRARY_CHECKSUM'):runtime_identity(binary,{})
if __name__=='__main__':unittest.main()
