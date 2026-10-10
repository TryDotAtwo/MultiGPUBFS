import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
from multigpubfs.native_distribution import runtime_identity
class RuntimeIdentityTests(unittest.TestCase):
 def test_verified_manifest_identity_and_library_tamper(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'bin').mkdir();(root/'lib').mkdir();binary=root/'bin/mgbfs';binary.write_bytes(b'binary');library=root/'lib/libmgbfs_cuda.so';library.write_bytes(b'cuda')
   manifest={'schema':1,'source_commit':'source','cuda_toolkit':'13.2','architectures':[86],'sha256':{'bin/mgbfs':hashlib.sha256(binary.read_bytes()).hexdigest(),'lib/libmgbfs_cuda.so':hashlib.sha256(library.read_bytes()).hexdigest()}}
   (root/'manifest.json').write_text(json.dumps(manifest))
   with patch('multigpubfs.autotune._dependency_identity',return_value={'cuda_library_sha256':manifest['sha256']['lib/libmgbfs_cuda.so']}):v=runtime_identity(binary,{})
   self.assertEqual(v['source_commit'],'source');self.assertEqual(v['native_sha256'],manifest['sha256']['bin/mgbfs']);self.assertEqual(len(v['python_runtime_sha256']),64)
   library.write_bytes(b'tamper')
   with self.assertRaisesRegex(RuntimeError,'LIBRARY_CHECKSUM'):runtime_identity(binary,{})
 def test_real_loader_override_does_not_claim_package_source(self):
  import subprocess,os
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'bin').mkdir();(root/'lib').mkdir();(root/'other').mkdir()
   binary=root/'bin/mgbfs';library=root/'lib/libmgbfs_cuda.so';other=root/'other/libmgbfs_cuda.so'
   for path,value in ((library,1),(other,2)):
    source=path.with_suffix('.cpp');source.write_text('extern "C" int identity(){return '+str(value)+';}')
    subprocess.run(['g++','-shared','-fPIC',str(source),'-o',str(path)],check=True,capture_output=True)
   source=root/'main.cpp';source.write_text('extern "C" int identity(); int main(){return identity();}')
   subprocess.run(['g++',str(source),'-L'+str(root/'lib'),'-lmgbfs_cuda','-o',str(binary)],check=True,capture_output=True)
   manifest={'schema':1,'source_commit':'package-source','cuda_toolkit':'13.2','architectures':[86],'sha256':{'bin/mgbfs':hashlib.sha256(binary.read_bytes()).hexdigest(),'lib/libmgbfs_cuda.so':hashlib.sha256(library.read_bytes()).hexdigest()}}
   (root/'manifest.json').write_text(json.dumps(manifest))
   own=runtime_identity(binary,dict(os.environ,LD_LIBRARY_PATH=str(root/'lib')))
   self.assertEqual(own['source_commit'],'package-source')
   override=runtime_identity(binary,dict(os.environ,LD_LIBRARY_PATH=str(root/'other')))
   self.assertIsNone(override['source_commit']);self.assertEqual(override['cuda_library_sha256'],hashlib.sha256(other.read_bytes()).hexdigest())
   with patch('multigpubfs.autotune._dependency_identity',return_value=None):
    self.assertIsNone(runtime_identity(binary,{})['source_commit'])
if __name__=='__main__':unittest.main()
