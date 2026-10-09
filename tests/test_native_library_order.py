import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import multigpubfs.native_distribution as distribution

class LibraryOrder(unittest.TestCase):
 def test_packaged_libraries_precede_system_and_inherited_stubs(self):
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);package=root/'package';native=package/'_native';pure=root/'pure'
   hashes={}
   for name in ('bin/mgbfs','lib/libmgbfs_cuda.so'):
    p=native/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
   (native/'manifest.json').write_text(json.dumps({'schema':1,'sha256':hashes}))
   for name in ('cuda_runtime','nccl'):(pure/'nvidia'/name/'lib').mkdir(parents=True)
   with patch.object(distribution,'__file__',str(package/'native_distribution.py')),patch.object(distribution.sysconfig,'get_paths',return_value={'purelib':str(pure)}),patch.dict(distribution.os.environ,{'LD_LIBRARY_PATH':'/usr/local/cuda/compat'},clear=True):
    binary,env=distribution.native_runtime();paths=env['LD_LIBRARY_PATH'].split(':')
   self.assertEqual(binary,str(native/'bin/mgbfs'))
   self.assertEqual(paths[:3],[str(native/'lib'),str(pure/'nvidia/cuda_runtime/lib'),str(pure/'nvidia/nccl/lib')])
   self.assertEqual(paths[-1],'/usr/local/cuda/compat')
   if '/usr/lib/x86_64-linux-gnu' in paths:self.assertGreater(paths.index('/usr/lib/x86_64-linux-gnu'),2)
if __name__=='__main__':unittest.main()
