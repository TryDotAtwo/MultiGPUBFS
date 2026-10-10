import hashlib,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from multigpubfs import autotune,generic_session,native_distribution
class ResidentIdentityInvalidation(unittest.TestCase):
 def test_same_size_same_mtime_atomic_library_replacement_invalidates(self):
  with tempfile.TemporaryDirectory() as d:
   binary=Path(d)/'mgbfs';binary.write_bytes(b'bin');library=Path(d)/'libmgbfs_cuda.so';library.write_bytes(b'a');old=library.stat()
   def resolve(*args,**kwargs):return SimpleNamespace(stdout=f'libmgbfs_cuda.so => {library} (0x0)')
   with patch.object(generic_session,'_active',SimpleNamespace()),patch.object(autotune.subprocess,'run',side_effect=resolve) as run:
    autotune._dependency_cache.clear();a=autotune._dependency_identity(str(binary),{});b=autotune._dependency_identity(str(binary),{});self.assertEqual(a,b);self.assertEqual(run.call_count,1)
    replacement=Path(d)/'replacement';replacement.write_bytes(b'b');os.utime(replacement,ns=(old.st_atime_ns,old.st_mtime_ns));replacement.replace(library);c=autotune._dependency_identity(str(binary),{});self.assertEqual(run.call_count,2);self.assertEqual(c['cuda_library_sha256'],hashlib.sha256(b'b').hexdigest())
 def test_native_bundle_digest_rechecks_replacement(self):
  with tempfile.TemporaryDirectory() as d,patch.object(generic_session,'_active',SimpleNamespace()):
   p=Path(d)/'file';p.write_bytes(b'a');a=native_distribution._file_digest(p);replacement=Path(d)/'replacement';replacement.write_bytes(b'b');replacement.replace(p);b=native_distribution._file_digest(p);self.assertNotEqual(a,b)
