"""Private runtime resolution and integrity contracts; no GPU claims."""
import hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import multigpubfs.native_distribution as distribution

class PrivateRuntime(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.package=Path(self.temp.name)/'multigpubfs';self.root=self.package/'_native'
        self.files={'bin/mgbfs':b'exe','lib/libmgbfs_cuda.so':b'cuda','lib/libnccl.so.2':b'nccl','lib/libcudart.so.12':b'runtime'}
        for name,data in self.files.items():
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        manifest=dict(schema=1,bundled_runtime=[{'wheel':'fixture'}],sha256={name:hashlib.sha256(data).hexdigest() for name,data in self.files.items()})
        (self.root/'manifest.json').write_text(json.dumps(manifest))
    def resolve(self):
        with patch.object(distribution,'__file__',str(self.package/'native_distribution.py')),patch.dict('os.environ',{},clear=True):
            return distribution.native_runtime()
    def test_private_lib_directory_precedes_system_and_excludes_torch(self):
        native,env=self.resolve();self.assertEqual(native,str(self.root/'bin/mgbfs'))
        self.assertEqual(env['LD_LIBRARY_PATH'].split(':')[0],str(self.root/'lib'))
        self.assertNotIn('nvidia/nccl',env['LD_LIBRARY_PATH'])
    def test_corrupt_private_nccl_is_rejected(self):
        (self.root/'lib/libnccl.so.2').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'NATIVE_BUNDLE_CHECKSUM_lib/libnccl.so.2'):self.resolve()
    def test_missing_private_cudart_is_rejected(self):
        (self.root/'lib/libcudart.so.12').unlink()
        with self.assertRaises(FileNotFoundError):self.resolve()

if __name__=='__main__':unittest.main()
