import copy, os
from unittest.mock import patch
import test_native_assets_diagnostics as fixture

class SeparateExecutable(fixture.Assets):
 def split_index(self):
  idx=copy.deepcopy(self.index)
  data=fixture.archive({'bin/mgbfs':b'new executable'})
  self.payload['executable.tar.gz']=data
  idx['components']['executable']={'url':fixture.BASE+'executable.tar.gz','sha256':fixture.h(data),'size':len(data),'files':{'bin/mgbfs':{'sha256':fixture.h(b'new executable'),'size':len(b'new executable'),'mode':0o755}}}
  idx['executable']='executable'
  idx['native_metadata']['native_binary_source_commit']='e'*40
  return idx
 def test_reuses_old_common_prefetches_new_cli_and_prepares_offline(self):
  self.manager().prefetch([100])
  before=len(self.calls)
  idx=self.split_index()
  response=self.manager(idx).prefetch([100])
  self.assertEqual([name for name,_ in self.calls[before:]],['executable.tar.gz'])
  self.assertEqual(response['downloaded_bytes'],idx['components']['executable']['size'])
  with patch.dict(os.environ,{'MGBFS_NATIVE_OFFLINE':'1'}):
   root=self.manager(idx).prepare(fixture.hardware(100))
  self.assertEqual((root/'bin/mgbfs').read_bytes(),b'new executable')
  self.assertTrue(os.access(root/'bin/mgbfs',os.X_OK))
  self.assertEqual((root/'lib/libcudart.so.12').read_bytes(),b'private runtime')
 def test_executable_index_requires_executable_alone_and_correct_mode(self):
  for files in ({},{'bin/mgbfs':{'sha256':fixture.h(b'x'),'size':1,'mode':0o644}},{'other':{'sha256':fixture.h(b'x'),'size':1,'mode':0o644}}):
   idx=self.split_index();idx['components']['executable']['files']=files
   with self.assertRaises(self.module.AssetError):self.manager(idx)
 def test_corrupt_executable_never_assembles(self):
  idx=self.split_index();self.manager(idx).prefetch([100])
  c=idx['components']['executable'];(self.cache/'components'/c['sha256']/'bin/mgbfs').write_bytes(b'corrupt')
  with self.assertRaises(self.module.AssetError):self.manager(idx).prepare(fixture.hardware(100))
