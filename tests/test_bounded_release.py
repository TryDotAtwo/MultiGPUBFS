import copy,json,tempfile,unittest
from pathlib import Path
from multigpubfs.native_assets import AssetManager,AssetError
from scripts.package_bounded_nccl_release import retire_incompatible_controls

class ReleaseControlTests(unittest.TestCase):
 def setUp(self):
  self.index=json.loads((Path(__file__).resolve().parents[1]/'multigpubfs/native_assets.json').read_text())
 def manager(self,index):
  with tempfile.TemporaryDirectory() as root:AssetManager(index,root)
 def test_retire_mismatched_controls_without_weakening_production_gate(self):
  index=copy.deepcopy(self.index);old=copy.deepcopy(index['diagnostics'])
  for name in index['production'].values():index['components'][name]['build']['cuda_kernel_source_sha256']='a'*64
  if old:
   with self.assertRaisesRegex(AssetError,'KERNEL_INPUT_MISMATCH'):self.manager(index)
  retired=retire_incompatible_controls(index,'a'*64)
  self.assertEqual(len(retired),len(old));self.assertEqual(index['diagnostics'],{})
  for name in old.values():self.assertNotIn(name,index['components'])
  self.manager(index)
  name=index['production']['universal'];index['components'][name]['build']['cuda_kernel_source_sha256']='b'*64
  with self.assertRaisesRegex(AssetError,'KERNEL_INPUT_MISMATCH'):self.manager(index)
 def test_matching_controls_are_preserved(self):
  index=copy.deepcopy(self.index)
  fingerprint=index['components'][next(iter(index['production'].values()))]['build']['cuda_kernel_source_sha256']
  before=copy.deepcopy(index['diagnostics'])
  self.assertEqual(retire_incompatible_controls(index,fingerprint),[])
  self.assertEqual(index['diagnostics'],before);self.manager(index)
if __name__=='__main__':unittest.main()
