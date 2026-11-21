"""Owner-page validation and cache separation without CUDA."""
import unittest
from unittest.mock import patch
from multigpubfs import GraphDefinition
from multigpubfs.autotune import _configuration_digest
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.launch import run_graph

GRAPH=GraphDefinition.permutation([[1,0]],[0,1])
class BitmapOwnerPageApiTests(unittest.TestCase):
 def test_default_and_supported_values(self):
  self.assertIsNone(NativeOptions().bitmap_owner_page)
  for value in (0,1,8,64,256,1<<20):
   self.assertEqual(NativeOptions(memory_backend='permutation_bitmap',bitmap_owner_page=value).bitmap_owner_page,value)
 def test_invalid_values_and_backend_fail_before_native(self):
  for value in (-1,3,1<<21,True,'8',1.0):
   with self.subTest(value=value):
    with self.assertRaisesRegex(ValueError,'INVALID_BITMAP_OWNER_PAGE'):
     NativeOptions(memory_backend='permutation_bitmap',bitmap_owner_page=value)
    with patch('multigpubfs.launch.native_runtime') as native:
     with self.assertRaisesRegex(ValueError,'INVALID_BITMAP_OWNER_PAGE'):
      run_graph(GRAPH,'/tmp/not-created',memory_backend='permutation_bitmap',bitmap_owner_page=value)
     native.assert_not_called()
  with self.assertRaisesRegex(ValueError,'BITMAP_OWNER_PAGE_REQUIRES_BITMAP'):
   NativeOptions(memory_backend='segmented24',bitmap_owner_page=8)
  with patch('multigpubfs.launch.native_runtime') as native:
   with self.assertRaisesRegex(ValueError,'BITMAP_OWNER_PAGE_REQUIRES_BITMAP'):
    run_graph(GRAPH,'/tmp/not-created',memory_backend='segmented24',bitmap_owner_page=8)
   native.assert_not_called()
 def test_cached_profiles_separate_layout_and_recovery(self):
  values=[{}, {'MGBFS_SEGMENT_BITMAP_OWNER_PAGE':'0'}, {'MGBFS_SEGMENT_BITMAP_OWNER_PAGE':'64'}, {'MGBFS_SEGMENT_BITMAP_OWNER_PAGE':'64','MGBFS_SEGMENT_BATCH_RECOVERY':'1'}]
  self.assertEqual(len({_configuration_digest(v) for v in values}),len(values))

if __name__=='__main__':unittest.main()
