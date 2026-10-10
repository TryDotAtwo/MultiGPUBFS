import unittest
from multigpubfs import GraphDefinition
from multigpubfs.distributed_launch import _external_variants,_phase_environment
class ExternalHistoryPolicy(unittest.TestCase):
 def setUp(self):self.g=GraphDefinition.permutation([[1,2,0]],list(range(3)))
 def test_unforced_adds_four_sorted_lane_geometries(self):
  v=_external_variants(self.g,{})
  self.assertEqual({p[4] for p in v},{'hash','sorted'})
  self.assertEqual({p[5] for p in v if p[4]=='sorted'},{1,2,4,8})
  self.assertTrue(all(p[3]=='none' and p[0]>=p[5] for p in v if p[4]=='sorted'))
 def test_forced_hash_excludes_sorted(self):self.assertEqual({p[4] for p in _external_variants(self.g,{'MGBFS_GENERIC_HISTORY':'hash'})},{'hash'})
 def test_forced_sorted_respects_lane_and_removes_radix(self):
  v=_external_variants(self.g,{'MGBFS_GENERIC_HISTORY':'sorted','MGBFS_GENERIC_OWNER_LANES':'8'})
  self.assertTrue(all(p[4]=='sorted' and p[5]==8 and p[0]>=8 and p[3]=='none' for p in v))
 def test_forced_radix_excludes_sorted_history_candidates(self):self.assertEqual({p[4] for p in _external_variants(self.g,{'MGBFS_GENERIC_SORT':'radix'})},{'hash'})
 def test_phase_environment_isolation(self):
  e={'MGBFS_GENERIC_HISTORY':'hash','MGBFS_GENERIC_OWNER_LANES':'1'}
  v=_phase_environment(e,'parent','none','sorted',8)
  self.assertEqual(e['MGBFS_GENERIC_HISTORY'],'hash');self.assertEqual(v['MGBFS_GENERIC_OWNER_LANES'],'8');self.assertEqual(v['MGBFS_GENERIC_HISTORY'],'sorted')
if __name__=='__main__':unittest.main()
