import unittest
from multigpubfs import GraphDefinition
from multigpubfs.distributed_launch import _external_variants,_phase_environment,_merge_local_sorted_plans
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
class LocalSortedPlans(unittest.TestCase):
 def parts(self,world):
  fields={'world':world,'shards':8,'elements':25,'state_bytes':1,'history_layers':3,'history_algorithm':'SORTED_RUNS','owner_lanes':4,'batch':64,'queue_capacity':32,'generator_bytes':300,'parent_transport':True,'sort_candidates':False}
  return [{'graph_digest':'g','device':i%8,'free_bytes':100000,'total_bytes':200000,'plan':dict(fields,capacity=100+i,sorted_owner_bytes=1000+i)} for i in range(world)]
 def test_weighted_capacity_and_duplicate_local_ids_at_128_ranks(self):
  parts=self.parts(128);v=_merge_local_sorted_plans(parts,'g',None)
  self.assertEqual(v['owner_cuts'][0],0);self.assertEqual(v['owner_cuts'][-1],1<<32);self.assertEqual(len(v['rank_plans']),128)
  self.assertTrue(all(a<b for a,b in zip(v['owner_cuts'],v['owner_cuts'][1:])))
 def test_different_native_scratch_sizes_are_preserved(self):
  p=self.parts(2);v=_merge_local_sorted_plans(p,'g',None);self.assertNotEqual(v['rank_plans'][0]['sorted_owner_bytes'],v['rank_plans'][1]['sorted_owner_bytes'])
 def test_shared_geometry_mismatch_rejected(self):
  p=self.parts(2);p[1]['plan']['batch']=32
  with self.assertRaisesRegex(RuntimeError,'TRANSPORT_GEOMETRY'):_merge_local_sorted_plans(p,'g',None)
 def test_rank_graph_mismatch_rejected(self):
  p=self.parts(2);p[1]['graph_digest']='other'
  with self.assertRaisesRegex(RuntimeError,'IDENTITY'):_merge_local_sorted_plans(p,'g',None)
 def test_explicit_capacity_mismatch_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'CAPACITY'):_merge_local_sorted_plans(self.parts(2),'g',100)
if __name__=='__main__':unittest.main()
