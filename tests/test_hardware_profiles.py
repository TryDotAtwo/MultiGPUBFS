import unittest
from multigpubfs.hardware_profiles import startup_policy,shard_candidates
class HardwareProfiles(unittest.TestCase):
 def test_real_properties_change_candidates(self):
  plan={'capacity':900000000,'elements':14,'state_bytes':1}
  a=startup_policy([{'sm_count':28,'l2_bytes':3<<20}],plan)
  b=startup_policy([{'sm_count':148,'l2_bytes':126<<20}],plan)
  self.assertNotEqual(a['shards'],b['shards']);self.assertGreater(b['probe_capacity_per_rank'],a['probe_capacity_per_rank'])
 def test_probe_does_not_cap_production(self):
  p={'capacity':900000000,'elements':14,'state_bytes':1};v=startup_policy([{'sm_count':148,'l2_bytes':126<<20}],p)
  self.assertLess(v['probe_capacity_per_rank'],p['capacity']);self.assertEqual(p['capacity'],900000000)
 def test_tiny_admitted_graph_bounds_probe(self):
  v=startup_policy([{'sm_count':148,'l2_bytes':126<<20}],{'capacity':24,'elements':4,'state_bytes':1});self.assertEqual(v['probe_capacity_per_rank'],24)
 def test_missing_properties_stays_conservative(self):self.assertEqual(shard_candidates([],{'capacity':1}),[1,4,16])

class AdmissionInventory(unittest.TestCase):
 def test_snapshot_uses_cpu_plan_not_repeated_device_probe(self):
  import tempfile,json,subprocess
  from unittest.mock import patch
  from multigpubfs import GraphDefinition
  from multigpubfs.autotune import _admit
  calls=[]
  def query(command,**kwargs):calls.append(command);return subprocess.CompletedProcess(command,0,json.dumps({'plan':{'capacity':24}}),'')
  with tempfile.TemporaryDirectory() as d,patch('multigpubfs.autotune.native_query',side_effect=query):
   _admit(GraphDefinition.permutation([[1,0]],[0,1]),[0],None,4,'native',{'_MGBFS_PROFILE_INVENTORY':'[{"device":0,"free_bytes":123}]'},d)
  self.assertEqual(calls[0][1],'graph-plan');self.assertEqual(calls[0][-1],'auto')
