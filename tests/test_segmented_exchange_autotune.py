"""Bounded segmented exchange selection must measure both modes and honor overrides."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multigpubfs import autotune
from multigpubfs.graph_definition import GraphDefinition

class ExchangeAutotune(unittest.TestCase):
 def measure(self, forced=None, capable=True):
  with tempfile.TemporaryDirectory() as tmp:
   native=Path(tmp)/'native'; native.write_bytes(b'unit test native')
   graph=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2]); calls=[]
   def admit(graph,devices,capacity,shards,native,env,directory,**kwargs):
    return {'devices':[0,1], 'inventory':[], 'plan':{'capacity':100000,'batch':131072,'history_algorithm':'HASH','owner_lanes':0}}
   def run(graph,target,**kw):
    env=kw['_native_env']; mode=env.get('MGBFS_SEGMENT_EXACT_EXCHANGE','0'); lane=int(env['MGBFS_SEGMENT_OWNER_LANES'])
    calls.append((kw['shards'],lane,kw['_batch'],mode))
    seconds=.1 if mode=='1' else 1.
    receipt={'owner_lanes':lane}; receipt.update({'exact_exchange':mode=='1'} if capable else {})
    return {'backend':'EXPERIMENTAL_PERMUTATION_BITMAP','segment_memory_by_rank':[receipt]*2,'layer_sizes':[1,20000,20000,20000,1],'layer_seconds':[seconds]*4,'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
   env={'MGBFS_GENERIC_HISTORY':'hash','MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_EXPERIMENTAL_BITMAP':'1','_MGBFS_SEGMENT_EXCHANGE_CAPABLE':'1' if capable else '0'}
   if forced is not None:env['MGBFS_SEGMENT_EXACT_EXCHANGE']=forced
   policy={'shards':[1,4,8], 'probe_capacity_per_rank':50000}; missing=subprocess.CompletedProcess([],1,'','')
   with patch.object(autotune,'_admit',side_effect=admit), patch.object(autotune,'_system_info',return_value=missing), patch.object(autotune,'_dependency_identity',return_value=None), patch('multigpubfs.hardware_profiles.startup_policy',return_value=policy), patch('multigpubfs.hardware_profiles.small_permutation_workload',return_value=False), patch('multigpubfs.launch.run_graph',side_effect=run):
    result=autotune.choose_profile(graph,[0,1],None,90,str(native),env,allow_specialized=False,fixed_shards=4)
   return result,calls
 def test_auto_measures_same_geometry_and_carries_winner(self):
  result,calls=self.measure()
  self.assertIn(calls[0][:3]+('1',),calls)
  self.assertEqual(result['segment_exact_exchange'],True)
  self.assertEqual(autotune._profile_environment({},result)['MGBFS_SEGMENT_EXACT_EXCHANGE'],'1')
 def test_matched_exchange_runs_on_parallel_baseline(self):
  result,calls=self.measure()
  self.assertEqual(calls[0][1],4)
  self.assertEqual(calls[1][:3],calls[0][:3])
  self.assertEqual({calls[0][3],calls[1][3]}, {'0','1'})
 def test_forced_modes_are_not_overridden(self):
  for mode in ('0','1'):
   with self.subTest(mode=mode):
    result,calls=self.measure(mode)
    self.assertTrue(calls)
    self.assertTrue(all(c[3]==mode for c in calls))
    self.assertEqual(autotune._profile_environment({'MGBFS_SEGMENT_EXACT_EXCHANGE':mode},result)['MGBFS_SEGMENT_EXACT_EXCHANGE'],mode)
 def test_old_native_only_uses_padded_without_claiming_exact(self):
  result,calls=self.measure(capable=False)
  self.assertTrue(calls and all(c[3]=='0' for c in calls))
  self.assertFalse(result['segment_exact_exchange'])
  with self.assertRaisesRegex(RuntimeError,'NATIVE_SEGMENTED_EXCHANGE_UNAVAILABLE'):
   self.measure(forced='1',capable=False)
 def test_invalid_mode_fails_before_probes(self):
  with self.assertRaisesRegex(ValueError,'INVALID_SEGMENTED_EXCHANGE_MODE'):
   self.measure('other')

if __name__=='__main__': unittest.main()
