import unittest,tempfile,json,os
from unittest.mock import patch
from pathlib import Path
from multigpubfs.size_profiles import select_size_profiles,save_online_profiles
class SizeProfiles(unittest.TestCase):
 def pilots(self):
  a=dict(batch=64,generator_backend='cuda',layer_sizes=[1,100,200,40000,50000,100000],layer_seconds=[.1,.01,.01,.04,.08])
  return a,dict(a,batch=256,generator_backend='gemm',layer_seconds=[.1,.02,.02,.01,.02])
 def test_distinct_small_large(self):
  p=select_size_profiles(self.pilots(),2);self.assertEqual([v['generator_backend'] for v in p],['cuda','gemm'])
 def test_unknown_large_is_not_extrapolated(self):
  a,b=self.pilots();a=dict(a,layer_sizes=[1,100,200,300],layer_seconds=[.1,.02,.02]);b=dict(b,layer_sizes=a['layer_sizes'],layer_seconds=[.1,.01,.01]);p=select_size_profiles([a,b],2)
  self.assertFalse(p[1]['measured']);self.assertEqual(p[1]['generator_backend'],'cuda')
 def test_single_candidate_not_measured(self):self.assertFalse(any(v['measured'] for v in select_size_profiles([self.pilots()[0]],2)))
 def test_bad_prefix_rejected(self):
  a,b=self.pilots();b=dict(b,layer_sizes=[1,99,200,40000,50000,100000])
  with self.assertRaisesRegex(RuntimeError,'CORRECTNESS'):select_size_profiles([a,b],2)
 def test_marginal_switch_keeps_baseline(self):
  a,b=self.pilots();b=dict(b,layer_seconds=[v*.98 for v in a['layer_seconds']]);self.assertTrue(all(p['generator_backend']=='cuda' for p in select_size_profiles([a,b],2)))
 def test_live_cache_keeps_small_profile(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,MGBFS_PROFILE_CACHE=d):
   profile={'status':'MEASURED_FRONTIER_SIZE_PROFILES','identity':{'fixture':1},'batch':256,'size_profiles':[{'minimum_frontier':0,'batch':64,'generator_backend':'cuda'}]}
   report={'autotune':profile,'online_size_profile_events':[{'winner':1,'choices':[[256,False],[64,True]],'frontier_global':40000}]}
   save_online_profiles(report);p=json.loads(next(Path(d).glob('size-*.json')).read_text())
   self.assertEqual(p['size_profiles'][0]['batch'],64);self.assertEqual(p['size_profiles'][1]['generator_backend'],'gemm');self.assertEqual(p['size_profiles'][1]['status'],'LIVE_CHUNK_EMPIRICAL_PROFILE')
class SizeBackendPolicy(unittest.TestCase):
 def test_explicit_generic_never_enters_specialized(self):
  from multigpubfs import GraphDefinition
  from multigpubfs.autotune import choose_size_profile
  g=GraphDefinition.permutation([[1,0]],[0,1]);base={'shards':1,'status':'FIXTURE'}
  with patch('multigpubfs.autotune.choose_profile',return_value=base) as choose,patch('multigpubfs.autotune._admit',return_value={'devices':[0],'plan':{}}):
   p=choose_size_profile(g,[0],None,30,'fixture',{},allow_specialized=False)
   self.assertFalse(choose.call_args.kwargs['allow_specialized']);self.assertEqual(p['size_tuning_status'],'NATIVE_SIZE_PROFILE_CAPABILITY_UNAVAILABLE')
 def test_short_run_has_no_extra_native_probes(self):
  from multigpubfs import GraphDefinition
  from multigpubfs.autotune import choose_size_profile
  g=GraphDefinition.permutation([[1,0]],[0,1]);base={'shards':1,'status':'SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE'}
  with patch('multigpubfs.autotune.choose_profile',return_value=base),patch('multigpubfs.autotune._admit') as admit:
   self.assertEqual(choose_size_profile(g,[0],None,1,'fixture',{}),base);admit.assert_not_called()
if __name__=='__main__':unittest.main()

