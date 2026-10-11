import unittest
from types import SimpleNamespace
from unittest.mock import patch
from multigpubfs import autotune as a,generic_session as g
class Reuse(unittest.TestCase):
 def test_live_geometry_generation_and_environment(self):
  pool=SimpleNamespace(generation=1);graph=SimpleNamespace(digest=lambda:'graph')
  profile={'backend':'generic','shards':1,'size_profiles':[{'minimum_frontier':0,'batch':8,'generator_backend':'cuda'}],'measured':True,'batch':8,'generator_backend':'cuda','history_algorithm':'HASH'}
  admission={'devices':[0,1],'plan':{'capacity':100,'batch':8},'rank_plans':[],'inventory':[]}
  with patch.object(g,'_active',pool),patch.object(a,'_dependency_identity',return_value={'resolved_dependencies':[]}),patch.object(a,'_choose_size_profile_uncached',side_effect=lambda *args,**kw:dict(profile)) as choose,patch.object(a,'_admit',return_value=admission) as admit:
   call=lambda env={}:a.choose_size_profile(graph,[0,1],100,60,'/bin/true',env)
   call();result=call();self.assertTrue(result['session_profile_hit']);self.assertEqual(choose.call_count,1);self.assertEqual(admit.call_count,2)
   admit.return_value=dict(admission,plan={'capacity':80,'batch':4});call();self.assertEqual(choose.call_count,2)
   pool.generation=2;call();self.assertEqual(choose.call_count,3)
   call({'MGBFS_GENERIC_TRANSPORT':'parent'});self.assertEqual(choose.call_count,4)
   admit.side_effect=RuntimeError('CUDA failure')
   with self.assertRaisesRegex(RuntimeError,'CUDA failure'):call({'MGBFS_GENERIC_TRANSPORT':'parent'})
   self.assertEqual(choose.call_count,4)
