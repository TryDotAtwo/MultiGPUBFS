from types import SimpleNamespace
from unittest.mock import patch
from copy import deepcopy
from multigpubfs import GraphDefinition
import multigpubfs.generic_session as session

def fixture():
 g=GraphDefinition.permutation([[1,0]],[0,1]);plan={'history_algorithm':'HASH','batch':64,'capacity':100,'generator_backend':'cuda'}
 geometry={'devices':[0],'plan':plan,'rank_plans':[plan]}
 class Job:
  response={'ok':True,'cache':{'owned_raw_complete':True,'cached_streams':2,'cached_events':2,'owned_raw_plan':plan,'owned_raw_generator':'cuda'}}
  def poll(self):return 0
 pool=SimpleNamespace(workers=[SimpleNamespace(job=Job())],select=lambda *a:None)
 return g,geometry,pool

def test_complete_owned_geometry_is_rebased_without_query():
 g,geometry,pool=fixture()
 with patch.object(session,'_active',pool):
  got=session.owned_resource_admission(g,'native',{'MGBFS_GENERIC_GENERATOR':'cuda'},geometry)
 assert got['owned_resource_admission'] and got['graph_digest']==g.digest()
 assert 'graph_digest' not in geometry

def test_incomplete_or_different_geometry_is_rejected():
 for field,value in [('owned_raw_complete',False),('owned_raw_plan',{}),('owned_raw_generator','gemm')]:
  g,geometry,pool=fixture();pool.workers[0].job.response=deepcopy(pool.workers[0].job.response);pool.workers[0].job.response['cache'][field]=value
  with patch.object(session,'_active',pool):assert session.owned_resource_admission(g,'native',{},geometry) is None

def test_active_or_failed_rank_is_not_owned_acceptance():
 for status in [None,1]:
  g,geometry,pool=fixture();pool.workers[0].job.poll=lambda:status
  with patch.object(session,'_active',pool):assert session.owned_resource_admission(g,'native',{},geometry) is None

def test_health_failure_propagates_without_restart():
 g,geometry,pool=fixture()
 def failed(*a):raise RuntimeError('FAILED_GROUP_NO_RESTART')
 pool.select=failed
 with patch.object(session,'_active',pool):
  import pytest
  with pytest.raises(RuntimeError,match='NO_RESTART'):session.owned_resource_admission(g,'native',{},geometry)


def test_same_lrx_action_can_borrow_profile_without_admission_or_pilots():
 import multigpubfs.autotune as tune
 n=9;generators=[list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))]
 source=GraphDefinition.permutation(generators,list(range(n)))
 target=GraphDefinition.permutation(generators,list(range(n-1))+[n-2])
 plan={'history_algorithm':'HASH','batch':64,'capacity':65536,'generator_backend':'cuda'}
 geometry={'devices':[0],'plan':plan,'rank_plans':[plan]}
 saved=dict(status='MEASURED_FRONTIER_SIZE_PROFILES',identity={'graph':source.digest()},backend='generic',shards=1,batch=64,generator_backend='cuda',history_algorithm='HASH',size_profiles=[{'minimum_frontier':0,'batch':64,'generator_backend':'cuda','measured':True}],measured=True)
 pool=SimpleNamespace(generation=1,size_profile_results={})
 with patch.object(session,'_active',pool),patch.object(tune,'_session_profile_key',side_effect=lambda p,g,*a,**k:(1,g.digest(),'context')):
  alias=tune._lrx_resource_alias_key((1,source.digest(),'context'),source);pool.size_profile_aliases={alias:(saved,geometry)}
  with patch.object(session,'owned_resource_admission',return_value=dict(geometry,graph_digest=target.digest(),owned_resource_admission=True)),patch.object(session,'remember_launch_admission'),patch.object(tune,'_admit') as admit,patch.object(tune,'_choose_size_profile_uncached') as choose:
   result=tune.choose_size_profile(target,[0],65536,60,'native',{})
  admit.assert_not_called();choose.assert_not_called();assert result['owned_resource_hit'];assert result['identity']['graph']==target.digest();assert result['source_graph_digest']==source.digest();assert not result['measured']
