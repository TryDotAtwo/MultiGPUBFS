import unittest,tempfile,copy
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.profile_families import identity,store,load,transferred

def graph(n,r=1):return GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))],list(range(n-r+1))+[n-r]*(r-1))
class SharedProfiles(unittest.TestCase):
 def setUp(self):
  self.plan={'state_bytes':1,'history_layers':3,'batch':1000}
  self.exact={'selector_sha256':'code','configuration_digest':'cfg','native_dependencies':{'cuda':'hash'},'native_sha256':'native','devices':[0,1],'hardware':'two3060','topology':'link','allow_specialized':False,'environment':{},'graph':graph(9).digest()}
  self.key=identity(graph(9),self.plan,self.exact)
  self.saved={'status':'MEASURED_EQUAL_PREFIX_GPU_PROFILE','measured':True,'backend':'generic','shards':4,'batch_fraction':.25,'generator_backend':'cuda','history_algorithm':'HASH','owner_lanes':0,'transport':'full','candidate_order':'none','scores_seconds':[.1,.2],'identity':self.exact}
 def test_compatible_n_r(self):
  self.assertEqual(self.key,identity(graph(9,2),self.plan,self.exact));self.assertEqual(self.key,identity(graph(10),self.plan,self.exact))
 def test_hardware_and_code_invalidation(self):
  for name in ('selector_sha256','configuration_digest','native_dependencies','native_sha256','devices','hardware','topology','allow_specialized','environment'):
   x=copy.deepcopy(self.exact);x[name]='changed';self.assertNotEqual(self.key,identity(graph(9),self.plan,x),name)
 def test_encoding_history_shape_invalidation(self):
  for field in ('state_bytes','history_layers'):
   p=dict(self.plan);p[field]+=1;self.assertNotEqual(self.key,identity(graph(9),p,self.exact))
  self.assertNotEqual(self.key,identity(graph(17),self.plan,self.exact))
 def test_capacity_recomputed_batch_scaled(self):
  x=transferred(self.saved,self.exact,{'batch':400},.03)
  self.assertEqual(x['batch'],100);self.assertFalse(x['measured']);self.assertEqual(x['status'],'REUSED_COMPATIBLE_GPU_PROFILE');self.assertNotIn('capacity',x)
 def test_atomic_roundtrip_and_invalid_payload(self):
  with tempfile.TemporaryDirectory() as root:
   store(root,self.key,self.saved);self.assertEqual(load(root,self.key),self.saved)
   for field,value in [('shards',True),('history_algorithm','wrong'),('transport','wrong'),('scores_seconds',[float('nan')]),('backend','wrong'),('generator_backend','gemm')]:
    p=dict(self.saved);p[field]=value;store(root,self.key,p);self.assertIsNone(load(root,self.key),field)
 def test_unmeasured_not_shared(self):
  with tempfile.TemporaryDirectory() as root:
   store(root,self.key,dict(self.saved,measured=False));self.assertIsNone(load(root,self.key))
 def test_partial_inventory_disables(self):self.assertIsNone(identity(graph(9),{},self.exact))
