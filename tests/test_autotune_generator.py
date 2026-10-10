import unittest,tempfile,subprocess
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition
from multigpubfs.generation import gemm_supported
from multigpubfs.autotune import choose_profile
class GeneratorPolicy(unittest.TestCase):
 def graph(self,value=1,mod=251,n=8):return GraphDefinition.matrix(n,1,[([value]*(n*n),mod)],[1]*n)
 def test_exact_gate(self):
  self.assertTrue(gemm_supported(self.graph()))
  self.assertTrue(gemm_supported(GraphDefinition.permutation([[1,0]],[0,1])))
  for g in [self.graph(-1),self.graph(256),self.graph(mod=0),self.graph(mod=257),self.graph(n=7)]:self.assertFalse(gemm_supported(g))
 def test_measured_generator_selection(self):
  for gain,want in [(.4,'gemm'),(.98,'cuda'),(1.4,'cuda')]:
   with tempfile.TemporaryDirectory() as d:
    native=Path(d)/'native';native.write_bytes(b'fixture')
    def admit(g,devices,c,sh,n,e,tmp):return {'devices':[0,1],'plan':{'capacity':50000,'batch':128,'history_algorithm':'HASH','owner_lanes':0,'generator_backend':e.get('MGBFS_GENERIC_GENERATOR','cuda')}}
    def pilot(*a,**kw):
     value=gain if kw['_native_env'].get('MGBFS_GENERIC_GENERATOR')=='gemm' else 1
     return {'layer_sizes':[1,20000,20000,20000,1],'layer_seconds':[.04,value,value,value],'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
    with patch('multigpubfs.autotune._gemm_hardware_available',return_value=True),patch('multigpubfs.autotune._admit',side_effect=admit),patch('multigpubfs.autotune._system_info',return_value=subprocess.CompletedProcess([],1,'','')),patch('multigpubfs.launch.run_graph',side_effect=pilot):
     p=choose_profile(self.graph(),None,None,60,str(native),{'MGBFS_GENERIC_HISTORY':'hash'},allow_specialized=False);self.assertEqual(p['generator_backend'],want);self.assertTrue(any(x.get('generator_backend')=='gemm' for x in p['pilots']))
if __name__=='__main__':unittest.main()
