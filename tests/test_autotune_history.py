import tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition
from multigpubfs.autotune import choose_profile
class HistoryPolicy(unittest.TestCase):
 def test_forced_sorted_geometry(self):
  from multigpubfs.autotune import _transport_variants
  g=GraphDefinition.permutation([[1,0]],[0,1])
  v=_transport_variants(g,{'MGBFS_GENERIC_HISTORY':'sorted','MGBFS_GENERIC_OWNER_LANES':'8'})
  self.assertTrue(all(sh>=8 and order=='none' for sh,_,_,order in v))
  with self.assertRaisesRegex(ValueError,'INVALID_HISTORY_ALGORITHM'):_transport_variants(g,{'MGBFS_GENERIC_HISTORY':'typo'})
 def test_sorted_lane_winner_and_override(self):
  with tempfile.TemporaryDirectory() as d:
   native=Path(d)/'native';native.write_bytes(b'fixture');g=GraphDefinition.permutation([[1,0]],[0,1])
   def admit(g,devices,c,sh,n,e,tmp):return {'devices':[0,1],'plan':{'capacity':50000,'batch':64,'history_algorithm':'SORTED_RUNS' if e.get('MGBFS_GENERIC_HISTORY')=='sorted' else 'HASH','owner_lanes':int(e.get('MGBFS_GENERIC_OWNER_LANES','0'))}}
   def pilot(*a,**kw):
    e=kw['_native_env'];v=.1 if e.get('MGBFS_GENERIC_HISTORY')=='sorted' and e.get('MGBFS_GENERIC_OWNER_LANES')=='2' else 1
    return {'layer_sizes':[1,20000,20000,20000,1],'layer_seconds':[.04,v,v,v],'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
   with patch('multigpubfs.autotune._admit',side_effect=admit),patch('multigpubfs.autotune._system_info',return_value=subprocess.CompletedProcess([],1,'','')),patch('multigpubfs.launch.run_graph',side_effect=pilot):
    p=choose_profile(g,None,None,60,str(native),{},allow_specialized=False);self.assertEqual((p['history_algorithm'],p['owner_lanes'],p['batch']),('SORTED_RUNS',2,64));self.assertEqual(len(p['pilots']),9)
    p=choose_profile(g,None,None,60,str(native),{'MGBFS_GENERIC_HISTORY':'hash'},allow_specialized=False);self.assertEqual(p['history_algorithm'],'HASH');self.assertEqual(len(p['pilots']),5)
if __name__=='__main__':unittest.main()
