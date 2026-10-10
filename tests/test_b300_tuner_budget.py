import unittest,tempfile,subprocess,json
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition
from multigpubfs.autotune import choose_profile
from scripts.b300_current_production import verify_oracle
class BudgetTests(unittest.TestCase):
 def test_multigpu_bootstrap_slower_than_one_second_can_reach_prefix(self):
  with tempfile.TemporaryDirectory() as d:
   native=Path(d)/'native';native.write_bytes(b'test');g=GraphDefinition.permutation([[1,0]],[0,1]);durations=[]
   def admit(*args):return {'devices':[0,1],'plan':{'capacity':50000,'batch':64}}
   def pilot(*a,**kw):
    limit=kw['max_seconds'];durations.append(limit)
    if limit<3:return {'layer_sizes':[1,3],'layer_seconds':[2.5],'status':'INCOMPLETE','reason':'DEADLINE'}
    return {'layer_sizes':[1,20000,20000,20000,1],'layer_seconds':[2.5,.01,.01,.01],'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
   with patch('multigpubfs.autotune._admit',side_effect=admit),patch('multigpubfs.autotune._system_info',return_value=subprocess.CompletedProcess([],1,'','')),patch('multigpubfs.launch.run_graph',side_effect=pilot):
    p=choose_profile(g,None,None,60,str(native),{'MGBFS_GENERIC_HISTORY':'hash'},allow_specialized=False)
   self.assertTrue(p['measured']);self.assertGreaterEqual(p['common_depth'],4);self.assertTrue(all(v>=3 for v in durations))
 def test_incomplete_startup_does_not_claim_wrong_completed_states(self):
  g=GraphDefinition.permutation([[1,0]],[0,1])
  with patch.object(type(g),'exact_layers',side_effect=AssertionError('oracle should not run for incomplete')):
   with self.assertRaisesRegex(RuntimeError,'STARTUP_INCOMPLETE_DEADLINE'):verify_oracle({'status':'INCOMPLETE','reason':'DEADLINE'},g,Path('/unused'),2)
 def test_empty_pilot_budget_is_conservative_not_min_empty_crash(self):
  with tempfile.TemporaryDirectory() as d:
   native=Path(d)/'native';native.write_bytes(b'test');g=GraphDefinition.permutation([[1,0]],[0,1]);ticks=[0]
   def clock():ticks[0]+=10;return ticks[0]
   def admit(*args):return {'devices':[0,1],'plan':{'capacity':50000,'batch':64}}
   with patch('multigpubfs.autotune._admit',side_effect=admit),patch('multigpubfs.autotune._system_info',return_value=subprocess.CompletedProcess([],1,'','')),patch('multigpubfs.autotune.time.monotonic',side_effect=clock),patch('multigpubfs.launch.run_graph',side_effect=AssertionError('deadline reached')):
    p=choose_profile(g,None,None,10,str(native),{'MGBFS_GENERIC_HISTORY':'hash'},allow_specialized=False)
   self.assertEqual(p['status'],'TUNING_BUDGET_EXHAUSTED_CONSERVATIVE_PROFILE');self.assertFalse(p['measured'])
if __name__=='__main__':unittest.main()
