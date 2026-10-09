import tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition
from multigpubfs.autotune import choose_profile
class ProfilePolicy(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.native=Path(self.tmp.name)/'native';self.native.write_bytes(b'test-native');self.g=GraphDefinition.permutation([[1,0]],[0,1]);self.calls=0
 def tearDown(self):self.tmp.cleanup()
 def admission(self,g,d,c,shards,n,e,tmp):return {'devices':[0,1],'plan':{'capacity':50000,'batch':64}}
 def run_case(self,times):
  def pilot(*a,**kw):
   i=self.calls;self.calls+=1
   return {'layer_sizes':[1,20000,20000,20000,1],'layer_seconds':[.04,times[i],times[i],times[i]],'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
  unavailable=subprocess.CompletedProcess([],1,'unavailable','')
  with patch('multigpubfs.autotune._admit',side_effect=self.admission),patch('multigpubfs.autotune._system_info',return_value=unavailable),patch('multigpubfs.launch.run_graph',side_effect=pilot):return choose_profile(self.g,None,None,60,str(self.native),{})
 def test_margin_keeps_baseline(self):self.assertEqual(self.run_case([1,.98,1.1,.98,1.1])['shards'],1)
 def test_substantial_gain_selects_batch(self):
  p=self.run_case([1,.9,.8,.7,1.1]);self.assertEqual((p['shards'],p['batch_fraction']),(4,.25))
 def test_memory_inadmissible_alternatives_skip_gpu_pilots(self):
  def admit(*args):
   if args[3]!=1:raise RuntimeError('PROFILE_ADMISSION_FAILED: REQUESTED_CAPACITY_EXCEEDS_ADMISSION')
   return self.admission(*args)
  with patch('multigpubfs.autotune._admit',side_effect=admit),patch('multigpubfs.autotune._system_info',return_value=subprocess.CompletedProcess([],1,'','')),patch('multigpubfs.launch.run_graph',side_effect=AssertionError('unneeded pilot')):
   p=choose_profile(self.g,None,50000,60,str(self.native),{});self.assertEqual(p['status'],'NO_ADMITTED_ALTERNATIVE_CONSERVATIVE_PROFILE')
 def test_real_admission_error_not_hidden(self):
  def admit(*args):
   if args[3]!=1:raise RuntimeError('PROFILE_ADMISSION_FAILED: CUDA_STATUS_700')
   return self.admission(*args)
  with patch('multigpubfs.autotune._admit',side_effect=admit),patch('multigpubfs.autotune._system_info',return_value=subprocess.CompletedProcess([],1,'','')):
   with self.assertRaisesRegex(RuntimeError,'CUDA_STATUS_700'):choose_profile(self.g,None,None,60,str(self.native),{})
 def test_wide_parent_mode_is_measured_and_selected(self):
  self.g=GraphDefinition.permutation([list(range(1,25))+[0]],list(range(25)))
  p=self.run_case([1,.95,.9,.8,.6,.5,.8]);self.assertEqual((p['shards'],p['batch_fraction'],p['transport']),(4,.25,'parent'));self.assertEqual(len(p['pilots']),7)
 def test_packed_does_not_measure_duplicate_parent_transport(self):
  p=self.run_case([1,.9,.8,.7,1.1]);self.assertEqual(len(p['pilots']),5);self.assertEqual(p['transport'],'full')
 def test_transport_override_is_respected(self):
  from multigpubfs.autotune import _transport_variants
  wide=GraphDefinition.permutation([list(range(1,25))+[0]],list(range(25)))
  self.assertTrue(all(p[2]=='parent' for p in _transport_variants(wide,{'MGBFS_GENERIC_TRANSPORT':'parent'})))
  self.assertTrue(all(p[2]=='full' for p in _transport_variants(wide,{'MGBFS_GENERIC_TRANSPORT':'full'})))
  with self.assertRaisesRegex(ValueError,'INVALID_TRANSPORT'):_transport_variants(wide,{'MGBFS_GENERIC_TRANSPORT':'typo'})
 def test_radix_order_can_win_without_changing_transport(self):
  p=self.run_case([1,.9,.8,.7,.5]);self.assertEqual((p['shards'],p['transport'],p['candidate_order']),(4,'full','radix'))
if __name__=='__main__':unittest.main()
