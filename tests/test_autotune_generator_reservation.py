import os,subprocess,unittest
from unittest.mock import patch
from multigpubfs import autotune
import test_autotune_candidate_failures as harness

class OptionalGeneratorSizeAdmission(unittest.TestCase):
 def setUp(self):
  self.h=harness.MeasuredSelectionAfterCandidateFailure();self.h.setUp();self.calls=[]
  p=self.h.report(2);p.update(backend='generic',shards=4,transport='full',candidate_order='none',history_algorithm='SORTED_RUNS',owner_lanes=1,batch=64,generator_backend='cuda')
  gemm=dict(p,generator_backend='gemm',layer_seconds=[.001]*4)
  self.global_profile=dict(p,status='MEASURED_EQUAL_PREFIX_GPU_PROFILE',measured=True,batch_fraction=1.,pilots=[p,gemm])
 def tearDown(self):self.h.tearDown()
 def choose(self,admit,capacity=50000):
  with patch.object(autotune,'choose_profile',return_value=self.global_profile),patch.object(autotune,'_admit',side_effect=admit),patch('multigpubfs.generation.gemm_supported',return_value=True),patch.object(autotune,'_gemm_hardware_available',return_value=True),patch.object(autotune,'_dependency_identity',return_value=None),patch.object(autotune,'_system_info',return_value=subprocess.CompletedProcess([],1,'','')):
   return autotune._choose_size_profile_uncached(self.h.graph,[0,1],capacity,90,str(self.h.native),self.h.env,allow_specialized=False)
 def admission(self,g,devices,capacity,shards,native,env,directory):
  backend=env['MGBFS_GENERIC_GENERATOR'];self.calls.append((backend,capacity))
  if backend=='gemm':raise RuntimeError("GENERIC_SESSION_LOCAL_PLAN_FAILED {'error': 'REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION'}")
  return {'size_profile_capability':1,'devices':[0,1],'inventory':[{'sm_count':28}],'plan':{'capacity':50000,'batch':64,'history_algorithm':'SORTED_RUNS','owner_lanes':1,'generator_backend':backend}}
 def test_optional_gemm_capacity_rejection_preserves_measured_selected_cuda_geometry(self):
  try:result=self.choose(self.admission)
  except RuntimeError as e:self.fail('Optional unadmitted generator aborted measured choice: '+str(e))
  self.assertTrue(result['measured']);self.assertTrue(result['selection_measured'])
  self.assertEqual(result['generator_backend'],'cuda');self.assertEqual(result['history_algorithm'],'SORTED_RUNS')
  self.assertTrue(all(p['generator_backend']=='cuda' for p in result['size_profiles']))
  self.assertEqual(len(result['size_generator_rejections']),1)
  self.assertEqual(result['size_generator_rejections'][0]['generator'],'gemm')
  self.assertIn('REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION',result['size_generator_rejections'][0]['reason'])
  self.assertEqual(self.calls,[('cuda',50000),('gemm',50000)])
 def test_optional_rejection_during_second_common_capacity_admission_is_bounded(self):
  def admit(g,devices,capacity,shards,native,env,directory):
   if env['MGBFS_GENERIC_GENERATOR']=='gemm' and capacity is None:
    self.calls.append(('gemm',None));return {'devices':[0,1],'plan':{'capacity':50000,'batch':32,'history_algorithm':'SORTED_RUNS','owner_lanes':1,'generator_backend':'gemm'}}
   return self.admission(g,devices,capacity,shards,native,env,directory)
  try:result=self.choose(admit,capacity=None)
  except RuntimeError as e:self.fail('Second optional admission aborted measured choice: '+str(e))
  self.assertTrue(result['measured']);self.assertEqual(result['generator_backend'],'cuda')
  self.assertEqual(self.calls,[('cuda',None),('gemm',None),('gemm',50000)])
 def test_compatible_source_tiers_cannot_restore_an_unadmitted_generator(self):
  self.global_profile.update(status='REUSED_COMPATIBLE_GPU_PROFILE',source_pilots=self.global_profile['pilots'])
  try:result=self.choose(self.admission)
  except RuntimeError as e:self.fail('Optional admission aborted compatible measured choice: '+str(e))
  self.assertTrue(result['measured']);self.assertTrue(all(p['generator_backend']=='cuda' for p in result['size_profiles']))
  self.assertEqual(len(result['size_generator_rejections']),1)
 def test_unrecognized_error_is_not_hidden_as_optional_resource_rejection(self):
  def admit(*args,**kwargs):
   if args[5]['MGBFS_GENERIC_GENERATOR']=='gemm':raise RuntimeError('GENERIC_CUDA_700')
   return self.admission(*args,**kwargs)
  with self.assertRaisesRegex(RuntimeError,'GENERIC_CUDA_700'):self.choose(admit)
 def test_selected_generator_admission_failure_is_not_downgraded(self):
  self.global_profile['generator_backend']='gemm'
  with self.assertRaisesRegex(RuntimeError,'REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION'):self.choose(self.admission)
  self.assertEqual(self.calls,[('gemm',50000)])

 def test_rejected_optional_modes_have_stable_serialized_cache_identity(self):
  cache=self.h.root/'disk-profiles'
  with patch.dict(os.environ,MGBFS_PROFILE_CACHE=str(cache)),patch.object(autotune,'choose_profile',return_value=self.global_profile),patch.object(autotune,'_admit',side_effect=self.admission),patch('multigpubfs.generation.gemm_supported',return_value=True),patch.object(autotune,'_gemm_hardware_available',return_value=True),patch.object(autotune,'_dependency_identity',return_value={'test_library':'fixed'}),patch.object(autotune,'_system_info',return_value=subprocess.CompletedProcess([],0,'stable-hardware','')):
   first=autotune._choose_size_profile_uncached(self.h.graph,[0,1],50000,90,str(self.h.native),self.h.env,allow_specialized=False)
   second=autotune._choose_size_profile_uncached(self.h.graph,[0,1],50000,90,str(self.h.native),self.h.env,allow_specialized=False)
  self.assertFalse(first['cache_hit']);self.assertTrue(second['cache_hit'],'Disk profile identity must survive JSON round-trip')
  self.assertTrue(second['measured']);self.assertTrue(all(p['generator_backend']=='cuda' for p in second['size_profiles']))

if __name__=='__main__':unittest.main()
