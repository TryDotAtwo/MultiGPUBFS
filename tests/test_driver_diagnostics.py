import unittest
from unittest.mock import patch
from multigpubfs.driver_diagnostics import classify,_driver
class DriverDiagnosticTests(unittest.TestCase):
 def sample(self):return {'driver':{'loaded':True,'resolved_library':'/usr/lib/libcuda.so.1','api_version':12020,'initialization':{'code':0},'device_query':{'code':0},'visible_devices':2},'runtime':{'loaded':True,'runtime_version':12090,'visible_devices':2,'cudaGetDeviceCount':{'code':0}}}
 def test_lower_minor_is_not_called_incompatible(self):self.assertEqual(classify(self.sample())[0],'DRIVER_RUNTIME_READY')
 def test_runtime_35_does_not_prove_old_major(self):
  data=self.sample();data['runtime']['cudaGetDeviceCount']['code']=35;self.assertEqual(classify(data)[0],'RUNTIME_DRIVER_COMPATIBILITY_FAILED')
 def test_older_major_and_35(self):
  data=self.sample();data['driver']['api_version']=11080;data['runtime']['cudaGetDeviceCount']['code']=35;self.assertEqual(classify(data)[0],'DRIVER_MAJOR_TOO_OLD')
 def test_stub_and_kernel_mismatch(self):
  data=self.sample();data['driver']['resolved_library']='/cuda/stubs/libcuda.so.1';data['driver']['initialization']={'code':34,'name':'CUDA_ERROR_STUB_LIBRARY'};self.assertEqual(classify(data)[0],'DRIVER_STUB_LOADED')
  data=self.sample();data['driver']['initialization']={'code':803,'name':'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH'};self.assertEqual(classify(data)[0],'DRIVER_KERNEL_LIBRARY_MISMATCH')
 def test_no_gpu_and_missing_driver(self):
  data=self.sample();data['driver']['visible_devices']=0;self.assertEqual(classify(data)[0],'NO_VISIBLE_GPU')
  data['driver']={'loaded':False};self.assertEqual(classify(data)[0],'DRIVER_LIBRARY_UNAVAILABLE')
 def test_missing_driver_loader_error_is_retained(self):
  with patch('multigpubfs.driver_diagnostics.C.CDLL',side_effect=OSError('missing driver fixture')):
   self.assertEqual(_driver()['error'],'missing driver fixture')
 def test_runtime_failure_retained(self):
  data=self.sample();data['runtime']['cudaGetDeviceCount']={'code':999,'description':'unknown driver error'};self.assertEqual(classify(data)[0],'RUNTIME_DEVICE_QUERY_FAILED')

 def test_directory_name_alone_does_not_prove_stub(self):
  data=self.sample();data['driver']['resolved_library']='/custom/stubs/libcuda.so.1';self.assertEqual(classify(data)[0],'DRIVER_RUNTIME_READY')
