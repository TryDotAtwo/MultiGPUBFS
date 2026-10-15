"""Regression coverage for B200 discovery and multi-architecture packaging."""
import json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import diagnostics,runtime_variants
try:
 from multigpubfs import architecture_support as arch
except ImportError:
 arch=None

def driver_step(count=2,runtime=2):
 return {'status':'PASS','result':{'status':'PASS','data':{'classification':'DRIVER_RUNTIME_READY','driver':{'visible_devices':count,'devices':[{'logical_id':i,'compute_capability':[10,0]} for i in range(count)]},'runtime':{'visible_devices':runtime}}}}

class UniversalArchitectureRegression(unittest.TestCase):
 def test_production_and_runtime_control_cover_a100_h100_b200_b300_rtx(self):
  for name in ('shared-sm120','static-sm120'):
   self.assertTrue({75,80,86,89,90,100,103,120}.issubset(runtime_variants.SPECS[name][1]))
 def coverage(self,manifest,devices):
  self.assertIsNotNone(arch,'Missing explicit native/PTX compatibility evidence')
  return arch.coverage(manifest,devices)
 def test_b200_does_not_accept_sm120_cubin_as_compatible(self):
  r=self.coverage({'architectures':[75,86,120],'ptx_architectures':[]},[{'logical_id':0,'compute_capability':[10,0]}])
  self.assertEqual(r['devices'][0]['code'],'UNSUPPORTED')
 def test_compatible_ptx_is_a_jit_path_not_native_code(self):
  r=self.coverage({'architectures':[75,86,120],'ptx_architectures':[80]},[{'logical_id':0,'compute_capability':[10,0]}])
  self.assertEqual(r['devices'][0]['code'],'PTX_JIT')
 def test_b300_can_use_same_major_b200_cubin(self):
  r=self.coverage({'architectures':[100],'ptx_architectures':[]},[{'logical_id':0,'compute_capability':[10,3]}])
  self.assertEqual(r['devices'][0]['code'],'NATIVE')
 def test_unknown_ptx_inventory_is_not_a_claim_of_incompatibility(self):
  r=self.coverage({'architectures':[75,86,120]},[{'logical_id':0,'compute_capability':[10,0]}])
  self.assertEqual(r['devices'][0]['code'],'UNKNOWN_PTX')
 def test_new_production_has_native_b200_without_jit(self):
  r=self.coverage({'architectures':runtime_variants.SPECS[runtime_variants.PRODUCTION][1],'ptx_architectures':[80]},[{'logical_id':0,'compute_capability':[10,0]}])
  self.assertEqual(r['devices'][0]['code'],'NATIVE')

class IndependentDiscoveryRegression(unittest.TestCase):
 def recover(self,step,requested=None):
  fn=getattr(diagnostics,'recover_visible_devices',None)
  self.assertIsNotNone(fn,'Native discovery timeout prevents verified configuration search')
  return fn(step,requested)
 def test_independent_driver_and_runtime_can_supply_logical_ids(self):
  self.assertEqual(self.recover(driver_step()),[0,1])
  self.assertEqual(self.recover(driver_step(),[1]),[1])
 def test_mismatch_missing_driver_and_nvml_counts_are_rejected(self):
  for s in (driver_step(2,1),{'status':'FAIL'}, {'status':'PASS','result':{'data':{'visible_devices':2}}}):
   self.assertIsNone(self.recover(s))
 def test_outside_visible_gpu_and_duplicate_ids_are_rejected(self):
  self.assertIsNone(self.recover(driver_step(),[2]));self.assertIsNone(self.recover(driver_step(),[0,0]))
 def test_failed_native_count_still_attempts_oracle_verified_recipe(self):
  class Runner:
   def __init__(self,*a,**kw):self.report={'steps':[]};self.deep_cleanup_blocked=False;self.root=Path(a[0]);self.root.mkdir(parents=True,exist_ok=True)
   def component(self,name,stage,**kw):
    result=driver_step() if stage=='driver' else {'status':'TIMEOUT' if stage=='count' else 'PASS','result':{'data':{}}}
    self.report['steps'].append(dict(result,name=name));return result
   def skip(self,*a):pass
   def save(self):pass
   def finish(self):return self.report
  with tempfile.TemporaryDirectory() as d,patch.object(diagnostics,'Runner',Runner),patch('multigpubfs.server_recipe.find_working',return_value={'status':'READY'}) as search:
   r=diagnostics.run_diagnostics(output=Path(d)/'out')
  search.assert_called_once();self.assertEqual(search.call_args.args[1],[0,1])
  self.assertEqual(r['working_configuration']['status'],'READY')

class LoaderTraceRegression(unittest.TestCase):
 def test_timeout_preserves_partial_loader_evidence(self):
  from multigpubfs import system_diagnostics as system
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'trace';root.mkdir()
   error=subprocess.TimeoutExpired(['native','graph-count'],6,stderr=b'find library=libcuda.so.1\ncalling init: libmgbfs_cuda.so\n')
   with patch('multigpubfs.native_distribution.native_runtime',return_value=('/native',{})),patch.object(system.subprocess,'run',side_effect=error):
    value=system.loader_trace(root,60)
   self.assertEqual(value['status'],'FAIL');self.assertIn('libmgbfs_cuda.so',(root/'loader.log').read_text())
   self.assertTrue(value['timed_out'])

if __name__=='__main__':unittest.main()
