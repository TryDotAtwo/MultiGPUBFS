"""Diagnostic-only variant isolation, integrity and causal limits. No GPU claims."""
import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import runtime_variants as variants
from multigpubfs import native_distribution

class RuntimeVariants(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name)/'_native';self.root.mkdir()
  self.hashes={}
  def put(name,data):
   path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
   self.hashes[name]=hashlib.sha256(data).hexdigest();return path
  self.put=put;put('bin/mgbfs',b'executable');put('lib/libmgbfs_cuda.so',b'baseline')
  put('lib/libcudart.so.12',b'runtime');put('lib/libnccl.so.2',b'nccl')
  entries=[]
  for identifier,(runtime,architectures) in variants.SPECS.items():
   library='lib/libmgbfs_cuda.so' if identifier==variants.BASELINE else 'diagnostic_variants/'+identifier+'/libmgbfs_cuda.so'
   if identifier!=variants.BASELINE:put(library,identifier.encode())
   receipt='diagnostic_variants/'+identifier+'/build-receipt.json'
   put(receipt,json.dumps({'id':identifier,'runtime':runtime,'architectures':architectures,'source_commit':'a'*40,'cuda_source_tree':'b'*40}).encode())
   entries.append({'id':identifier,'runtime':runtime,'architectures':architectures,'library':library,'sha256':self.hashes[library],'receipt':receipt,'source_commit':'a'*40,'cuda_source_tree':'b'*40})
  self.metadata={'schema':1,'variants':entries,'production_default':variants.BASELINE}
  self.save_metadata()
 def save_metadata(self):
  self.put('diagnostic_variants/variants.json',json.dumps(self.metadata).encode())
  (self.root/'manifest.json').write_text(json.dumps({'schema':1,'bundled_runtime':[{}],'sha256':self.hashes}))
 def inventory(self):return variants.inventory(self.root,environ={})
 def test_complete_verified_factorial_inventory(self):
  value=self.inventory();self.assertEqual(value['status'],'AVAILABLE');self.assertEqual(len(value['variants']),4)
  self.assertEqual({(x['runtime'],120 in x['architectures']) for x in value['variants']},{('Static',False),('Static',True),('Shared',False),('Shared',True)})
 def test_corrupt_library_stops_selection(self):
  (self.root/'diagnostic_variants/shared-sm120/libmgbfs_cuda.so').write_bytes(b'corrupt')
  with self.assertRaisesRegex(RuntimeError,'CHECKSUM'):self.inventory()
 def test_corrupt_metadata_is_detected(self):
  (self.root/'diagnostic_variants/variants.json').write_text('{}')
  with self.assertRaisesRegex(RuntimeError,'CHECKSUM'):self.inventory()
 def test_metadata_cannot_redirect_library(self):
  self.metadata['variants'][1]['library']='../outside.so';self.save_metadata()
  with self.assertRaisesRegex(RuntimeError,'PATH'):self.inventory()
 def test_wrong_factor_label_is_rejected(self):
  self.metadata['variants'][1]['runtime']='Static';self.save_metadata()
  with self.assertRaisesRegex(RuntimeError,'FACTOR'):self.inventory()
 def test_mixed_cuda_sources_are_rejected(self):
  self.metadata['variants'][1]['cuda_source_tree']='c'*40;self.save_metadata()
  with self.assertRaisesRegex(RuntimeError,'SOURCE'):self.inventory()
 def test_missing_variant_is_not_silent_partial_matrix(self):
  self.metadata['variants'].pop();self.save_metadata()
  with self.assertRaisesRegex(RuntimeError,'MATRIX'):self.inventory()
 def test_custom_production_executable_is_not_overridden(self):
  value=variants.inventory(self.root,environ={'MGBFS_EXECUTABLE':'/custom/mgbfs'})
  self.assertEqual(value['status'],'UNAVAILABLE');self.assertIn('custom',value['reason'].lower())
 def test_child_environment_preserves_visibility_and_parent(self):
  parent={'CUDA_VISIBLE_DEVICES':'2,5','LD_LIBRARY_PATH':'/host/driver','TOKEN':'private'}
  env=variants.environment('shared-sm120',self.inventory(),parent)
  self.assertEqual(parent,{'CUDA_VISIBLE_DEVICES':'2,5','LD_LIBRARY_PATH':'/host/driver','TOKEN':'private'})
  self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2,5');self.assertEqual(env['LD_LIBRARY_PATH'].split(':')[0],str(self.root/'diagnostic_variants/shared-sm120'))
  self.assertEqual(env['MGBFS_EXECUTABLE'],str(self.root/'bin/mgbfs'))
 def test_production_resolver_ignores_diagnostic_alternative(self):
  with patch.object(native_distribution,'__file__',str(self.root.parent/'native_distribution.py')),patch.dict(os.environ,{},clear=True):
   native,env=native_distribution.native_runtime()
  self.assertEqual(env['LD_LIBRARY_PATH'].split(':')[0],str(self.root/'lib'))
  self.assertNotIn('diagnostic_variants',env['LD_LIBRARY_PATH'])
 def test_actual_loader_path_must_match_selected_bytes(self):
  value=self.inventory()
  with self.assertRaisesRegex(RuntimeError,'RESOLUTION'):
   variants.confirm_resolution('shared-sm120',value,{'libmgbfs_cuda.so':str(self.root/'lib/libmgbfs_cuda.so')})
 def test_cases_cover_both_devices_and_native_paths_without_visibility_override(self):
  cases=variants.cases([0,1]);self.assertEqual(len(cases),24)
  self.assertEqual(len({x['name'] for x in cases}),24)
  self.assertEqual({x['config']['variant'] for x in cases},set(variants.SPECS))
  self.assertTrue(all('CUDA_VISIBLE_DEVICES' not in x['config'] for x in cases))
  self.assertEqual({x['config']['kind'] for x in cases},{'api','component','bfs'})
 def test_comparison_reports_effect_without_claiming_unique_cause(self):
  steps=[{'name':x,'status':'TIMEOUT' if x==variants.BASELINE else 'PASS','diagnostic_case':{'variant':x,'kind':'api','device':0},'seconds':1} for x in variants.SPECS]
  result=variants.compare(steps);self.assertFalse(result['root_cause_confirmed'])
  self.assertEqual(result['comparisons'][0]['outcome'],'VARIANT_DEPENDENT')
 def test_budget_and_missing_results_are_explicit(self):
  result=variants.compare([{'name':'x','status':'SKIPPED','reason':'GLOBAL_DIAGNOSTIC_BUDGET_EXHAUSTED','diagnostic_case':{'variant':variants.BASELINE,'kind':'api','device':0}}])
  self.assertEqual(result['comparisons'][0]['outcome'],'INCOMPLETE')
  self.assertFalse(result['root_cause_confirmed'])
