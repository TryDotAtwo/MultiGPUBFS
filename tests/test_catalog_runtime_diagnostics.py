import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import deep_diagnostics as deep

class CatalogRuntimeReuse(unittest.TestCase):
 def test_one_catalog_worker_resolves_driver_and_dependencies_once(self):
  from multigpubfs.deep_diagnostics import catalog_cuda
  cache={};config={'device':3,'_catalog_runtime_cache':cache};runtime=(object(),object(),object())
  with patch('multigpubfs.diagnostics._cuda',return_value=runtime) as resolve:
   self.assertIs(catalog_cuda(config),runtime);self.assertIs(catalog_cuda(config),runtime)
  resolve.assert_called_once_with(3,True)

 def test_cache_is_local_to_one_batch_and_device(self):
  from multigpubfs.deep_diagnostics import catalog_cuda
  cache={};a=(object(),object(),object());b=(object(),object(),object())
  with patch('multigpubfs.diagnostics._cuda',side_effect=[a,b,a,a]) as resolve:
   self.assertIs(catalog_cuda({'device':2,'_catalog_runtime_cache':cache}),a)
   self.assertIs(catalog_cuda({'device':5,'_catalog_runtime_cache':cache}),b)
   self.assertIs(catalog_cuda({'device':2,'_catalog_runtime_cache':cache}),a)
   self.assertIs(catalog_cuda({'device':2}),a);self.assertIs(catalog_cuda({'device':2}),a)
  self.assertEqual(resolve.call_count,4)

 def test_batch_passes_the_same_runtime_cache_to_every_constructor(self):
  from multigpubfs.collective_diagnostics import catalog_batch
  cases=[{'family':'PermutationGroups','constructor':str(i),'kwargs':{},'unsupported':None} for i in range(3)];observed=[]
  def probe(config,root):observed.append(config.get('_catalog_runtime_cache'));return {'all_generators_exact_readback_verified':True}
  with tempfile.TemporaryDirectory() as d,patch.object(deep,'catalog_probe',side_effect=probe):catalog_batch({'device':0,'catalog':cases},Path(d))
  self.assertTrue(all(isinstance(x,dict) for x in observed));self.assertEqual(len({id(x) for x in observed}),1)
