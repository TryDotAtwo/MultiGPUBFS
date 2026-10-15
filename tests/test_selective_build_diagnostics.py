import importlib.util,pathlib,unittest
class NativeTargets(unittest.TestCase):
 def targets(self,*args):
  path=pathlib.Path(__file__).resolve().parents[1]/'scripts/native_build_targets.py'
  self.assertTrue(path.is_file(),'Selective native target configuration is missing')
  s=importlib.util.spec_from_file_location('native_targets',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.targets(*args)
 def test_exact_target_has_only_its_native_image(self):
  for target in (75,80,86,89,90,100,103,120):
   arch,ptx,cmake=self.targets('shared-sm120',str(target));self.assertEqual(arch,[target]);self.assertEqual(ptx,[]);self.assertEqual(cmake,str(target)+'-real')
 def test_universal_keeps_all_native_and_ordinary_ptx(self):
  arch,ptx,cmake=self.targets('shared-sm120',None);self.assertEqual(arch,[75,80,86,89,90,100,103,120]);self.assertEqual(ptx,[80]);self.assertIn('80-virtual',cmake)
 def test_unpublished_or_legacy_target_override_rejected(self):
  for variant,target in [('shared-sm120','99'),('shared-legacy','100'),('static-sm120','75')]:
   with self.assertRaises(ValueError):self.targets(variant,target)
