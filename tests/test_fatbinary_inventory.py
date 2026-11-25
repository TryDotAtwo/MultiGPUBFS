import unittest
from scripts.fatbinary_inventory import verify_bundled_inventory, verify_inventory
class FatbinaryInventory(unittest.TestCase):
 def test_every_module_needs_every_advertised_target(self):
  sass='\n'.join(f'ELF file: {m}.sm_{a}.cubin' for m in ('hash','generic_graph') for a in (75,80,100,120))
  ptx='\n'.join(f'PTX file: {m}.1.sm_80.ptx' for m in ('hash','generic_graph'))
  value=verify_inventory(sass,ptx,[75,80,100,120],[80]);self.assertEqual(value['native_module_counts']['100'],2)
  with self.assertRaisesRegex(RuntimeError,'CUBIN_MISSING_SM_100'):verify_inventory(sass.replace('generic_graph.sm_100.cubin','wrong'),ptx,[75,80,100,120],[80])
  with self.assertRaisesRegex(RuntimeError,'PTX_MISSING_SM_80'):verify_inventory(sass,ptx.replace('hash.1.sm_80.ptx','wrong'),[75,80,100,120],[80])

 def test_bundled_build_requires_pack4_and_exact_targets(self):
  arches = [75, 80, 86, 89, 90, 100, 103, 120]
  device_modules = ('hash', 'bitmap_pack4_convert', 'bitmap_pack4_owner', 'bitmap_pack4_route')
  sass = '\n'.join(f'ELF file: {module}.sm_{arch}.cubin' for arch in arches
                   for module in device_modules)
  ptx = '\n'.join(f'PTX file: {module}.1.sm_80.ptx' for module in device_modules)
  inventory = verify_bundled_inventory(sass, ptx, arches, [80])
  self.assertEqual(inventory['native_module_counts']['75'], len(device_modules))
  without_pack4_owner = '\n'.join(line for line in sass.splitlines()
                                   if 'bitmap_pack4_owner' not in line)
  with self.assertRaisesRegex(RuntimeError, 'PACK4_MISSING_MODULES: bitmap_pack4_owner'):
   verify_bundled_inventory(without_pack4_owner, ptx, arches, [80])
  with self.assertRaisesRegex(RuntimeError, 'PTX_MISSING_SM_80: bitmap_pack4_route'):
   verify_bundled_inventory(sass, ptx.replace('bitmap_pack4_route.1.sm_80.ptx', ''), arches, [80])
  with self.assertRaisesRegex(RuntimeError, 'PTX_UNEXPECTED_TARGETS: \\[86\\]'):
   verify_bundled_inventory(sass, ptx + '\nPTX file: hash.1.sm_86.ptx', arches, [80])
