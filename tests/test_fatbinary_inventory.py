import unittest
from scripts.fatbinary_inventory import verify_inventory
class FatbinaryInventory(unittest.TestCase):
 def test_every_module_needs_every_advertised_target(self):
  sass='\n'.join(f'ELF file: {m}.sm_{a}.cubin' for m in ('hash','generic_graph') for a in (75,80,100,120))
  ptx='\n'.join(f'PTX file: {m}.1.sm_80.ptx' for m in ('hash','generic_graph'))
  value=verify_inventory(sass,ptx,[75,80,100,120],[80]);self.assertEqual(value['native_module_counts']['100'],2)
  with self.assertRaisesRegex(RuntimeError,'CUBIN_MISSING_SM_100'):verify_inventory(sass.replace('generic_graph.sm_100.cubin','wrong'),ptx,[75,80,100,120],[80])
  with self.assertRaisesRegex(RuntimeError,'PTX_MISSING_SM_80'):verify_inventory(sass,ptx.replace('hash.1.sm_80.ptx','wrong'),[75,80,100,120],[80])
