import json, unittest
import test_runtime_variant_diagnostics as fixtures
class KernelIdentity(unittest.TestCase):
 def fixture(self):
  value=fixtures.RuntimeVariants();value.setUp();self.addCleanup(value.doCleanups)
  for i,row in enumerate(value.metadata['variants']):
   row['cuda_source_tree']=str(i)*40;row['cuda_kernel_source_sha256']='a'*64
   proof=json.loads((value.root/row['receipt']).read_text());proof.update(cuda_source_tree=row['cuda_source_tree'],cuda_kernel_source_sha256=row['cuda_kernel_source_sha256'])
   value.put(row['receipt'],json.dumps(proof).encode());row['receipt_sha256']=value.hashes[row['receipt']]
  value.save_metadata();return value
 def test_same_kernel_inputs_with_separate_build_default_trees(self):
  self.assertEqual(self.fixture().inventory()['status'],'AVAILABLE')
 def test_declared_kernel_fingerprint_must_match_checked_receipt(self):
  v=self.fixture();v.metadata['variants'][1]['cuda_kernel_source_sha256']='c'*64;v.save_metadata()
  with self.assertRaisesRegex(RuntimeError,'SOURCE'):v.inventory()
 def test_actual_different_kernel_input_is_rejected(self):
  v=self.fixture();row=v.metadata['variants'][1];row['cuda_kernel_source_sha256']='c'*64
  proof=json.loads((v.root/row['receipt']).read_text());proof['cuda_kernel_source_sha256']='c'*64
  v.put(row['receipt'],json.dumps(proof).encode());row['receipt_sha256']=v.hashes[row['receipt']];v.save_metadata()
  with self.assertRaisesRegex(RuntimeError,'SOURCE_MISMATCH'):v.inventory()
 def test_partial_fingerprint_matrix_is_rejected(self):
  v=self.fixture();row=v.metadata['variants'][1];del row['cuda_kernel_source_sha256'];v.save_metadata()
  with self.assertRaisesRegex(RuntimeError,'SOURCE'):v.inventory()
