import unittest
from multigpubfs import GraphDefinition
from multigpubfs.generation import gemm_supported

class PermutationGemmDomain(unittest.TestCase):
 def graph(self,n,labels=None):return GraphDefinition.permutation([list(range(1,n))+[0]],list(range(n)) if labels is None else labels)
 def test_exact_byte_permutations(self):
  for n in (1,7,14,24,33,64):self.assertTrue(gemm_supported(self.graph(n)))
 def test_full_unsigned_alphabet(self):self.assertTrue(gemm_supported(self.graph(14,[255]*14)))
 def test_wide_or_negative_alphabet(self):
  self.assertFalse(gemm_supported(self.graph(14,[256]*14)))
  self.assertFalse(gemm_supported(self.graph(14,[-1]*14)))
 def test_overwide_permutation(self):self.assertFalse(gemm_supported(self.graph(65)))
