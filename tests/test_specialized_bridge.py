import unittest
from multigpubfs import GraphDefinition
from multigpubfs.specialized import match_lrx
class Match(unittest.TestCase):
 def graph(self,start):
  n=len(start);return GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))],start)
 def test_lossless_relabelled_multiset(self):
  g=self.graph([-9,1<<40,7,7]);m=match_lrx(g);self.assertEqual((m['n'],m['r'],m['labels']),(4,2,[-9,1<<40,7]));self.assertEqual(m['order'],12)
 def test_distinct_arbitrary_root(self):self.assertEqual(match_lrx(self.graph([9,2,-8,5]))['labels'],[9,2,-8,5])
 def test_other_repeated_pattern_is_not_substituted(self):self.assertIsNone(match_lrx(self.graph([0,0,1,2])))
 def test_directed_subset_is_not_substituted(self):self.assertIsNone(match_lrx(GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,1,2,3])))
 def test_matrix_is_not_coerced(self):self.assertIsNone(match_lrx(GraphDefinition.matrix(1,1,[([1],7)],[0])))
if __name__=='__main__':unittest.main()
