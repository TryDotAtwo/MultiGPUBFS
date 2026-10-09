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
class ExactCapability(unittest.TestCase):
 def test_unproved_key_domain_rejects_before_native_launch(self):
  from multigpubfs.specialized import exact_specialized_supported
  from multigpubfs import run_graph
  from unittest.mock import patch
  self.assertFalse(exact_specialized_supported())
  n=26;g=GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))],list(range(n)))
  with patch('multigpubfs.launch.native_runtime',return_value=('native',{})):
   with self.assertRaisesRegex(RuntimeError,'SPECIALIZED_LOSSLESS_KEY_DOMAIN_UNSUPPORTED'):run_graph(g,'/tmp/not-created-exact-test',backend='shard_ab_hash')
class LosslessDomain(unittest.TestCase):
 def test_entire_normalized_word_must_fit(self):
  from multigpubfs.specialized import exact_specialized_supported
  for n,repeated,expected in [(14,1,True),(15,4,True),(25,1,True),(26,1,False),(128,128,True),(128,1,False)]:
   g=GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))],[min(j,n-repeated) for j in range(n)])
   self.assertEqual(exact_specialized_supported(g),expected,(n,repeated))
 def test_legacy_binary_cannot_silently_ignore_key_mode(self):
  from unittest.mock import patch
  import subprocess
  from multigpubfs.specialized import require_lossless_native
  for text in ('{}','{"schema":1,"lossless_bitpack128_feistel_v1":false}','invalid'):
   with patch('multigpubfs.specialized.subprocess.run',return_value=subprocess.CompletedProcess([],0,text,'')):
    with self.assertRaisesRegex(RuntimeError,'SPECIALIZED_NATIVE_LOSSLESS_KEYS_UNAVAILABLE'):require_lossless_native('native',{})
if __name__=='__main__':unittest.main()
