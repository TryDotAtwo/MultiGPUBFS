import unittest
from multigpubfs.graph_definition import GraphDefinition
from multigpubfs.workload_geometry import coxeter_frontier_bound
class Geometry(unittest.TestCase):
 def graph(self,n):
  gs=[]
  for i in range(n-1):
   g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gs.append(g)
  return GraphDefinition.permutation(gs,list(range(n)))
 def test_exact_layers(self):
  for n in range(2,8):
   g=self.graph(n);bound=coxeter_frontier_bound(g);layers=g.exact_layers(5040)
   self.assertEqual(bound['maximum_frontier'],max(map(len,layers)))
   self.assertEqual(bound['order'],sum(map(len,layers)))
 def test_unrelated_and_repeats_rejected(self):
  g=self.graph(4);v=g.to_dict();v['start']=[0,0,1,2]
  self.assertIsNone(coxeter_frontier_bound(GraphDefinition.from_dict(v)))
  self.assertIsNone(coxeter_frontier_bound(GraphDefinition.permutation([[1,2,3,0]],[0,1,2,3])))

class MatrixBounds(unittest.TestCase):
 def test_small_modular_orbit(self):
  from multigpubfs.workload_geometry import certified_frontier_bound
  g=GraphDefinition.matrix(2,2,[([1,1,0,1],3),([1,0,1,1],3)],[1,0,0,1])
  b=certified_frontier_bound(g);self.assertEqual(b['order_upper_bound'],81)
  self.assertLessEqual(max(map(len,g.exact_layers(81))),b['maximum_frontier'])
 def test_noncanonical_unbounded_and_mixed_rejected(self):
  from multigpubfs.workload_geometry import certified_frontier_bound
  for generators,start in [([([1,1,0,1],0)],[1,0,0,1]),([([1,1,0,1],3)],[3,0,0,1]),([([1,1,0,1],3),([1,0,1,1],2)],[1,0,0,1])]:
   self.assertIsNone(certified_frontier_bound(GraphDefinition.matrix(2,2,generators,start)))
