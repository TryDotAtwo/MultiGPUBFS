"""Integration with the actual optional CayleyPy package, CPU semantic oracle."""
import unittest
from cayleypy import CayleyGraph, CayleyGraphDef, MatrixGenerator, PermutationGroups
from multigpubfs.graph_definition import from_cayleypy
import numpy as np
class RealCayleyAdapter(unittest.TestCase):
 def test_catalog_and_runtime_wrapper(self):
  fixtures=[PermutationGroups.lrx(5),PermutationGroups.lx(5),PermutationGroups.all_transpositions(4),PermutationGroups.pancake(5),PermutationGroups.signed_reversals(3)]
  fixtures.append(CayleyGraphDef.for_matrix_group(generators=[MatrixGenerator.create([[1,1],[0,1]],7)],central_state=[[0,2],[1,3]]))
  for definition in fixtures:
   with self.subTest(name=definition.name,kind=definition.generators_type.name):
    graph=CayleyGraph(definition,device='cpu',bit_encoding_width=None)
    adapter=from_cayleypy(graph)
    self.assertEqual(adapter.digest(),from_cayleypy(definition).digest())
    frontier=[adapter.start]
    for depth in range(3):
     following=[]
     for state in frontier[:10]:
      actual=graph.get_neighbors_decoded(np.array([state])).tolist()
      for move,expected in enumerate(actual):
       expected=np.array(expected).reshape(-1).tolist()
       self.assertEqual(adapter.successor(state,move),expected)
       following.append(expected)
     frontier=following
if __name__=='__main__':unittest.main()
