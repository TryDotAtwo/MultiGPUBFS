import json,unittest
from types import SimpleNamespace
from multigpubfs.graph_definition import GraphDefinition,from_cayleypy

class GraphDefinitionTests(unittest.TestCase):
 def test_directed_and_repeated_labels(self):
  g=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,0,1,2])
  self.assertFalse(g.inverse_closed());self.assertEqual(g.successor([0,0,1,2],0),[0,1,2,0])
  layers=g.exact_layers(100);self.assertEqual(sum(map(len,layers)),12)
 def test_rectangular_matrix(self):
  g=GraphDefinition.matrix(2,1,[([1,1,0,1],257)],[256,2])
  self.assertEqual(g.successor([256,2],0),[1,2]);self.assertFalse(g.inverse_closed())
 def test_wrapping_before_modulus(self):
  g=GraphDefinition.matrix(1,1,[([2],7)],[(1<<63)-1])
  self.assertEqual(g.successor(g.start,0),[5])
  h=GraphDefinition.matrix(1,1,[([2],0)],[(1<<63)-1]);self.assertEqual(h.successor(h.start,0),[-2])
 def test_mixed_moduli(self):
  g=GraphDefinition.matrix(1,1,[([2],5),([3],7)],[4])
  self.assertEqual(g.successor([4],0),[3]);self.assertEqual(g.successor([4],1),[5])
 def test_shape_range_and_schema_validation(self):
  for gs,start in [([[0,0]],[0,1]),([[0,1]],[0]),([[0,1]],[0,1<<63])]:
   with self.assertRaises(ValueError):GraphDefinition.permutation(gs,start)
  with self.assertRaises(ValueError):GraphDefinition.matrix(2,1,[([1],3)],[1,2])
  with self.assertRaises(ValueError):GraphDefinition.matrix(1,1,[([1],1)],[1])
 def test_identity_name_independent_order_sensitive(self):
  g=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2],name='a')
  h=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2],name='b')
  self.assertEqual(g.digest(),h.digest());self.assertTrue(g.inverse_closed())
  self.assertNotEqual(g.digest(),GraphDefinition.permutation(list(reversed(g.action['generators'])),g.start).digest())
  self.assertEqual(GraphDefinition.from_dict(json.loads(g.to_json())).digest(),g.digest())
 def test_cayleypy_adapter(self):
  x=SimpleNamespace(generators_type=SimpleNamespace(name='PERMUTATION'),generators_permutations=[[1,0]],central_state=[0,0],name='coset',generator_names=['x'])
  g=from_cayleypy(x);self.assertEqual(g.start,[0,0]);self.assertEqual(g.generator_names,['x'])
  x=SimpleNamespace(generators_type=SimpleNamespace(name='MATRIX'),generators_matrices=[SimpleNamespace(matrix=[[1,1],[0,1]],modulo=257)],central_state=[1,2],name='rect',generator_names=['t'])
  self.assertEqual(from_cayleypy(x).successor([1,2],0),[3,2])
 def test_bounded_oracle_stops_without_claiming_complete(self):
  g=GraphDefinition.permutation([[1,2,0]],[0,1,2])
  with self.assertRaises(ValueError):g.exact_layers(2)
if __name__=='__main__':unittest.main()
