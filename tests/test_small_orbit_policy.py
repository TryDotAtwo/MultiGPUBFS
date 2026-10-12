from multigpubfs.hardware_profiles import small_permutation_workload
from multigpubfs.graph_definition import GraphDefinition
def graph(n):return GraphDefinition.permutation([list(range(n))],list(range(n)))
def test_small_entire_orbit():
 assert small_permutation_workload(graph(8),[{'sm_count':28},{'sm_count':28}])
 assert not small_permutation_workload(graph(9),[{'sm_count':28},{'sm_count':28}])
def test_capacity_hint_cannot_hide_large_orbit():
 g=GraphDefinition.permutation([list(range(14))],list(range(14)),expected_max_unique_states=100)
 assert not small_permutation_workload(g,[{'sm_count':132}])
def test_repeated_symbols():
 g=GraphDefinition.permutation([list(range(14))],[0]*13+[1])
 assert small_permutation_workload(g,[{'sm_count':28}])

def test_api_owns_and_releases_pool(tmp_path):
 import json
 from unittest.mock import patch
 from multigpubfs import generic_session
 from multigpubfs import cayleypy_backend as api
 import cayleypy
 g=cayleypy.CayleyGraph(cayleypy.PermutationGroups.lrx(3),device='cpu')
 def run(d,output,**kwargs):
  assert generic_session._active is not None
  output.mkdir();(output/'states.json').write_text(json.dumps({'current':[],'previous_small':[]}))
  return {'status':'COMPLETE','layer_sizes':[1]}
 with patch.object(api,'run_graph',side_effect=run):api.bfs(g,output=tmp_path/'one')
 assert generic_session._active is None
 with generic_session.persistent_graph_workers():
  pool=generic_session._active
  with patch.object(api,'run_graph',side_effect=run):api.bfs(g,output=tmp_path/'two')
  assert generic_session._active is pool
 assert generic_session._active is None
