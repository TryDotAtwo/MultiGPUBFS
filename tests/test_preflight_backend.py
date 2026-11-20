"""Preflight must test and cache the selected backend and graph."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition

class PreflightBackendTests(unittest.TestCase):
 def test_backend_selection(self):
  from multigpubfs.nccl_preflight import _requested_memory_backend
  self.assertEqual(_requested_memory_backend({}), 'legacy')
  self.assertEqual(_requested_memory_backend({'MGBFS_EXPERIMENTAL_SEGMENTED24':'1'}), 'segmented24')
  self.assertEqual(_requested_memory_backend({'MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_EXPERIMENTAL_BITMAP':'1'}), 'permutation_bitmap')
 def test_key_distinguishes_graph_and_backend(self):
  from multigpubfs.nccl_preflight import _preflight_cache_key
  class Graph:
   def __init__(self,digest): self.value=digest
   def digest(self): return self.value
  with patch('multigpubfs.nccl_preflight._identity',return_value='same'):
   keys=[_preflight_cache_key('native',[0,1],env,graph) for env,graph in [({},Graph('a')),({'MGBFS_EXPERIMENTAL_SEGMENTED24':'1'},Graph('a')),({'MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_EXPERIMENTAL_BITMAP':'1'},Graph('a')),({},Graph('b')),({},None)]]
  self.assertEqual(len(set(keys)),len(keys))
 def test_key_distinguishes_segment_configuration(self):
  from multigpubfs.nccl_preflight import _preflight_cache_key
  class Graph:
   def digest(self):return 'same-graph'
  base={'MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_EXPERIMENTAL_BITMAP':'1'}
  with patch('multigpubfs.nccl_preflight._identity',return_value='same'):
   keys=[_preflight_cache_key('native',[0,1],dict(base,**extra),Graph()) for extra in ({},{'MGBFS_SEGMENT_EXACT_EXCHANGE':'1'},{'MGBFS_SEGMENT_MAX_ROWS':'4096'},{'MGBFS_SEGMENT_OWNER_LANES':'8'})]
  self.assertEqual(len(set(keys)),len(keys))
 def test_prefix_passes_backend(self):
  from multigpubfs.nccl_preflight import _run_graph_prefix
  graph=GraphDefinition.permutation([[1,0,2]],[0,1,2])
  for backend in ('legacy','segmented24','permutation_bitmap'):
   with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
    root=Path(directory); (root/'graph.json').write_text(graph.to_json()); (root/'request.json').write_text(json.dumps({'devices':[0,1], 'memory_backend':backend}))
    with patch('multigpubfs.launch.run_graph',return_value={'status':'COMPLETE'}) as launch:
     _run_graph_prefix(root)
     self.assertEqual(launch.call_args.kwargs['memory_backend'],backend)
     self.assertEqual(launch.call_args.kwargs['devices'],[0,1])
     self.assertEqual(launch.call_args.kwargs['_profile_layers'],4)

class TransportReuseTests(unittest.TestCase):
 def test_new_graph_rehearses_backend_without_repeating_transport_probe(self):
  from types import SimpleNamespace
  from unittest.mock import patch
  from multigpubfs.nccl_preflight import prepare
  graph=GraphDefinition.permutation([[1,0,2]],[0,1,2])
  old={'status':'VERIFIED','selected':'inherited','policy':{},'ranks':[{'rank':0}],
       'rehearsal_graph':'old-graph','rehearsal_backend':'permutation_bitmap',
       'bfs_prefix':{'status':'COMPLETE','layer_sizes':[1,1]},'seconds':3.0,'evidence':'old-evidence'}
  active=SimpleNamespace(communication_profiles={('same','permutation_bitmap','old-graph'):({'MGBFS_GENERIC_VRAM_RESERVE_BYTES':'42'},old)})
  env={'MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_EXPERIMENTAL_BITMAP':'1'}
  with tempfile.TemporaryDirectory() as directory,patch('multigpubfs.generic_session._active',active),patch('multigpubfs.nccl_preflight._identity',return_value='same'),patch('multigpubfs.nccl_preflight._cached_prefix',return_value={'status':'COMPLETE','layer_sizes':[1,1]}) as prefix,patch('multigpubfs.nccl_preflight.alternatives',side_effect=AssertionError('transport was repeated')):
   patched,receipt=prepare('native',[0,1],env,Path(directory)/'out',graph)
   self.assertEqual(patched['MGBFS_GENERIC_VRAM_RESERVE_BYTES'],'42')
   self.assertTrue(receipt['cache_hit']);self.assertTrue(receipt['transport_cache_hit'])
   self.assertEqual(receipt['rehearsal_graph'],graph.digest())
   self.assertEqual(receipt['bfs_prefix']['layer_sizes'],[1,1])
   self.assertEqual(prefix.call_count,1)
 def test_new_segment_mode_rehearses_but_reuses_transport(self):
  from types import SimpleNamespace
  from multigpubfs.nccl_preflight import prepare, _preflight_cache_key
  graph=GraphDefinition.permutation([[1,0,2]],[0,1,2])
  env={'MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_EXPERIMENTAL_BITMAP':'1'}
  newer=dict(env,MGBFS_SEGMENT_EXACT_EXCHANGE='1')
  old={'status':'VERIFIED','selected':'inherited','policy':{},'ranks':[{'rank':0}],
       'rehearsal_graph':graph.digest(),'rehearsal_backend':'permutation_bitmap',
       'bfs_prefix':{'status':'COMPLETE','layer_sizes':[1,1]},'seconds':3.0,'evidence':'old'}
  with patch('multigpubfs.nccl_preflight._identity',return_value='same'):
   active=SimpleNamespace(communication_profiles={_preflight_cache_key('native',[0,1],env,graph):({},old)})
   with tempfile.TemporaryDirectory() as directory,patch('multigpubfs.generic_session._active',active),patch('multigpubfs.nccl_preflight._cached_prefix',return_value={'status':'COMPLETE','layer_sizes':[1,1]}) as prefix,patch('multigpubfs.nccl_preflight.alternatives',side_effect=AssertionError('transport was repeated')):
    _,receipt=prepare('native',[0,1],newer,Path(directory)/'out',graph)
    self.assertTrue(receipt['transport_cache_hit'])
    self.assertEqual(prefix.call_count,1)
    self.assertEqual(prefix.call_args.args[3]['MGBFS_SEGMENT_EXACT_EXCHANGE'],'1')
 def test_failed_cached_prefix_does_not_get_accepted(self):
  from types import SimpleNamespace
  from unittest.mock import patch
  from multigpubfs.nccl_preflight import prepare
  graph=GraphDefinition.permutation([[1,0,2]],[0,1,2]);env={}
  old={'status':'VERIFIED','selected':'inherited','policy':{},'ranks':[],
       'rehearsal_graph':'old','rehearsal_backend':'legacy','seconds':1,'evidence':'old'}
  active=SimpleNamespace(communication_profiles={('same','legacy','old'):({},old)})
  with tempfile.TemporaryDirectory() as directory,patch('multigpubfs.generic_session._active',active),patch('multigpubfs.nccl_preflight._identity',return_value='same'),patch('multigpubfs.nccl_preflight._cached_prefix',side_effect=RuntimeError('BROKEN_PREFIX')),patch('multigpubfs.nccl_preflight.alternatives',return_value=[]):
   with self.assertRaisesRegex(RuntimeError,'NO_VERIFIED_TRANSPORT'):
    prepare('native',[0,1],env,Path(directory)/'out',graph)
   result=json.loads((Path(directory)/'out-communication-evidence/report.json').read_text())
   self.assertEqual(result['attempts'][0]['error'],'BROKEN_PREFIX')

if __name__=='__main__':unittest.main()
