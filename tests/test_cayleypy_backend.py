"""CayleyPy dispatch and compact-output semantics; no hardware claims."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cayleypy import CayleyGraph, PermutationGroups
import multigpubfs.cayleypy_backend as api


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.graph = CayleyGraph(PermutationGroups.lrx(3), device='cpu')
        self.original = CayleyGraph.bfs
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        api.disable_native()
        self.temp.cleanup()

    def native(self, definition, output, **kwargs):
        self.definition, self.options = definition, kwargs
        output.mkdir(parents=True)
        (output/'states.json').write_text(json.dumps(dict(current=[[2,1,0]], previous_small=[[0,2,1]])))
        return dict(status='COMPLETE', layer_sizes=[1,2,3])

    def test_simple_call_and_compact_samples(self):
        api.enable_native(api.NativeOptions(output_root=self.root))
        with patch.object(api, 'run_graph', side_effect=self.native):
            result = self.graph.bfs()
        self.assertTrue(result.bfs_completed)
        self.assertEqual(result.layer_sizes, [1,2,3])
        self.assertEqual(result.diameter(), 2)
        self.assertEqual(result.num_vertices(), 6)
        self.assertEqual(set(result.layer_samples), {1,2})
        self.assertEqual(result.layers, {})
        self.assertIsNone(self.options['devices'])
        self.assertTrue(self.options['autotune'])
        with self.assertRaises(KeyError): result.last_layer()

    def test_disable_and_reenable(self):
        api.enable_native(); api.enable_native(); api.disable_native()
        self.assertIs(CayleyGraph.bfs, self.original)

    def test_one_shot_does_not_patch(self):
        with patch.object(api, 'run_graph', side_effect=self.native):
            api.bfs(self.graph, output=self.root/'one')
        self.assertIs(CayleyGraph.bfs, self.original)

    def test_torch_preserves_kwargs(self):
        received = {}
        def original(graph, **kwargs): received.update(kwargs); return 'torch'
        with patch.object(CayleyGraph, 'bfs', original):
            api.enable_native()
            self.assertEqual(self.graph.bfs(backend='torch', max_diameter=2), 'torch')
            api.disable_native()
        self.assertEqual(received, dict(max_diameter=2))

    def test_unsupported_before_launch(self):
        api.enable_native()
        for kwargs in (dict(return_all_edges=True), dict(max_diameter=2),
                       dict(max_layer_size_to_store=None), dict(misspelled=True)):
            with patch.object(api, 'run_graph') as run:
                with self.assertRaises(NotImplementedError): self.graph.bfs(**kwargs)
                run.assert_not_called()

    def test_single_decoded_start(self):
        with patch.object(api, 'run_graph', side_effect=self.native):
            api.bfs(self.graph, output=self.root/'start', start_states=[[2,1,0]])
        self.assertEqual(self.definition.start, [2,1,0])

    def test_multisource_rejected(self):
        with self.assertRaises(NotImplementedError):
            api.bfs(self.graph, start_states=[[0,1,2],[2,1,0]])

    def test_incomplete_not_diameter(self):
        def incomplete(*args, **kwargs):
            report = self.native(*args, **kwargs); report['status']='INCOMPLETE'; return report
        with patch.object(api, 'run_graph', side_effect=incomplete):
            result = api.bfs(self.graph, output=self.root/'partial')
        self.assertFalse(result.bfs_completed)
        with self.assertRaises(RuntimeError): result.diameter()

    def test_runtime_failure_no_cpu_restart(self):
        with patch.object(api, 'run_graph', side_effect=RuntimeError('GPU failed')):
            with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
                api.bfs(self.graph, backend='auto', output=self.root/'failed')

    def test_auto_falls_back_only_for_unsupported_options(self):
        def original(graph, **kwargs): return kwargs
        with patch.object(CayleyGraph, 'bfs', original):
            api.enable_native(default_backend='auto')
            with patch.object(api, 'run_graph') as run:
                self.assertEqual(self.graph.bfs(return_all_edges=True), dict(return_all_edges=True))
                run.assert_not_called()
            api.disable_native()

    def test_full_retained_layer_is_available(self):
        def full(*args, **kwargs):
            report = self.native(*args, **kwargs); report['layer_sizes']=[1,1,1]; return report
        with patch.object(api, 'run_graph', side_effect=full):
            result=api.bfs(self.graph, output=self.root/'full')
        self.assertEqual(result.last_layer().tolist(), [[2,1,0]])

    def test_explicit_devices_and_budget_forwarded(self):
        with patch.object(api, 'run_graph', side_effect=self.native):
            api.bfs(self.graph, output=self.root/'devices',
                    native_options=api.NativeOptions(devices=(1,0),max_seconds=12,autotune=False))
        self.assertEqual(self.options, dict(devices=(1,0),max_seconds=12,autotune=False))

    def test_collision_with_other_integration(self):
        api.enable_native()
        installed = CayleyGraph.bfs
        CayleyGraph.bfs = lambda graph: None
        try:
            with self.assertRaises(RuntimeError): api.disable_native()
        finally: CayleyGraph.bfs = installed

if __name__ == '__main__': unittest.main()
