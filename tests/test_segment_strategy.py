"""Public strategy selection, validation and capability safety."""
import os
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.graph_definition import GraphDefinition
from multigpubfs.launch import run_graph
from multigpubfs.autotune import _configuration_digest

GRAPH=GraphDefinition.permutation([[1,0]],[0,1])
class SegmentStrategyContract(unittest.TestCase):
    def test_default_is_atomic(self):
        self.assertEqual(NativeOptions().segment_strategy,'atomic')

    def test_all_three_are_explicit_options(self):
        for value in ('atomic','warp','batch'):
            self.assertEqual(NativeOptions(segment_strategy=value).segment_strategy,value)

    def test_unknown_value_rejected_before_native_work(self):
        with patch('multigpubfs.launch.native_runtime') as native:
            with self.assertRaisesRegex(ValueError,'INVALID_SEGMENT_STRATEGY'):
                run_graph(GRAPH,'/tmp/mgbfs-strategy-unused',memory_backend='segmented24',segment_strategy='bad')
            native.assert_not_called()

    def test_legacy_rejects_nonatomic_before_native_work(self):
        with patch('multigpubfs.launch.native_runtime') as native:
            with self.assertRaisesRegex(ValueError,'SEGMENT_STRATEGY_REQUIRES_SEGMENTED_BACKEND'):
                run_graph(GRAPH,'/tmp/mgbfs-strategy-unused',memory_backend='legacy',segment_strategy='warp')
            native.assert_not_called()

    def test_options_inherit_environment_when_not_explicit(self):
        with patch.dict(os.environ, {'MGBFS_SEGMENT_STRATEGY':'warp'}):
            self.assertEqual(NativeOptions().segment_strategy,'warp')
            self.assertEqual(NativeOptions(segment_strategy='atomic').segment_strategy,'atomic')

    def test_scoped_native_environment_checked_before_native(self):
        with patch('multigpubfs.launch.native_runtime') as native:
            with self.assertRaisesRegex(ValueError,'INVALID_SEGMENT_STRATEGY'):
                run_graph(GRAPH,'/tmp/mgbfs-strategy-unused',memory_backend='segmented24',_native_env={'MGBFS_SEGMENT_STRATEGY':'wrong'})
            native.assert_not_called()

    def test_old_native_rejects_new_strategies_before_calculation(self):
        with patch('multigpubfs.host_memory.admit',return_value=None), patch('multigpubfs.launch.native_runtime',return_value=('/fake/mgbfs',{})), patch('multigpubfs.launch._require_memory_backend',return_value={'segmented32_v2':True}):
            for value in ('warp','batch'):
                with self.assertRaisesRegex(RuntimeError,'NATIVE_SEGMENT_STRATEGY_UNAVAILABLE'):
                    run_graph(GRAPH,'/tmp/mgbfs-strategy-unused',memory_backend='segmented24',segment_strategy=value)

    def test_capability_cache_tracks_replaced_cuda_library(self):
        from multigpubfs.launch import _memory_backend_capabilities, _require_memory_backend
        _memory_backend_capabilities.clear()
        with tempfile.NamedTemporaryFile() as binary:
            versions = iter([
                {'cuda_library_sha256':'old','resolved_dependencies':[['libmgbfs_cuda.so','/tmp/a',1,1]]},
                {'cuda_library_sha256':'old','resolved_dependencies':[['libmgbfs_cuda.so','/tmp/a',1,1]]},
                {'cuda_library_sha256':'new','resolved_dependencies':[['libmgbfs_cuda.so','/tmp/a',2,2]]},
                None, None])
            def query(*args, **kwargs):
                value = query.calls == 0
                query.calls += 1
                return SimpleNamespace(returncode=0,stdout=json.dumps(
                    {'schema':1,'segmented24_v1':True,'segmented32_v2':True,
                     'segmented_memory_presets_v1':True,'segmented_strategies_v1':{'batch':value}}))
            query.calls=0
            with patch('multigpubfs.autotune._dependency_identity',side_effect=lambda *a:next(versions)), \
                 patch('multigpubfs.launch.native_query',side_effect=query):
                env={'LD_LIBRARY_PATH':'/tmp'}
                self.assertTrue(_require_memory_backend(binary.name,env,'segmented24')['segmented_strategies_v1']['batch'])
                self.assertTrue(_require_memory_backend(binary.name,env,'segmented24')['segmented_strategies_v1']['batch'])
                self.assertFalse(_require_memory_backend(binary.name,env,'segmented24')['segmented_strategies_v1']['batch'])
                _require_memory_backend(binary.name,env,'segmented24')
                _require_memory_backend(binary.name,env,'segmented24')
                self.assertEqual(query.calls,4)
        _memory_backend_capabilities.clear()

    def test_strategy_scopes_tuning_and_preflight_keys(self):
        a=_configuration_digest({'MGBFS_SEGMENT_STRATEGY':'atomic'})
        b=_configuration_digest({'MGBFS_SEGMENT_STRATEGY':'warp'})
        c=_configuration_digest({'MGBFS_SEGMENT_STRATEGY':'batch'})
        self.assertEqual(len({a,b,c}),3)

    def test_options_reject_invalid_values(self):
        for value in ('invalid','',None,1):
            with self.assertRaisesRegex(ValueError,'INVALID_SEGMENT_STRATEGY'):
                NativeOptions(segment_strategy=value)

if __name__=='__main__':unittest.main()
