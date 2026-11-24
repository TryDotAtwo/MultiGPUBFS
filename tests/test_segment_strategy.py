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
            for value, error in (('warp','SEGMENT_WARP_SAFE_NATIVE_ABI_MISSING'),('batch','NATIVE_SEGMENT_STRATEGY_UNAVAILABLE')):
                with self.subTest(value=value), self.assertRaisesRegex(RuntimeError,error):
                    run_graph(GRAPH,'/tmp/mgbfs-strategy-unused',memory_backend='segmented24',segment_strategy=value)

    def test_implicit_segmented_run_passes_bounded_strategy_to_preflight(self):
        capabilities = {
            'segmented32_v2': True,
            'segmented_strategies_v1': {'batch': True, 'warp': True},
            'segmented_hash_series_v1': True,
            'segmented_warp_hash_safe_v1': True,
        }
        with tempfile.TemporaryDirectory() as temp, patch(
            'multigpubfs.host_memory.admit', return_value=None,
        ), patch('multigpubfs.launch.native_runtime', return_value=('/fake/mgbfs', {})), patch(
            'multigpubfs.launch._require_memory_backend', return_value=capabilities,
        ), patch('multigpubfs.nccl_preflight.prepare', side_effect=RuntimeError('STOP_AT_PREFLIGHT')) as preflight:
            for scoped_env, expected in (({}, 'batch'), ({'MGBFS_SEGMENT_STRATEGY': 'warp'}, 'warp')):
                with self.subTest(scoped_env=scoped_env), self.assertRaisesRegex(RuntimeError, 'STOP_AT_PREFLIGHT'):
                    run_graph(GRAPH, os.path.join(temp, 'result'), devices=[0], memory_backend='segmented24',
                              _native_env=scoped_env)
                self.assertEqual(preflight.call_args.args[2]['MGBFS_SEGMENT_STRATEGY'], expected)

    def test_hash_warp_requires_safe_abi_even_when_strategy_is_advertised(self):
        capabilities={'segmented32_v2':True,'segmented_strategies_v1':{'warp':True},'segmented_warp_hash_safe_v1':False}
        with patch('multigpubfs.host_memory.admit',return_value=None), patch('multigpubfs.launch.native_runtime',return_value=('/fake/mgbfs',{})), patch('multigpubfs.launch._require_memory_backend',return_value=capabilities):
            with self.assertRaisesRegex(RuntimeError,'SEGMENT_WARP_SAFE_NATIVE_ABI_MISSING'):
                run_graph(GRAPH,'/tmp/mgbfs-strategy-unused',memory_backend='segmented24',segment_strategy='warp')

    def test_hash_warp_requires_both_safe_abi_and_strategy_capability(self):
        capabilities={'segmented32_v2':True,'segmented_strategies_v1':{},'segmented_warp_hash_safe_v1':True}
        with patch('multigpubfs.host_memory.admit',return_value=None), patch('multigpubfs.launch.native_runtime',return_value=('/fake/mgbfs',{})), patch('multigpubfs.launch._require_memory_backend',return_value=capabilities):
            with self.assertRaisesRegex(RuntimeError,'NATIVE_SEGMENT_STRATEGY_UNAVAILABLE'):
                run_graph(GRAPH,'/tmp/mgbfs-strategy-unused',memory_backend='segmented24',segment_strategy='warp')

    def test_bitmap_warp_still_requires_strategy_capability(self):
        capabilities={'segmented32_v2':True}
        with patch('multigpubfs.host_memory.admit',return_value=None), patch('multigpubfs.launch.native_runtime',return_value=('/fake/mgbfs',{})), patch('multigpubfs.launch._require_memory_backend',return_value=capabilities):
            with self.assertRaisesRegex(RuntimeError,'NATIVE_SEGMENT_STRATEGY_UNAVAILABLE'):
                run_graph(GRAPH,'/tmp/mgbfs-strategy-unused',memory_backend='permutation_bitmap',segment_strategy='warp')

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
