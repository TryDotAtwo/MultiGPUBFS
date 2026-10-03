import unittest
from replay_lsa_cancel_candidate import configure_owner_environment, configure_epoch_window, rank_arguments
from pathlib import Path


class OwnerEnvironmentTests(unittest.TestCase):
    def test_typed_epoch_window_is_authoritative_not_inherited(self):
        from replay_lsa_cancel_candidate import configure_run_epoch_window
        env = {'MGBFS_EPOCH_WINDOW': 'invalid-inherited-value'}
        self.assertEqual(configure_run_epoch_window(env, {'completion_epoch_window': 3}), 3)
        self.assertEqual(env['MGBFS_EPOCH_WINDOW'], '3')
        self.assertEqual(configure_run_epoch_window(env, {}), 2)
        self.assertEqual(configure_run_epoch_window(env, {'completion_epoch_window': 4}, 4), 4)
        with self.assertRaises(ValueError):
            configure_run_epoch_window(env, {'completion_epoch_window': 3}, 4)
        self.assertEqual(env['MGBFS_EPOCH_WINDOW'], '4')
    def test_typed_cuco_uses_snapshot_pool_not_benchmark_or_inherited_pool(self):
        env = {'MGBFS_LIBRARY_POOL_BYTES': '123', 'sentinel': 'keep'}
        configure_owner_environment(env, 'CUCO_RANK', '1,0',
            {'owner_backend': 'CUCO_RANK', 'library_pool_bytes': 100663296})
        self.assertEqual(env['MGBFS_LIBRARY_POOL_BYTES'], '100663296')
        self.assertEqual(env['sentinel'], 'keep')

    def test_typed_pool_errors_do_not_mutate_environment(self):
        for pool in (None, 0, 257, True, '67108864'):
            env = {'sentinel': 'keep'}
            with self.assertRaises(ValueError):
                configure_owner_environment(env, 'CUCO_RANK', '0,1',
                    {'owner_backend': 'CUCO_RANK', 'library_pool_bytes': pool})
            self.assertEqual(env, {'sentinel': 'keep'})
        env = {'sentinel': 'keep'}
        with self.assertRaises(ValueError):
            configure_owner_environment(env, 'CUB_SORT_MERGE', '0,1',
                {'owner_backend': 'CUB_SORT_MERGE', 'library_pool_bytes': 67108864})
        self.assertEqual(env, {'sentinel': 'keep'})
    def test_typed_replay_calls_run_not_bench(self):
        case = Path('case')
        config = Path('snapshot.json')
        self.assertEqual(rank_arguments(case, 's4', 16, config),
            ['run', str(config), str(case / 'bootstrap'), str(case / 'archive'), str(case / 'result')])
        self.assertEqual(rank_arguments(case, 's4', 16)[:4], ['bench', '--reference', 's4', '16'])

    def test_epoch_window_explicit_and_inherited(self):
        for requested, inherited, expected in ((None, None, 2), (None, '3', 3), (4, '3', 4)):
            env = {} if inherited is None else {'MGBFS_EPOCH_WINDOW': inherited}
            self.assertEqual(configure_epoch_window(env, requested), expected)
            self.assertEqual(env['MGBFS_EPOCH_WINDOW'], str(expected))

    def test_epoch_window_rejects_without_mutation(self):
        for value in ('bad', '1', '4294967296', '-2'):
            env = {'MGBFS_EPOCH_WINDOW': value}
            with self.assertRaises(ValueError):
                configure_epoch_window(env, None)
            self.assertEqual(env, {'MGBFS_EPOCH_WINDOW': value})

    def test_native_drops_inherited_pool_and_preserves_other_environment(self):
        for backend in ('CUB_SORT_MERGE', 'BMMA_BUCKET'):
            env = {'MGBFS_LIBRARY_POOL_BYTES': '123', 'CUDA_VISIBLE_DEVICES': '0,1'}
            configure_owner_environment(env, backend, '1,0')
            self.assertEqual(env, {'CUDA_VISIBLE_DEVICES': '0,1',
                'MGBFS_OWNER_BACKEND': backend, 'MGBFS_RANK_MAP': '1,0'})

    def test_cuco_retains_explicit_fixed_pool(self):
        env = {}
        configure_owner_environment(env, 'CUCO_RANK', '0,1')
        self.assertEqual(env['MGBFS_LIBRARY_POOL_BYTES'], '67108864')
        self.assertEqual(env['MGBFS_RANK_MAP'], '0,1')

    def test_invalid_selection_does_not_mutate_environment(self):
        for backend, mapping in [('UNKNOWN', '0,1'), ('CUCO_RANK', '0,0'),
                                 ('BMMA_BUCKET', '2,0')]:
            env = {'sentinel': 'keep'}
            with self.assertRaises(ValueError):
                configure_owner_environment(env, backend, mapping)
            self.assertEqual(env, {'sentinel': 'keep'})
