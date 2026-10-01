import unittest
from replay_lsa_cancel_candidate import configure_owner_environment


class OwnerEnvironmentTests(unittest.TestCase):
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
