import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('typed_gate', ROOT / 'kaggle/lsa-bfs-gate/kernel.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class TypedRankGateTests(unittest.TestCase):
    def test_reuse_fixture_has_more_batches_than_four_banks_on_one_rank(self):
        base = json.loads((ROOT / 'tests/run-s4-two-rank.json').read_text())
        saved = copy.deepcopy(base)
        cases = gate.typed_reuse_configs(base)
        self.assertEqual(base, saved)
        self.assertEqual(len(cases), 6)
        self.assertEqual({(c['frontier_profile'], c['capacities']['route_slot_count']) for c in cases},
            {(p, b) for p in ('DENSE', 'HASH_FIRST') for b in (2, 3, 4)})
        for config in cases:
            self.assertEqual(config['graph']['expected_max_unique_states'], 64)
            self.assertEqual(config['graph']['modulus'], 2)
            self.assertEqual(config['parent_batch'], 1)
            self.assertEqual(config['capacities']['route_slot_records'], 6)
            # Elementary row additions, each its own inverse over F_2.
            generators = config['graph']['generators']
            self.assertEqual([i for i, x in enumerate(generators[0]) if x], [0, 1, 5, 10, 15])
            self.assertEqual([i for i, x in enumerate(generators[1]) if x], [0, 5, 6, 10, 15])
            self.assertEqual([i for i, x in enumerate(generators[2]) if x], [0, 5, 10, 11, 15])
            self.assertEqual(generators[:3], generators[3:])
            self.assertEqual(config['graph']['inverse_map'], [3, 4, 5, 0, 1, 2])

    def test_gate_covers_profiles_credits_prededup_and_rank_maps_without_mutating_fixture(self):
        base = json.loads((ROOT / 'tests/run-s4-two-rank.json').read_text())
        saved = copy.deepcopy(base)
        cases = gate.typed_rank_configs(base)
        self.assertEqual(base, saved)
        self.assertEqual(len(cases), 48)
        self.assertEqual({(c['frontier_profile'], c['completion_epoch_window'],
            c['local_pre_dedup'], tuple(c['topology']['logical_owner_to_rank']),
            c['capacities']['route_slot_count']) for c in cases},
            {(p, k, d, m, b) for p in ('DENSE', 'HASH_FIRST') for k in (2, 3)
             for d in (False, True) for m in ((0, 1), (1, 0)) for b in (2, 3, 4)})
        for c in cases:
            self.assertEqual(c['owner_backend'], 'CUCO_RANK')
            self.assertEqual(c['library_pool_bytes'], 100663296)
            self.assertIn(c['capacities']['route_slot_count'], (2, 3, 4))
            self.assertEqual(c['graph'], saved['graph'])


if __name__ == '__main__':
    unittest.main()
