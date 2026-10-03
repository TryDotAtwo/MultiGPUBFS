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
