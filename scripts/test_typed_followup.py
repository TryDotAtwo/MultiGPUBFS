"""Check selection of full-runtime follow-up cases, not GPU correctness."""
import importlib.util
import json
from pathlib import Path
import unittest

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('typed_gate', root / 'kaggle/lsa-bfs-gate/kernel.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FollowupTests(unittest.TestCase):
    def test_initcheck_repeats_every_profile_and_bank_without_filters(self):
        base = json.loads((root / 'tests/run-s4-two-rank.json').read_text())
        before = json.dumps(base, sort_keys=True)
        cases = gate.typed_followup_cases(base)
        checks = [case for case in cases if case['tool'] == 'initcheck']
        self.assertEqual(len(checks), 18)
        self.assertEqual(len({case['label'] for case in cases}), 20)
        observed = {(case['config']['frontier_profile'],
                     case['config']['capacities']['route_slot_count'], case['repeat'])
                    for case in checks}
        self.assertEqual(observed, {(p, b, r) for p in ('DENSE', 'HASH_FIRST')
                                   for b in (2, 3, 4) for r in range(3)})
        for case in checks:
            self.assertEqual(case['extra'], ['--healthy-only', '--instrument-processes', 'initcheck'])
            self.assertEqual(case['config']['completion_epoch_window'], 3)
        self.assertEqual(json.dumps(base, sort_keys=True), before)

    def test_timelines_use_full_reuse_graph_in_both_profiles(self):
        base = json.loads((root / 'tests/run-s4-two-rank.json').read_text())
        cases = [case for case in gate.typed_followup_cases(base) if case['tool'] == 'nsys']
        self.assertEqual([case['config']['frontier_profile'] for case in cases], ['DENSE', 'HASH_FIRST'])
        for case in cases:
            self.assertEqual(case['config']['graph']['expected_max_unique_states'], 64)
            self.assertEqual(case['config']['capacities']['route_slot_records'], 6)
            self.assertEqual(case['config']['capacities']['route_slot_count'], 3)
            self.assertEqual(case['extra'], ['--healthy-only', '--unitriangular-modulus', '2',
                '--require-bank-reuse', '--instrument-processes', 'nsys'])


if __name__ == '__main__':
    unittest.main()
