"""Check selection of full-runtime follow-up cases, not GPU correctness."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('typed_gate', root / 'kaggle/lsa-bfs-gate/kernel.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FollowupTests(unittest.TestCase):
    def test_every_typed_execution_installs_archive_reader_dependency(self):
        for mode in ('typed_rank_gate', 'typed_followup_gate', 'typed_stress_gate'):
            self.assertIn('pyarrow==19.0.1', gate.oracle_dependency_packages(mode), mode)

    def test_stress_configs_pass_real_offline_cli_admission(self):
        binary = root / 'target/debug' / ('mgbfs.exe' if os.name == 'nt' else 'mgbfs')
        if not binary.exists():
            self.skipTest('build mgbfs-cli before real offline admission check')
        base = json.loads((root / 'tests/run-s4-two-rank.json').read_text())
        with tempfile.TemporaryDirectory(prefix='mgbfs-stress-admission-') as directory:
            path = Path(directory) / 'config.json'
            for config in gate.typed_stress_configs(base, 3):
                path.write_text(json.dumps(config), encoding='utf-8')
                result = subprocess.run([str(binary), 'preflight', '--offline', str(path)],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report['status'], 'CONFIG_VALIDATED')
                self.assertFalse(report['hardware_ready'])

    def test_stress_configs_cover_inverse_closed_u4_without_tiny_capacities(self):
        base = json.loads((root / 'tests/run-s4-two-rank.json').read_text())
        before = json.dumps(base, sort_keys=True)
        configs = gate.typed_stress_configs(base, 3)
        self.assertEqual(len(configs), 72)
        self.assertEqual({c['owner_backend'] for c in configs},
                         {'CUCO_RANK', 'CUB_SORT_MERGE', 'BMMA_BUCKET'})
        self.assertEqual({(c['owner_backend'], c['frontier_profile'], c['local_pre_dedup'],
                          tuple(c['topology']['logical_owner_to_rank']),
                          c['capacities']['route_slot_count']) for c in configs},
            {(o, p, d, m, b) for o in ('CUCO_RANK', 'CUB_SORT_MERGE', 'BMMA_BUCKET')
             for p in ('DENSE', 'HASH_FIRST') for d in (False, True)
             for m in ((0, 1), (1, 0)) for b in (2, 3, 4)})
        for config in configs:
            graph = config['graph']
            self.assertEqual(graph['modulus'], 3)
            self.assertEqual(graph['expected_max_unique_states'], 729)
            for index, matrix in enumerate(graph['generators']):
                inverse = graph['generators'][graph['inverse_map'][index]]
                product = [sum(matrix[r*4+k] * inverse[k*4+c] for k in range(4)) % 3
                           for r in range(4) for c in range(4)]
                self.assertEqual(product, graph['start'])
            caps = config['capacities']
            self.assertGreaterEqual(caps['layer_hash_records_per_arena'], 729)
            self.assertGreaterEqual(caps['state_ring_records'], 2 * 729)
            self.assertGreaterEqual(caps['next_bucket_capacity_records'], 729)
            self.assertGreaterEqual(caps['route_slot_records'], config['parent_batch'] * 6)
        self.assertEqual(json.dumps(base, sort_keys=True), before)

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
