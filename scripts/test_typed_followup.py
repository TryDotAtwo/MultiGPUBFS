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
    def test_matrix_gate_covers_moduli_profiles_owners_maps_prededup_and_frozen_seeds(self):
        base = json.loads((root / 'tests/run-s4-two-rank.json').read_text())
        before = json.dumps(base, sort_keys=True)
        cases = gate.typed_matrix_cases(base)
        observed = {(c['config']['graph']['modulus'], c['config']['frontier_profile'],
            c['config']['owner_backend'], c['config']['local_pre_dedup'],
            tuple(c['config']['topology']['logical_owner_to_rank']),
            int.from_bytes(bytes(c['config']['seed']), 'little')) for c in cases}
        expected = {(m, p, o, d, ranks, seed) for m in range(2, 7)
            for p in ('DENSE', 'HASH_FIRST')
            for o in ('CUB_SORT_MERGE', 'BMMA_BUCKET', 'CUCO_RANK')
            for d in (False, True) for ranks in ((0, 1), (1, 0))
            for seed in (0, 1, 20260828)}
        self.assertEqual(observed, expected)
        self.assertEqual(len(cases), 360)
        self.assertEqual(len({c['label'] for c in cases}), 360)
        for c in cases:
            config = c['config']
            self.assertEqual(config['graph']['expected_max_unique_states'], config['graph']['modulus'] ** 6)
            self.assertEqual(config['capacities']['route_slot_records'], 1536)
            self.assertEqual(config['capacities']['pinned_archive_slot_bytes'], 8192)
            self.assertEqual(c['extra'], ['--healthy-only', '--unitriangular-modulus', str(config['graph']['modulus'])])
        self.assertEqual(json.dumps(base, sort_keys=True), before)
        self.assertIn('pyarrow==19.0.1', gate.oracle_dependency_packages('typed_matrix_gate'))

    def test_wrong_or_missing_actual_instrumenter_cannot_pass_version_gate(self):
        expected = '/sdk/compute-sanitizer/compute-sanitizer'
        good = {'executable': expected, 'sha256': 'a' * 64, 'version': 'Version 2025.2'}
        self.assertTrue(gate.sanitizer_selection_matches(good, expected, 'a' * 64))
        for bad in (None, {}, dict(good, executable='/usr/local/cuda/bin/compute-sanitizer'),
                    dict(good, sha256='b' * 64), dict(good, version='')):
            self.assertFalse(gate.sanitizer_selection_matches(bad, expected, 'a' * 64))

    def test_macro_capture_gate_runs_real_lib_test_not_zero_test_filter(self):
        command = gate.macro_capture_command()
        self.assertEqual(command[:8], ['cargo', 'test', '--locked', '-p',
            'mgbfs-runtime', '--features', 'cuda,library-owner', '--lib'])
        self.assertIn('macro_native::producer_capture_tests::macro_produce_captures_and_runs_without_host_count_readback', command)
        self.assertEqual(command[-4:], ['--', '--exact', '--nocapture', '--test-threads=1'])

    def test_version_comparison_selects_host_before_sdk_wrapper_and_does_not_mutate_base(self):
        env = {'PATH': '/sdk/bin:/usr/bin', 'keep': 'yes'}
        for version, prefix in (('host', '/usr/local/cuda/bin'), ('cuda129', '/sdk/compute-sanitizer')):
            selected = gate.sanitizer_version_environment(env, '/usr/local/cuda/bin/compute-sanitizer',
                '/sdk/compute-sanitizer/compute-sanitizer', version)
            self.assertEqual(selected['PATH'], prefix + ':/sdk/bin:/usr/bin')
            self.assertEqual(selected['MGBFS_COMPUTE_SANITIZER'],
                '/usr/local/cuda/bin/compute-sanitizer' if version == 'host'
                else '/sdk/compute-sanitizer/compute-sanitizer')
        self.assertEqual(env, {'PATH': '/sdk/bin:/usr/bin', 'keep': 'yes'})

    def test_toolchain_comparison_keeps_all_four_tools_and_both_host_and_pinned_versions(self):
        base = json.loads((root / 'tests/run-s4-two-rank.json').read_text())
        cases = gate.typed_sanitizer_version_cases(base)
        self.assertEqual(len(cases), 16)
        self.assertEqual({(c['config']['frontier_profile'], c['version'], c['tool']) for c in cases},
            {(p, v, t) for p in ('DENSE', 'HASH_FIRST') for v in ('host', 'cuda129')
             for t in ('memcheck', 'racecheck', 'initcheck', 'synccheck')})
        for case in cases:
            self.assertEqual(case['extra'], ['--healthy-only', '--instrument-processes', case['tool']])
        self.assertIn('pyarrow==19.0.1', gate.oracle_dependency_packages('typed_sanitizer_version_gate'))

    def test_warmup_runs_both_profiles_with_asymmetric_archive_owner_capacity_faults(self):
        base = json.loads((root / 'tests/run-s4-two-rank.json').read_text())
        cases = gate.typed_warmup_cases(base)
        self.assertEqual(len(cases), 2)
        self.assertEqual({c['config']['frontier_profile'] for c in cases}, {'DENSE', 'HASH_FIRST'})
        for case in cases:
            self.assertEqual(case['extra'], ['--bench-warmup', '--capacity-faults'])
            self.assertEqual(case['config']['owner_backend'], 'CUCO_RANK')
            self.assertEqual(case['config']['capacities']['route_slot_count'], 3)
            self.assertEqual(case['config']['completion_epoch_window'], 3)
        self.assertIn('pyarrow==19.0.1', gate.oracle_dependency_packages('typed_warmup_gate'))

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
            for config in (gate.typed_stress_configs(base, 3)
                    + [c['config'] for c in gate.typed_warmup_cases(base)]
                    + [c['config'] for c in gate.typed_matrix_cases(base)]):
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
