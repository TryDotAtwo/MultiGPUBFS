import sys
import unittest
import tempfile
import json
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import library_gpu_screen as screen


class ScreenContract(unittest.TestCase):
    def test_profile_keeps_rank_command_and_archives_but_not_benchmark_statistics(self):
        row = self.row()
        row.update(search_complete_seconds=1, smi_peak_mib_per_rank=[100, 100],
                   smi_peak_mib_total=200)
        launched = []
        def launch(command, out, label, env, timeout):
            launched.append(command)
            (out/'timeline.nsys-rep').write_bytes(b'trace fixture')
            (out/'measure.json').write_text(json.dumps(row))
            return row
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(screen, 'run_group', launch), patch.object(
                    screen.subprocess, 'run', return_value=SimpleNamespace(
                        stdout='{"status":"VERIFIED"}', returncode=0)):
                result = screen.run_case('mgbfs', root/'logs', root/'archive',
                    's3', 6, 2, 7, 64, 128, 67108864, 'DENSE', 'ON', {},
                    nsys='/opt/nsys')
            command = launched[0]
            self.assertEqual(command[:2], ['/opt/nsys', 'profile'])
            self.assertIn('--sample=none', command)
            self.assertIn('--cpuctxsw=none', command)
            self.assertIn('--nproc-per-node=2', command)
            self.assertEqual(command[-1], '{RANK_OUT}')
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(len(result['archive_verification']), 2)
            self.assertTrue(result['profiled'])
            self.assertNotIn('statistics', result)
            saved = json.loads((root/'logs'/'screen-summary.json').read_text())
            self.assertNotIn('statistics', saved)
            self.assertTrue(saved['measurement']['profiled'])
            raw = json.loads((root/'logs'/'measure.json').read_text())
            self.assertTrue(raw.get('profiled'), 'raw timings must retain profiler provenance')

    def test_profile_without_trace_is_failure_not_timing_evidence(self):
        row = self.row()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(screen, 'run_group', return_value=row):
                with self.assertRaisesRegex(ValueError, 'SCREEN_TRACE_MISSING'):
                    screen.run_case('mgbfs', root/'logs', root/'archive',
                        's3', 6, 2, 7, 64, 128, 67108864, 'DENSE', 'ON', {}, nsys='nsys')
            saved = json.loads((root/'logs'/'screen-summary.json').read_text())
            self.assertEqual(saved['status'], 'FAILED')
            self.assertNotIn('statistics', saved)

    def test_preserved_native_reports_need_no_library_pool(self):
        row = self.row()
        for rank in row['rank_results']:
            rank['owner_backend'] = 'CUB_SORT_MERGE'
            rank['backend'] = 'native_nccl_dense_ring_v2'
            del rank['library_pool_reserved_bytes']
        screen.validate_result(row, 6, 0, owner='CUB_SORT_MERGE')
        row['rank_results'][0]['backend'] = 'library_nccl_dense_cuco_v1'
        with self.assertRaises(ValueError):
            screen.validate_result(row, 6, 0, owner='CUB_SORT_MERGE')

    def test_native_example_uses_old_binary_arguments_and_same_archive_verifier(self):
        row = self.row()
        for rank in row['rank_results']:
            rank.update(owner_backend='CUB_SORT_MERGE', backend='native_nccl_dense_ring_v2')
            del rank['library_pool_reserved_bytes']
        row.update(search_complete_seconds=1, smi_peak_mib_per_rank=[100, 100], smi_peak_mib_total=200)
        launched = []
        def launch(command, out, label, env, timeout):
            launched.append((command, env))
            return row
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(screen, 'run_group', launch), patch.object(
                    screen.subprocess, 'run', return_value=SimpleNamespace(
                        stdout='{"status":"VERIFIED"}', returncode=0)) as verify:
                result = screen.run_case('old-mgbfs', root/'logs', root/'archive',
                    's3', 6, 2, 7, 64, 128, 0, 'DENSE', 'ON', {},
                    owner='CUB_SORT_MERGE', native_example='old-distributed-bench')
            self.assertEqual(result['status'], 'COMPLETE')
            command, env = launched[0]
            self.assertEqual(command[command.index('--no-python') + 1:][:3],
                             ['old-distributed-bench', 's3', '7'])
            self.assertNotIn('--reference', command)
            self.assertEqual(env['MGBFS_BENCH_CAPACITY'], '64')
            self.assertEqual(env['MGBFS_STATE_CODEC'], 'matrix_u8')
            self.assertEqual(verify.call_args_list[0].args[0][0], 'old-mgbfs')

    def test_cuco_measurement_cannot_be_replaced_by_cudf(self):
        row = self.row()
        with self.assertRaisesRegex(ValueError, 'SCREEN_LIBRARY_CONTRACT'):
            screen.validate_result(row, 6, 67108864, owner='CUCO_INDEXED')
        for rank in row['rank_results']:
            rank['owner_backend'] = 'CUCO_INDEXED'
        screen.validate_result(row, 6, 67108864, owner='CUCO_INDEXED')

    def test_explicit_cuco_run_preserves_dispatch_and_archive_contract(self):
        row = self.row()
        row['rank_results'] = row['rank_results'][:1]
        row['rank_results'][0]['owner_backend'] = 'CUCO_INDEXED'
        row['rank_results'][0].update(world_size=1, pre_dedup='OFF')
        row.update(search_complete_seconds=1, smi_peak_mib_per_rank=[100], smi_peak_mib_total=100)
        launched = []
        def launch(command, out, label, env, timeout):
            launched.append(env.copy())
            return row
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(screen, 'run_group', launch), patch.object(
                    screen.subprocess, 'run', return_value=SimpleNamespace(
                        stdout='{"status":"VERIFIED"}', returncode=0)):
                report = screen.run_case('mgbfs', root/'logs', root/'archive',
                    's3', 6, 1, 7, 64, 128, 67108864, 'DENSE', 'OFF', {},
                    owner='CUCO_INDEXED')
            self.assertEqual(report['status'], 'COMPLETE')
            self.assertEqual(launched[0]['MGBFS_OWNER_BACKEND'], 'CUCO_INDEXED')
            self.assertEqual(launched[0]['MGBFS_PRE_DEDUP'], 'OFF')
            self.assertEqual(launched[0]['MGBFS_BENCH_SKIP_ARCHIVE'], '0')
            self.assertEqual(len(report['archive_verification']), 1)

    def test_failed_gpu_run_is_saved_without_archive_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(screen, 'run_group', return_value=dict(status='FAILED', exit_code=9)), \
                 patch.object(screen.subprocess, 'run') as verify:
                with self.assertRaisesRegex(ValueError, 'SCREEN_INCOMPLETE'):
                    screen.run_case('mgbfs', root/'logs', root/'archive',
                        's3', 6, 2, 7, 64, 128, 67108864, 'DENSE', 'ON', {})
                verify.assert_not_called()
            saved = json.loads((root/'logs'/'screen-summary.json').read_text())
            self.assertEqual(saved['status'], 'FAILED')
            self.assertEqual(saved['measurement']['exit_code'], 9)
            self.assertNotIn('statistics', saved)

    def test_case_runs_two_processes_and_verifies_archives(self):
        row = self.row()
        row.update(search_complete_seconds=1, smi_peak_mib_per_rank=[100, 100],
                   smi_peak_mib_total=200)
        launches = []
        def launch(command, out, label, env, timeout):
            launches.append((command, env))
            return row
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(screen, 'run_group', launch), patch.object(
                    screen.subprocess, 'run', return_value=SimpleNamespace(
                        stdout='{"status":"VERIFIED"}', returncode=0)):
                result = screen.run_case('mgbfs', root/'logs', root/'archive',
                    's3', 6, 2, 7, 64, 128, 67108864, 'DENSE', 'ON', {})
            self.assertEqual(result['status'], 'COMPLETE')
            saved = json.loads((root/'logs'/'screen-summary.json').read_text())
            self.assertEqual(len(saved['archive_verification']), 2)
            self.assertIn('--nproc-per-node=2', launches[0][0])
            self.assertEqual(launches[0][1]['MGBFS_BENCH_WARMUP'], '1')
            self.assertEqual(launches[0][1]['MGBFS_BENCH_SKIP_ARCHIVE'], '0')

    def row(self):
        return dict(status='COMPLETE', layer_sizes=[1, 5],
                    durable_run_commit_seconds=2,
                    rank_results=[dict(rank=rank, group='s3', batch=7, world_size=2,
                        frontier_profile='DENSE', pre_dedup='ON', generation_variant=1,
                        hash_first_generation='SCALAR', declared_capacity_records=64,
                        declared_state_ring_records=128, capacity_mode='MaxPerRank',
                        owner_backend='CUDF_RELATIONAL',
                        archive_enabled=True, warmup_completed=True,
                        library_pool_reserved_bytes=67108864) for rank in range(2)])

    def test_wrong_requested_workload_is_not_a_comparable_measurement(self):
        expected = dict(group='s3', batch=7, world_size=2, frontier_profile='DENSE',
                        pre_dedup='ON', declared_capacity_records=64,
                        declared_state_ring_records=128)
        screen.validate_result(self.row(), 6, 67108864, expected=expected)
        for key, value in [('group', 's4'), ('batch', 8), ('world_size', 1),
                           ('frontier_profile', 'HASH_FIRST'), ('pre_dedup', 'OFF'),
                           ('declared_capacity_records', 32), ('declared_state_ring_records', 64)]:
            row = self.row()
            row['rank_results'][1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'SCREEN_CONFIGURATION'):
                screen.validate_result(row, 6, 67108864, expected=expected)

    def test_missing_peer_is_not_a_complete_measurement(self):
        row = self.row()
        row['rank_results'].pop()
        with self.assertRaisesRegex(ValueError, 'SCREEN_RANKS'):
            screen.validate_result(row, 6, 67108864, expected=dict(world_size=2))

    def test_complete_counts_and_library_contract(self):
        screen.validate_result(self.row(), 6, 67108864)

    def test_wrong_count_is_not_a_measurement(self):
        with self.assertRaises(ValueError):
            screen.validate_result(self.row(), 7, 67108864)

    def test_missing_archive_or_warmup_or_wrong_owner_rejected(self):
        for key, value in [('archive_enabled', False), ('warmup_completed', False),
                           ('owner_backend', 'CUB_SORT_MERGE'),
                           ('library_pool_reserved_bytes', 0)]:
            row = self.row()
            row['rank_results'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate_result(row, 6, 67108864)

    def test_failure_and_absent_durable_rejected(self):
        for key, value in [('status', 'FAILED'), ('durable_run_commit_seconds', None)]:
            row = self.row()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate_result(row, 6, 67108864)
