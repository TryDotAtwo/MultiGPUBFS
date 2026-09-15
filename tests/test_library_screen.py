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
                    rank_results=[dict(owner_backend='CUDF_RELATIONAL',
                        archive_enabled=True, warmup_completed=True,
                        library_pool_reserved_bytes=67108864)])

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
