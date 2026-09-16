"""Reject skipped tests and sanitizer failures; exercise real process deadlines."""
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import eight_gpu_gate as gate
except ImportError:
    gate = None


class EightGpuGateTests(unittest.TestCase):
    def test_exactly_eight_distinct_h200_devices_required(self):
        self.assertIsNotNone(gate)
        rows = [f'{i}, NVIDIA H200, GPU-{i}, 143771, 140000' for i in range(8)]
        self.assertEqual(len(gate.validate_inventory('\n'.join(rows))), 8)
        for invalid in [rows[:7], rows + [rows[0]], rows[:-1] + [rows[0]],
                        [r.replace('H200', 'H100') for r in rows]]:
            with self.assertRaises(ValueError):
                gate.validate_inventory('\n'.join(invalid))

    def test_exit_zero_with_ignored_or_missing_test_is_not_pass(self):
        self.assertIsNotNone(gate)
        passed = 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out;'
        gate.validate_log(passed, 'plain')
        for bad in ['', passed.replace('1 passed', '0 passed'),
                    passed.replace('0 ignored', '1 ignored'),
                    passed.replace('0 failed', '1 failed')]:
            with self.assertRaises(ValueError):
                gate.validate_log(bad, 'plain')
        gate.validate_log(passed + '\n========= ERROR SUMMARY: 0 errors', 'memcheck')
        gate.validate_log(passed + '\n========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)', 'racecheck')
        for bad in [passed, passed + '\n========= ERROR SUMMARY: 1 errors',
                    passed + '\n========= ERROR SUMMARY: 0 errors\n========= ERROR SUMMARY: 2 errors']:
            with self.assertRaises(ValueError):
                gate.validate_log(bad, 'memcheck')

    def test_real_child_timeout_and_nonzero_are_preserved(self):
        self.assertIsNotNone(gate)
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'child.log'
            with self.assertRaises(TimeoutError):
                gate.run_command([sys.executable, '-c',
                                  'import time; print("started", flush=True); time.sleep(30)'],
                                 log, dict(__import__('os').environ), 0.3)
            self.assertIn('started', log.read_text())
            with self.assertRaises(RuntimeError):
                gate.run_command([sys.executable, '-c', 'raise SystemExit(9)'],
                                 log, dict(__import__('os').environ), 10)

    def test_orchestration_retains_failure_and_never_claims_pass(self):
        self.assertIsNotNone(gate)
        real_run = gate.run_command
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root/'test-binary'
            binary.write_bytes(b'fixture, never executed as CUDA')
            def commands(command, log, env, timeout):
                if command[0] == 'nvidia-smi':
                    output = '\n'.join(f'{i}, NVIDIA H200, GPU-{i}, 143771, 140000' for i in range(8))
                elif '--version' in command:
                    output = 'fixture sanitizer version'
                else:
                    self.assertIn('--ignored', command)
                    self.assertIn('--exact', command)
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], ','.join(f'GPU-{i}' for i in range(8)))
                    output = 'test result: ok. 1 passed; 0 failed; 0 ignored;'
                    if '--tool' in command:
                        output += '\n========= ERROR SUMMARY: 1 errors'
                return real_run([sys.executable, '-c', 'print(' + repr(output) + ')'], log, env, timeout)
            argv = ['gate', '--executable', str(binary), '--logs', str(root/'logs'), '--timeout-seconds', '30']
            with patch.object(sys, 'argv', argv), patch.object(gate, 'run_command', commands):
                with self.assertRaisesRegex(ValueError, 'SANITIZER_FINDINGS'):
                    gate.main()
            report = json.loads((root/'logs/summary.json').read_text())
            self.assertEqual(report['status'], 'INCOMPLETE')
            self.assertEqual([s['tool'] for s in report['stages']], ['plain'])
            self.assertIn('1 errors', (root/'logs/memcheck.log').read_text())
            self.assertFalse((root/'logs/racecheck.log').exists())


if __name__ == '__main__':
    unittest.main()
