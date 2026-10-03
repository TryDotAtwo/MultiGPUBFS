"""Supervisor contract tests; mocks do not prove Linux rank cleanup."""
import importlib.util
from pathlib import Path
import subprocess
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    'protocol_gate', Path(__file__).resolve().parents[1] / 'kaggle/lsa-bfs-gate/kernel.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class SupervisorTests(unittest.TestCase):
    def test_device_only_probe_requires_both_activation_markers_without_window(self):
        command = [sys.executable, '-c',
            "import os; print('rank='+os.environ['MGBFS_WINDOW_RANK']+' stage=device_comm_only_create result=PASS')"]
        with tempfile.TemporaryDirectory() as directory:
            row = gate.run_window_process_pair(command, '.', dict(os.environ), Path(directory),
                required_stage='device_comm_only_create', require_window=False)
            self.assertTrue(row['pass'])
            self.assertEqual(row['registered_ranks'], [0, 0])
            row = gate.run_window_process_pair([sys.executable, '-c', 'pass'], '.',
                dict(os.environ), Path(directory) / 'missing',
                required_stage='device_comm_only_create', require_window=False)
            self.assertFalse(row['pass'])

    def test_build_target_matches_admitted_hardware(self):
        for hardware, expected in [('T4', '75'), ('RTX2070', '75'), ('A4000', '86')]:
            with self.subTest(hardware=hardware):
                self.assertEqual(gate.cuda_build_target(hardware), expected)
        with self.assertRaises(ValueError):
            gate.cuda_build_target('unknown')

    def test_window_pair_requires_both_rank_registration_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, '-c',
                "import os; print('rank='+os.environ['MGBFS_WINDOW_RANK']+' mode=nonblocking stage=window_register result=PASS')"]
            row = gate.run_window_process_pair(command, '.', dict(os.environ), Path(directory), timeout=5)
            self.assertEqual(row['returncodes'], [0, 0])
            self.assertEqual(row['registered_ranks'], [1, 1])
            self.assertFalse(row['timed_out'])
            self.assertTrue(row['pass'])
            missing = gate.run_window_process_pair([sys.executable, '-c', 'pass'], '.',
                dict(os.environ), Path(directory) / 'missing', timeout=5)
            self.assertFalse(missing['pass'])

    def test_window_pair_timeout_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            row = gate.run_window_process_pair([sys.executable, '-c',
                'import time; time.sleep(30)'], '.', dict(os.environ), Path(directory), timeout=0.2)
            self.assertTrue(row['timed_out'])
            self.assertFalse(row['pass'])
            self.assertTrue(all(code is not None for code in row['returncodes']))

    def test_device_communicator_probe_must_reach_requested_stage_on_both_ranks(self):
        code = "import os; print('rank='+os.environ['MGBFS_WINDOW_RANK']+' mode=nonblocking stage=window_register result=PASS')"
        with tempfile.TemporaryDirectory() as directory:
            missing = gate.run_window_process_pair([sys.executable, '-c', code], '.',
                dict(os.environ), Path(directory) / 'missing', required_stage='device_comm_create')
            self.assertFalse(missing['pass'])
            complete = gate.run_window_process_pair([sys.executable, '-c', code +
                "; print('rank='+os.environ['MGBFS_WINDOW_RANK']+' stage=device_comm_create result=PASS')"], '.',
                dict(os.environ), Path(directory) / 'complete', required_stage='device_comm_create')
            self.assertTrue(complete['pass'])

    def run_case(self, waits, returncode):
        process = Mock(returncode=returncode)
        process.wait.side_effect = waits
        with patch.object(gate.subprocess, 'Popen', return_value=process), \
                patch.object(gate.os, 'killpg', create=True) as kill, \
                patch.object(gate.signal, 'SIGKILL', 9, create=True):
            result = gate.run_protocol_replay(['replay'], '.', {}, None, timeout=1)
        return result, process, kill

    def test_success(self):
        result, process, kill = self.run_case([0], 0)
        self.assertEqual(result, {'returncode': 0, 'timed_out': False})
        process.terminate.assert_not_called()
        kill.assert_not_called()

    def test_failure(self):
        result, _, _ = self.run_case([1], 1)
        self.assertEqual(result, {'returncode': 1, 'timed_out': False})

    def test_timeout_unwinds(self):
        result, process, kill = self.run_case(
            [subprocess.TimeoutExpired('replay', 1), -15], -15)
        self.assertTrue(result['timed_out'])
        process.terminate.assert_called_once()
        kill.assert_not_called()

    def test_timeout_escalates(self):
        result, process, kill = self.run_case(
            [subprocess.TimeoutExpired('replay', 1),
             subprocess.TimeoutExpired('replay', 15), -9], -9)
        self.assertTrue(result['timed_out'])
        process.terminate.assert_called_once()
        kill.assert_called_once_with(process.pid, 9)

    @unittest.skipUnless(sys.platform == 'linux', 'real process cleanup requires Linux')
    def test_linux_two_rank_sessions_are_reaped(self):
        # Exercises the production signal handler and supervisor with actual
        # independent OS sessions, not CUDA/NCCL or the BFS executable.
        code = '''
import os, signal, subprocess, sys, time
from pathlib import Path
from replay_lsa_cancel_candidate import install_termination_handler
install_termination_handler()
children = []
try:
    for rank in range(2):
        children.append(subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(60)'],
            start_new_session=True))
    Path(sys.argv[1]).write_text(','.join(str(p.pid) for p in children))
    time.sleep(60)
finally:
    for p in children:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=5)
'''
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / 'rank-pids'
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent))
            result = gate.run_protocol_replay(
                [sys.executable, '-c', code, str(pidfile)], '.', environment,
                subprocess.DEVNULL, timeout=2)
            self.assertTrue(result['timed_out'])
            self.assertEqual(result['returncode'], 143)
            pids = [int(pid) for pid in pidfile.read_text().split(',')]
            self.assertEqual(len(pids), 2)
            for pid in pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


if __name__ == '__main__':
    unittest.main()
