"""Supervisor contract tests; mocks do not prove Linux rank cleanup."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    'protocol_gate', Path(__file__).resolve().parents[1] / 'kaggle/lsa-bfs-gate/kernel.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class SupervisorTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
