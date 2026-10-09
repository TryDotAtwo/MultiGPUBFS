import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess
from replay_lsa_cancel_candidate import collect_stalled_rank_stacks

class StalledStackTests(unittest.TestCase):
    def test_exited_process_is_never_attached(self):
        process = Mock(pid=123)
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.run') as run:
            self.assertEqual(collect_stalled_rank_stacks([process], Path(tmp), '/usr/bin/gdb'), [])
            run.assert_not_called()

    def test_attach_is_bounded_and_uses_exact_owned_pid(self):
        process = Mock(pid=987654321)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.run') as run:
            run.return_value.returncode = 0
            rows = collect_stalled_rank_stacks([process], Path(tmp), '/usr/bin/gdb')
            command = run.call_args.args[0]
            self.assertEqual(command[:5], ['/usr/bin/gdb', '-nx', '-batch', '-p', '987654321'])
            self.assertEqual(run.call_args.kwargs['timeout'], 8)
            self.assertEqual(rows[0]['gdb_returncode'], 0)

    def test_debugger_timeout_remains_evidence_not_success(self):
        process = Mock(pid=987654321)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.run') as run:
            run.side_effect = subprocess.TimeoutExpired('gdb', 8)
            rows = collect_stalled_rank_stacks([process], Path(tmp), '/usr/bin/gdb')
            self.assertIn('gdb_error', rows[0])
            self.assertNotIn('gdb_returncode', rows[0])
            self.assertNotIn('pass', rows[0])
