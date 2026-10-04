"""Cleanup contracts; mocks do not prove CUDA/NCCL cancellation."""
import subprocess
import os
import sys
import unittest
from unittest.mock import Mock, patch
from replay_lsa_cancel_candidate import cleanup_rank_processes


class CleanupTests(unittest.TestCase):
    def setUp(self):
        sigkill = patch('replay_lsa_cancel_candidate.signal.SIGKILL', 9, create=True)
        sigkill.start()
        self.addCleanup(sigkill.stop)

    def test_signal_all_ranks_before_waiting_on_one(self):
        order = []
        ranks = [Mock(pid=11), Mock(pid=12)]
        for rank in ranks:
            rank.poll.return_value = None
            rank.wait.side_effect = lambda timeout, pid=rank.pid: order.append(('wait', pid))
        with patch('replay_lsa_cancel_candidate.os.killpg', create=True,
                   side_effect=lambda pid, sig: order.append(('kill', pid))):
            self.assertEqual(cleanup_rank_processes(ranks), [])
        self.assertEqual(order, [('kill', 11), ('kill', 12), ('wait', 11), ('wait', 12)])

    def test_one_timeout_does_not_skip_other_rank_reaping(self):
        ranks = [Mock(pid=11), Mock(pid=12)]
        for rank in ranks:
            rank.poll.return_value = None
        ranks[0].wait.side_effect = subprocess.TimeoutExpired('rank', 10)
        with patch('replay_lsa_cancel_candidate.os.killpg', create=True):
            errors = cleanup_rank_processes(ranks)
        self.assertEqual(len(errors), 1)
        ranks[1].wait.assert_called_once_with(timeout=10)

    def test_exit_race_is_not_cleanup_failure(self):
        rank = Mock(pid=11)
        rank.poll.return_value = None
        with patch('replay_lsa_cancel_candidate.os.killpg', create=True, side_effect=ProcessLookupError):
            self.assertEqual(cleanup_rank_processes([rank]), [])
        rank.wait.assert_called_once_with(timeout=10)

    @unittest.skipUnless(sys.platform == 'linux', 'real process sessions require Linux')
    def test_two_real_rank_sessions_are_reaped(self):
        ranks = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                  start_new_session=True) for _ in range(2)]
        try:
            self.assertEqual(cleanup_rank_processes(ranks), [])
            for rank in ranks:
                self.assertIsNotNone(rank.returncode)
                with self.assertRaises(ProcessLookupError):
                    os.kill(rank.pid, 0)
        finally:
            cleanup_rank_processes(ranks)
