"""Regression: selecting a fault must preserve its key/rank after env sanitization."""
import unittest
from replay_lsa_cancel_candidate import select_replay_cases

class ReplayCaseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cases = [('healthy', None, None),
            ('worker_write', 'MGBFS_TEST_ARCHIVE_WORKER_WRITE_FAULT_RANK', 0),
            ('worker_write', 'MGBFS_TEST_ARCHIVE_WORKER_WRITE_FAULT_RANK', 1)]
    def test_selects_real_rank_zero_fault_not_healthy_environment_override(self):
        self.assertEqual(select_replay_cases(self.cases, 'worker_write-0'),
            [('worker_write', 'MGBFS_TEST_ARCHIVE_WORKER_WRITE_FAULT_RANK', 0)])
    def test_preserves_default_full_case_matrix(self):
        self.assertIs(select_replay_cases(self.cases), self.cases)
    def test_healthy_only_does_not_inject_fault(self):
        self.assertEqual(select_replay_cases(self.cases, healthy_only=True), [('healthy', None, None)])
    def test_unknown_or_ambiguous_case_fails_closed(self):
        for cases, label in [(self.cases, 'worker_write-2'), (self.cases + [self.cases[1]], 'worker_write-0')]:
            with self.assertRaises(ValueError): select_replay_cases(cases, label)
    def test_fault_filter_cannot_silently_lose_to_healthy_only(self):
        with self.assertRaises(ValueError): select_replay_cases(self.cases, 'worker_write-0', True)

if __name__ == '__main__': unittest.main()
