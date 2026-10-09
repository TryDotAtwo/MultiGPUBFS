"""Acceptance deadlines remain bounded; no GPU or rank proof in these tests."""
import unittest
from replay_lsa_cancel_candidate import process_case_timeout


class DeadlineTests(unittest.TestCase):
    def test_uninstrumented_fault_bound_is_not_relaxed(self):
        self.assertEqual(process_case_timeout(None, None), 45)
        self.assertEqual(process_case_timeout('memcheck', None), 120)

    def test_racecheck_has_an_explicit_finite_coverage_window(self):
        self.assertEqual(process_case_timeout('racecheck', None), 600)
        self.assertEqual(process_case_timeout('racecheck', 900), 900)

    def test_override_is_strictly_positive_and_bounded(self):
        for value in (0, -1, True, 1.5, 3601):
            with self.assertRaises(ValueError):
                process_case_timeout(None, value)
        self.assertEqual(process_case_timeout(None, 30), 30)
