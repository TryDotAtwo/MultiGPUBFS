import tempfile
import unittest
from pathlib import Path

import replay_lsa_cancel_candidate as replay


class VerificationEvidenceTest(unittest.TestCase):
    def test_missing_archive_preserves_process_evidence_and_fails_case(self):
        checker = getattr(replay, 'record_archive_verification', None)
        self.assertTrue(callable(checker), 'archive verification must retain process evidence')
        row = dict(pass_=True, returncodes=[0, 0], group_complete=True)
        row['pass'] = row.pop('pass_')
        with tempfile.TemporaryDirectory() as directory:
            checker(row, Path(directory), n=4)
        self.assertFalse(row['pass'])
        self.assertEqual(row['returncodes'], [0, 0])
        self.assertTrue(row['group_complete'])
        self.assertIn('verification_error', row)
        self.assertNotIn('full_state_oracle', row)


if __name__ == '__main__':
    unittest.main()
