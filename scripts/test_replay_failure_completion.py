"""Real output-file fixtures; this does not stand in for a two-GPU run."""
import json
from pathlib import Path
import tempfile
import unittest
from replay_lsa_cancel_candidate import failure_has_no_complete


class FailureCompletionTests(unittest.TestCase):
    def test_missing_outputs_are_not_false_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(failure_has_no_complete(Path(directory)))

    def test_either_rank_complete_rejects_failed_run_without_group_marker(self):
        for rank in (0, 1):
            with tempfile.TemporaryDirectory() as directory:
                case = Path(directory)
                (case / 'result').mkdir()
                (case / f'result/rank-{rank}.json').write_text(
                    json.dumps({'status': 'COMPLETE'}))
                self.assertFalse(failure_has_no_complete(case))

    def test_explicit_incomplete_rank_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / 'result').mkdir()
            (case / 'result/rank-0.json').write_text('{"status":"INCOMPLETE"}')
            self.assertTrue(failure_has_no_complete(case))

    def test_group_marker_rejects_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / 'result').mkdir()
            (case / 'result/group-complete.json').write_text('{}')
            self.assertFalse(failure_has_no_complete(case))

    def test_malformed_rank_output_is_not_accepted_as_failure_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / 'result').mkdir()
            (case / 'result/rank-1.json').write_text('{truncated')
            self.assertFalse(failure_has_no_complete(case))
