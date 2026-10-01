import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_lsa_cancel_candidate as replay


class InstrumentationCommandTest(unittest.TestCase):
    def command(self, tool):
        self.assertTrue(hasattr(replay, 'instrument_rank_command'))
        return replay.instrument_rank_command('/bin/mgbfs', ['bench', '--reference'],
                                              tool, Path('/tmp/rank-1'))

    def test_memcheck_preserves_target_and_reports_api_errors(self):
        self.assertEqual(self.command('memcheck'), [
            '/usr/local/cuda/bin/compute-sanitizer', '--tool', 'memcheck',
            '--error-exitcode', '97', '/bin/mgbfs', 'bench', '--reference'])

    def test_plain_process_has_no_instrumentation(self):
        self.assertEqual(self.command(None), ['/bin/mgbfs', 'bench', '--reference'])

    def test_nsys_collects_copy_and_wait_callchains(self):
        command = self.command('nsys')
        self.assertIn('--sample=process-tree', command)
        self.assertIn('--cudabacktrace=memory:0,sync:0,other:0', command)

    def test_unknown_instrumentation_rejected(self):
        self.assertTrue(hasattr(replay, 'instrument_rank_command'))
        with self.assertRaises(ValueError):
            replay.instrument_rank_command('/bin/mgbfs', [], 'not-a-tool', Path('/tmp/x'))

    def test_missing_or_dirty_sanitizer_summary_cannot_pass(self):
        self.assertTrue(hasattr(replay, 'instrumentation_clean'))
        self.assertFalse(replay.instrumentation_clean('', 'memcheck'))
        self.assertFalse(replay.instrumentation_clean(
            'ERROR SUMMARY: 0 errors\nERROR SUMMARY: 1 errors', 'memcheck'))
        self.assertFalse(replay.instrumentation_clean(
            'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 1 warnings)', 'racecheck'))
        self.assertTrue(replay.instrumentation_clean('ERROR SUMMARY: 0 errors', 'synccheck'))
