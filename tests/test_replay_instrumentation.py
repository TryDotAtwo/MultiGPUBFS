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

    def test_unknown_instrumentation_rejected(self):
        self.assertTrue(hasattr(replay, 'instrument_rank_command'))
        with self.assertRaises(ValueError):
            replay.instrument_rank_command('/bin/mgbfs', [], 'not-a-tool', Path('/tmp/x'))
