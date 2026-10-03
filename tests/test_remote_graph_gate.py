import sys
import tempfile
import unittest
import subprocess
from unittest.mock import Mock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from remote_graph_gate import evidence_files, run_group


class EvidenceSelectionTests(unittest.TestCase):
    def test_timeout_kills_entire_child_group_and_reaps_leader(self):
        process = Mock(pid=123)
        process.wait.side_effect = [subprocess.TimeoutExpired(['cargo'], 1), 0]
        with patch('remote_graph_gate.subprocess.Popen', return_value=process) as start, \
             patch('remote_graph_gate.os.killpg', create=True) as kill, \
             patch('remote_graph_gate.signal.SIGKILL', 9, create=True):
            with self.assertRaises(subprocess.TimeoutExpired):
                run_group(['cargo'], env={}, stdout=None, timeout=1)
            self.assertTrue(start.call_args.kwargs['start_new_session'])
            kill.assert_called_once()
            self.assertEqual(kill.call_args.args[0], 123)
            self.assertEqual(process.wait.call_count, 2)

    def test_only_diagnostics_are_published_never_states_tokens_or_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work/'logs').mkdir()
            for name in ('gate-summary.json', 'build-summary.json', 'graph-oracle.log',
                         'build-driver.log', 'hf-token', 'layer.parquet', 'runtime-env.json', 'mgbfs'):
                (work/name).write_text('fixture')
            (work/'logs/compiler.log').write_text('fixture')
            (work/'logs/credential.json').write_text('fixture')
            self.assertEqual({str(p.relative_to(work)).replace('\\', '/') for p in evidence_files(work)},
                {'gate-summary.json', 'build-summary.json', 'graph-oracle.log',
                 'build-driver.log', 'logs/compiler.log'})


if __name__ == '__main__':
    unittest.main()
