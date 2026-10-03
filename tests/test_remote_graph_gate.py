import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from remote_graph_gate import evidence_files


class EvidenceSelectionTests(unittest.TestCase):
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
