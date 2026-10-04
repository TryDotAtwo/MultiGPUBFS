"""Selection/guard contracts only; no claim of GPU correctness."""
import importlib.util
from pathlib import Path
import unittest

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('vmm_gate', root / 'kaggle/lsa-bfs-gate/kernel.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class VmmProbeTests(unittest.TestCase):
    def test_matrix_contains_unfiltered_local_and_import_controls(self):
        cases = gate.vmm_probe_cases()
        self.assertEqual(len(cases), 10)
        self.assertEqual({(c['mode'], c['tool']) for c in cases},
                         {(m, t) for m in ('local', 'import')
                          for t in (None, 'memcheck', 'racecheck', 'initcheck', 'synccheck')})
        self.assertEqual(len({c['label'] for c in cases}), 10)

    def test_windowless_arbitrary_stage_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'WINDOWLESS_PROBE'):
            gate.run_window_process_pair([], root, {}, root / 'build/unused',
                                         require_window=False, required_stage='anything')


if __name__ == '__main__':
    unittest.main()
