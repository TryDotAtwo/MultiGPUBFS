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
        self.assertEqual(len(cases), 30)
        self.assertEqual({(c['mode'], c['tool'], c['runtime_init'], c['symmetric']) for c in cases},
                         {(m, t, r, s) for m in ('local', 'import')
                          for r, s in ((False, False), (True, False), (True, True))
                          for t in (None, 'memcheck', 'racecheck', 'initcheck', 'synccheck')})
        self.assertEqual(len({c['label'] for c in cases}), 30)

    def test_windowless_arbitrary_stage_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'WINDOWLESS_PROBE'):
            gate.run_window_process_pair([], root, {}, root / 'build/unused',
                                         require_window=False, required_stage='anything')


if __name__ == '__main__':
    unittest.main()
