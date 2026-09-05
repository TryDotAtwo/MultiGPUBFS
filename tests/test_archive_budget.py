import unittest
import importlib.util
from pathlib import Path

class Budget(unittest.TestCase):
    def test_replay_and_explicit_host_budget(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/archive_budget.py'
        self.assertTrue(path.exists(), 'archive replay planner missing')
        spec = importlib.util.spec_from_file_location('budget', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertEqual(m.replay([10, 30, 0], [1, 2, 4], 5),
                         {'peak_records': 25, 'remaining_records': 5})
        self.assertEqual(m.host_budget(1000, 2, 10, 2, 1, 20, 30, 40)['required_bytes'], 770)
        with self.assertRaisesRegex(ValueError, 'HOST_RAM_PREFLIGHT'):
            m.host_budget(769, 2, 10, 2, 1, 20, 30, 40)
        for counts, times, rate in [([1], [], 1), ([1], [-1], 1), ([1], [1], 0)]:
            with self.assertRaises(ValueError):
                m.replay(counts, times, rate)

if __name__ == '__main__':
    unittest.main()
