import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from distributed_gpu_bench import smi_peaks


class MemoryMetrics(unittest.TestCase):
    def test_missing_rank_is_unknown_not_zero(self):
        self.assertEqual(smi_peaks(""), ([None, None], None))
        self.assertEqual(smi_peaks("t, 0, uuid, 500, 1, 1, 1, 1\n"), ([500.0, None], None))

    def test_both_rank_maxima_and_sum(self):
        text = "\n".join([
            "t, 0, a, 500, 1, 1, 1, 1",
            "t, 1, b, 400, 1, 1, 1, 1",
            "t, 0, a, 600, 1, 1, 1, 1",
            "t, 3, c, 999, 1, 1, 1, 1",
            "t, 1, b, N/A, 1, 1, 1, 1",
            "t, 1, b, nan, 1, 1, 1, 1",
        ])
        self.assertEqual(smi_peaks(text), ([600.0, 400.0], 1000.0))


if __name__ == "__main__":
    unittest.main()
