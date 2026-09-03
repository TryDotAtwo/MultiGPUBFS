import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'single_gpu_bench.py'


class BenchmarkContract(unittest.TestCase):
    def test_comparison_rejects_different_layers_and_incomplete_runs(self):
        self.assertTrue(PATH.exists(), 'benchmark harness is missing')
        spec = importlib.util.spec_from_file_location('bench', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        good = dict(status='COMPLETE', layer_sizes=[1, 6, 9], layer_sha256=['a', 'b', 'c'])
        module.verify_pair(good, dict(good))
        for bad in [dict(good, status='INCOMPLETE'), dict(good, layer_sizes=[1, 7, 8]),
                    dict(good, layer_sha256=['a', 'x', 'c'])]:
            with self.assertRaises(ValueError):
                module.verify_pair(good, bad)

    def test_statistics_keep_all_repeats_and_failures(self):
        self.assertTrue(PATH.exists(), 'benchmark harness is missing')
        spec = importlib.util.spec_from_file_location('bench', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [dict(status='COMPLETE', search_seconds=x) for x in [1, 2, 3, 4, 100]]
        self.assertEqual(module.timing_stats(rows), dict(median_seconds=3, mad_seconds=1, repeats=5))
        with self.assertRaises(ValueError):
            module.timing_stats(rows + [dict(status='TIMEOUT')])


if __name__ == '__main__':
    unittest.main()
