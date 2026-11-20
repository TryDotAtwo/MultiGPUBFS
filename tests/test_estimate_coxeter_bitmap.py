"""Validate the layer/owner DP against independent enumeration of small S_n."""
import importlib.util
import itertools
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('capacity_estimator', Path(__file__).resolve().parents[1] / 'scripts/estimate_coxeter_bitmap.py')
estimator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator)


class CoxeterEstimateTests(unittest.TestCase):
    def test_rank_owner_and_inversions_against_enumeration(self):
        for n in range(1, 8):
            for world in (1, 2, 3, 8):
                with self.subTest(n=n, world=world):
                    actual = estimator.layers_by_owner(n, world)
                    expected = [[0] * world for _ in range(n * (n - 1) // 2 + 1)]
                    for lex_rank, perm in enumerate(itertools.permutations(range(n))):
                        inv = sum(perm[i] > perm[j] for i in range(n) for j in range(i + 1, n))
                        expected[inv][lex_rank % world] += 1
                    self.assertEqual(actual, expected)
    def test_exact_total_and_bitmap_word_alignment(self):
        r = estimator.estimate(14, 8, 1 << 20)
        self.assertEqual(r['vertices'], math.factorial(14))
        self.assertEqual(r['layers'], 92)
        self.assertEqual(r['peak_owner_layer_rows'], 469860208)
        self.assertEqual(r['bitmap_max_per_gpu_bytes'] % 8, 0)
    def test_balanced_hash_estimate_matches_segment_layout(self):
        r = estimator.estimate(14, 8, 1 << 20)
        rows = 1 << 20
        slots = 1 << ((6 * rows - 1).bit_length())
        expected_segment_bytes = 8 * slots + 3 * rows * (14 + 8) + 24
        peak = max(map(sum, estimator.layers_by_owner(14, 8)))
        segments = ((peak + 7) // 8 + rows - 1) // rows
        self.assertEqual(r['hash_balanced_ideal_segments_per_gpu'], segments)
        self.assertEqual(r['hash_balanced_ideal_segment_bytes_per_gpu'], segments * expected_segment_bytes)

    def test_unsupported_parameters_rejected(self):
        for n, world, rows in ((21,8,1), (0,8,1), (14,0,1), (14,129,1), (14,8,0), (14,8,(1<<32)+1)):
            with self.subTest(n=n, world=world, rows=rows), self.assertRaises(ValueError):
                estimator.estimate(n, world, rows)


if __name__ == '__main__':
    unittest.main()
