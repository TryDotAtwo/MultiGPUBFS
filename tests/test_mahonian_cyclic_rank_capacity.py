import itertools
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from mahonian_cyclic_rank_capacity import capacity_estimate, layer_owner_counts


class CyclicRankCapacityTests(unittest.TestCase):
    def test_small_domains_against_independent_lexical_enumeration(self):
        worlds = (1, 2, 3, 5, 8)
        for n in range(2, 9):
            depth_count = n * (n - 1) // 2 + 1
            expected = {g: [[0] * g for _ in range(depth_count)] for g in worlds}
            for state_id, state in enumerate(itertools.permutations(range(n))):
                depth = sum(state[i] > state[j] for i in range(n) for j in range(i + 1, n))
                for g in worlds:
                    expected[g][depth][state_id % g] += 1
            for g in worlds:
                with self.subTest(n=n, world=g):
                    actual = layer_owner_counts(n, g)
                    self.assertEqual(actual, expected[g])
                    self.assertEqual(sum(map(sum, actual)), math.factorial(n))

    def test_s12_s13_physical_live_row_receipts(self):
        observed = {
            12: [6343679, 6343679, 6455414, 6455414, 6455414, 6455414, 6343679, 6343679],
            13: [73646470, 73646470, 74675225, 74675225, 74675225, 74675225, 73646470, 73646470],
        }
        for n, rows in observed.items():
            with self.subTest(n=n):
                actual = capacity_estimate(n, 8)
                self.assertEqual([x["peak_live_rows"] for x in actual["by_logical_owner"]], rows)

    def test_invalid_u64_domain_and_packed_codec_eligibility(self):
        with self.assertRaises(ValueError):
            layer_owner_counts(21, 8)
        with self.assertRaises(ValueError):
            layer_owner_counts(8, 0)
        self.assertIsNone(capacity_estimate(17, 8)["max_packed_min_bytes"])


if __name__ == "__main__":
    unittest.main()
