"""Exact parent grouping for bounded resident edge windows."""
import random
import unittest


def parents_for_window(begin, count, degree):
    first = begin // degree
    end = 1 + (begin + count - 1) // degree if count else first
    records = []
    for parent in range(first, end):
        edge_base = parent * degree
        low = max(begin - edge_base, 0)
        high = min(degree, begin + count - edge_base)
        records.extend((parent, generator, edge_base + generator - begin)
                       for generator in range(low, high))
    return records


class EdgeWindowTest(unittest.TestCase):
    def test_exact_bijection_with_unaligned_endpoints(self):
        rng = random.Random(91231)
        for degree in (1, 2, 3, 11, 63, 255):
            for anchor in (0, 1, 31, 2**32 - 9, 2**64 - 256):
                for _ in range(40):
                    begin = anchor + rng.randrange(0, 16)
                    count = rng.randrange(0, min(65, 2**64 - 1 - begin))
                    expected = [(edge // degree, edge % degree, edge - begin)
                                for edge in range(begin, begin + count)]
                    self.assertEqual(parents_for_window(begin, count, degree), expected)


if __name__ == "__main__":
    unittest.main()
