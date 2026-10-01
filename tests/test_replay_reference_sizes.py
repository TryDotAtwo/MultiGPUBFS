import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_lsa_cancel_candidate as replay


class ReferenceSizes(unittest.TestCase):
    def test_s3_layers_and_canonical_matrix(self):
        self.assertTrue(hasattr(replay, 's_reference_layers'))
        layers = replay.s_reference_layers(3)
        self.assertEqual(list(map(len, layers)), [1, 3, 2])
        self.assertEqual(layers[0], {bytes([1, 0, 0, 0, 1, 0, 0, 0, 1])})
        self.assertEqual(len(set.union(*layers)), 6)

    def test_oracle_rejects_unbounded_sizes(self):
        self.assertTrue(hasattr(replay, 's_reference_layers'))
        for n in (0, 1, 9):
            with self.assertRaises(ValueError):
                replay.s_reference_layers(n)


if __name__ == '__main__':
    unittest.main()
