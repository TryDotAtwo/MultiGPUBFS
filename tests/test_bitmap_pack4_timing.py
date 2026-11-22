import unittest
from bitmap_pack4_timing import measured_layer_window

class MeasuredLayerWindowTest(unittest.TestCase):
    def test_inclusive_depths_start_after_root(self):
        self.assertEqual(measured_layer_window([11, 22, 33, 44], 5, 2, 3), 55)
        self.assertEqual(measured_layer_window([11, 22, 33, 44], 5, 1, 4), 110)

    def test_rejects_missing_timings_or_unfinished_depth(self):
        with self.assertRaises(ValueError):
            measured_layer_window([11, 22], 5, 1, 2)
        for first, last in ((0, 1), (3, 2), (1, 5)):
            with self.assertRaises(ValueError):
                measured_layer_window([11, 22, 33, 44], 5, first, last)

if __name__ == '__main__':
    unittest.main()
