import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lrx", ROOT / "scripts/verify_lrx_layers.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LayerComparison(unittest.TestCase):
    def rows(self):
        return [dict(rank=i, group="s3", status="COMPLETE", local_layer_sizes=layer)
                for i, layer in enumerate([[1,1,0], [0,2,2]])]

    def test_combines_each_rank_and_preserves_exact_depths(self):
        result = MODULE.verify_layers(self.rows()[::-1], [1,3,2], 3, 2)
        self.assertEqual(result["total_states"], 6)
        self.assertEqual(result["diameter"], 2)
        self.assertEqual(result["scope"], "LAYER_COUNTS_ONLY")

    def test_rejects_same_total_in_wrong_layers(self):
        rows = self.rows()
        rows[1]["local_layer_sizes"] = [0,1,3]
        with self.assertRaises(ValueError):
            MODULE.verify_layers(rows, [1,3,2], 3, 2)

    def test_rejects_missing_duplicate_partial_and_invalid_counts(self):
        variants = [self.rows()[:1], [self.rows()[0]] * 2]
        for field, value in [("status", "INCOMPLETE"), ("group", "s4"),
                             ("local_layer_sizes", [0,2]),
                             ("local_layer_sizes", [0,-1,5]),
                             ("local_layer_sizes", [False,2,2]),
                             ("local_layer_sizes", [0,2.0,2])]:
            rows = copy.deepcopy(self.rows())
            rows[1][field] = value
            variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError):
                MODULE.verify_layers(rows, [1,3,2], 3, 2)

    def test_reference_is_full_thirteen_factorial(self):
        ref = json.loads((ROOT / "data/reference/lrx13-layers.json").read_text())
        self.assertEqual(sum(ref["layers"]), 6227020800)
        self.assertEqual(len(ref["layers"]), 79)
        self.assertEqual(max(ref["layers"]), 369741101)


if __name__ == "__main__":
    unittest.main()
