"""Pure selection-contract tests; AST extraction avoids executing launcher ingress."""
import ast
from pathlib import Path
import sys
import unittest

source_path = Path(sys.argv.pop(1))
tree = ast.parse(source_path.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "focused_validation_plan"]
if len(functions) != 1:
    raise AssertionError("FOCUSED_VALIDATION_SELECTOR_MISSING")
namespace = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)
plan = namespace["focused_validation_plan"]

class SelectionContract(unittest.TestCase):
    def test_baseline_inventory_unchanged(self):
        p = plan("baseline")
        self.assertEqual((len(p["oracle_pairs"]) * 4, len(p["capture_pairs"]) * 4, len(p["fault_pairs"]), len(p["sanitizer_pairs"]) * 4), (20, 0, 4, 8))
        self.assertEqual(p["expected_panels"], 32)

    def test_extended_captures_every_profile_owner(self):
        p = plan("owner_capture_and_missing_modes")
        expected = {(profile, owner) for profile in ("DENSE", "HASH_FIRST") for owner in ("CUB_SORT_MERGE", "CUCO_RANK", "BMMA_BUCKET")}
        self.assertEqual(set(p["capture_pairs"]), expected)
        self.assertEqual(len(p["capture_pairs"]), len(expected))
        self.assertEqual(p["oracle_pairs"], [("DENSE", "BMMA_BUCKET")])

    def test_missing_fault_and_sanitizer_paths(self):
        p = plan("owner_capture_and_missing_modes")
        self.assertEqual(p["fault_pairs"], [("DENSE", "BMMA_BUCKET"), ("HASH_FIRST", "CUCO_RANK")])
        self.assertEqual(set(p["sanitizer_pairs"]), {("DENSE", "BMMA_BUCKET"), ("HASH_FIRST", "CUB_SORT_MERGE"), ("HASH_FIRST", "CUCO_RANK"), ("HASH_FIRST", "BMMA_BUCKET")})
        self.assertEqual(p["expected_panels"], 46)

    def test_integrated_profiles_inventory(self):
        p = plan("lsa_batch_profiles")
        expected = {(profile, owner) for profile in ("DENSE", "HASH_FIRST") for owner in ("CUB_SORT_MERGE", "CUCO_RANK", "BMMA_BUCKET")}
        for key in ("batch_capture_pairs", "fault_pairs", "sanitizer_pairs", "timeline_pairs"):
            self.assertEqual(set(p[key]), expected)
            self.assertEqual(len(p[key]), 6)
        self.assertEqual(p["oracle_pairs"], [])
        self.assertEqual(p["capture_pairs"], [])
        self.assertEqual(p["expected_panels"], 60)

    def test_batch_capture_labels_are_unique(self):
        labels = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "label" for target in node.targets)
                  and any(isinstance(item, ast.Constant) and item.value == "batch-capture-" for item in ast.walk(node.value))]
        self.assertEqual(len(labels), 1)
        observed = set()
        for profile in ("DENSE", "HASH_FIRST"):
            for owner in ("CUB_SORT_MERGE", "CUCO_RANK", "BMMA_BUCKET"):
                for prededup in (False, True):
                    for rank_map in ([0, 1], [1, 0]):
                        value = eval(compile(ast.Expression(labels[0]), str(source_path), "eval"), {}, locals())
                        observed.add(value)
        self.assertEqual(len(observed), 24)

    def test_unknown_suite_rejected(self):
        with self.assertRaises(ValueError):
            plan("fallback")

if __name__ == "__main__":
    unittest.main()
