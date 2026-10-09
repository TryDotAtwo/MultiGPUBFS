"""Regression for actual Nsight CUDA API aliases; no GPU/SQLite import required."""
import ast
from pathlib import Path
import unittest

class NsightLaunchAliases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/validation/kaggle-dense-packet-timeline-evidence.py"
        module = ast.parse(path.read_text())
        fields = [n.value for n in ast.walk(module)
                  if isinstance(n, ast.keyword) and n.arg == "kernel_launches"]
        if len(fields) != 1:
            raise AssertionError("Exactly one integrated paired launch metric required")
        cls.expression = compile(ast.Expression(fields[0]), str(path), "eval")

    def metric(self, inventory):
        return eval(self.expression, {"records": [{"full_capture_host_api_inventory": inventory}]})

    def test_actual_versioned_runtime_api_is_counted(self):
        self.assertEqual(self.metric([{"api": "cudaLaunchKernel_v7000", "calls": 18317}]), 18317)

    def test_driver_inventory_is_not_double_counted(self):
        self.assertEqual(self.metric([
            {"api": "cudaLaunchKernel", "calls": 2},
            {"api": "cudaLaunchKernelExC", "calls": 3},
            {"api": "cudaLaunchKernelExC_v11060", "calls": 7},
            {"api": "cuLaunchKernelEx", "calls": 634},
        ]), 12)

    def test_unrelated_or_invalid_alias_is_not_counted(self):
        self.assertEqual(self.metric([
            {"api": "cudaLaunchKernelFake", "calls": 100},
            {"api": "cudaLaunchKernel_vinvalid", "calls": 100},
            {"api": "cudaMemcpy_v3020", "calls": 100},
        ]), 0)

if __name__ == "__main__":
    unittest.main()
