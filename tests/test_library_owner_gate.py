import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "library_gate", Path(__file__).resolve().parents[1] / "kaggle/library-owner/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class IsolationTests(unittest.TestCase):
    def test_kaggle_python_injection_does_not_enter_library_environment(self):
        inherited = {"PYTHONPATH": "/kaggle/lib", "PYTHONHOME": "/old/python",
                     "PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "GPU-fixture"}
        actual = gate.isolated_environment(inherited)
        self.assertEqual(actual, {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "GPU-fixture"})
        self.assertEqual(inherited["PYTHONPATH"], "/kaggle/lib")


if __name__ == "__main__":
    unittest.main()
