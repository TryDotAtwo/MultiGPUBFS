import importlib.util
from pathlib import Path
import unittest
import tempfile

spec = importlib.util.spec_from_file_location(
    "library_gate", Path(__file__).resolve().parents[1] / "kaggle/library-owner/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class IsolationTests(unittest.TestCase):
    def test_lib64_export_and_transitive_package_are_discoverable(self):
        with tempfile.TemporaryDirectory() as root:
            site = Path(root)
            for relative in ("libcudf/lib64/cmake/cudf/cudf-config.cmake",
                             "librmm/lib64/rapids/cmake/cccl/cccl-config.cmake"):
                config = site / relative
                config.parent.mkdir(parents=True)
                config.write_text("# fixture", encoding="utf-8")
                self.assertIn(str(config.parent), gate.cmake_prefixes(site))

    def test_kaggle_python_injection_does_not_enter_library_environment(self):
        inherited = {"PYTHONPATH": "/kaggle/lib", "PYTHONHOME": "/old/python",
                     "PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "GPU-fixture"}
        actual = gate.isolated_environment(inherited)
        self.assertEqual(actual, {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "GPU-fixture"})
        self.assertEqual(inherited["PYTHONPATH"], "/kaggle/lib")


if __name__ == "__main__":
    unittest.main()
