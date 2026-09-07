import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "sanitizer_gate", Path(__file__).parents[1] / "kaggle/distributed-sanitizer/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FixtureGateTests(unittest.TestCase):
    def test_exact_completed_fixture_count_not_a_substring(self):
        gate.require_fixture("test result: ok. 2 passed; 0 failed; 0 ignored;", 2)
        for output in (
            "test result: ok. 12 passed; 0 failed; 0 ignored;",
            "test result: FAILED. 2 passed; 1 failed; 0 ignored;",
            "running 2 tests",
            "test result: ok. 1 passed; 0 failed; 0 ignored;",
            "test result: ok. 2 passed; 0 failed; 0 ignored;\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored;",
        ):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                gate.require_fixture(output, 2)


if __name__ == "__main__":
    unittest.main()
