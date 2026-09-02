import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location("native_gate", pathlib.Path(__file__).parents[1] / "kaggle/native-primitives/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

class GateTests(unittest.TestCase):
    def test_requires_two_distinct_t4_devices_and_reserve(self):
        good = "0, Tesla T4, GPU-a, 15360, 14000\n1, Tesla T4, GPU-b, 15360, 14000\n"
        self.assertEqual([r["index"] for r in gate.validate_gpus(good)], [0, 1])
        for invalid in [good.splitlines()[0], good.replace("GPU-b", "GPU-a"), good.replace("Tesla T4", "RTX 3070"), good.replace("14000", "512"), good.replace("1, Tesla", "0, Tesla")]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                gate.validate_gpus(invalid)

    def test_only_full_immutable_commits_are_accepted(self):
        commit = "93036b247833f2b7b84d8fe32416f074934fbb9f"
        self.assertEqual(gate.validate_commit(commit), commit)
        for bad in ["main", "codex/native-matrix-bfs", "93036b2", "x" * 40, "-" * 40]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gate.validate_commit(bad)

if __name__ == "__main__":
    unittest.main()
