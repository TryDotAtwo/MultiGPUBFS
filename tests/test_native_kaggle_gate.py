import importlib.util
import pathlib
import unittest
import threading
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("native_gate", pathlib.Path(__file__).parents[1] / "kaggle/native-primitives/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

class GateTests(unittest.TestCase):
    def test_concurrent_launch_preserves_all_eighty_device_test_tool_combinations(self):
        names = ("generate", "hash", "route", "owner", "pipeline", "materialize", "dense_device", "ping_pong")
        executables = {name: "exe/" + name for name in names}
        gpus = [{"index": i, "uuid": f"GPU-{i}"} for i in range(2)]
        calls, results = [], []
        lock = threading.Lock()
        def fake_run(command, **kwargs):
            with lock:
                calls.append((command, kwargs))
        def record(row):
            with lock:
                results.append(row)
        with patch.object(gate, "run", side_effect=fake_run):
            gate.run_gpu_suites(gpus, lambda gpu: gate.execute_gpu_suite(gpu, executables, ".", {}, ".", record))
        expected = {(f"GPU-{i}", name, tool) for i in range(2) for name in names for tool in ("plain",) + gate.SANITIZERS}
        self.assertEqual(len(calls), 80)
        self.assertEqual(len(results), 80)
        self.assertEqual({(r["gpu"], r["test"], r["tool"]) for r in results}, expected)
        for command, kwargs in calls:
            gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            self.assertTrue(kwargs["name"].startswith("gpu" + gpu + "-"))
            self.assertIn("--test-threads=1", command)

    def test_independent_gpu_suites_overlap_and_propagate_failure(self):
        barrier = threading.Barrier(2, timeout=5)
        def worker(gpu):
            barrier.wait()
            return gpu["index"]
        gpus = [{"index": 0}, {"index": 1}]
        self.assertEqual(gate.run_gpu_suites(gpus, worker), [0, 1])
        def fails(gpu):
            barrier.wait()
            if gpu["index"] == 1:
                raise RuntimeError("GPU1 failed")
            return 0
        with self.assertRaisesRegex(RuntimeError, "GPU1 failed"):
            gate.run_gpu_suites(gpus, fails)

    def test_racecheck_uses_small_variant_fixture_not_full_sweep(self):
        names = {
            "failure_with_both_slots_in_flight_is_sticky_and_drains_on_drop",
            "full_u4_pipelined_sweep",
            "generation_variants_preserve_full_layers",
            "generation_variants_small_feedback",
            "reused_slots_and_partial_tails_preserve_every_layer",
        }
        skips, fixture = gate.ping_pong_selection("racecheck")
        self.assertEqual(names - set(skips), names - {
            "full_u4_pipelined_sweep", "generation_variants_preserve_full_layers"})
        self.assertIn("m2-m3", fixture)
        self.assertEqual(gate.ping_pong_selection("plain"), ((), "all"))
        for tool in ("memcheck", "initcheck", "synccheck"):
            self.assertEqual(gate.ping_pong_selection(tool)[0], ("full_u4_pipelined_sweep",))

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
