"""CPU contracts for optional VRAM sampling and its benchmark consumer."""
import contextlib
import json
import os
import runpy
import sys
import tempfile
import threading
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from bitmap_pack4_sampling import sample_pattern, collect_vram_samples, vram_summary


class SamplingTest(unittest.TestCase):
    def test_selection_validation_and_missing_samples(self):
        self.assertEqual(sample_pattern(None, 3), [False, False, False])
        self.assertEqual(sample_pattern("0,1,0", 3), [False, True, False])
        for invalid in ("1", "0,yes,0", "0,,0"):
            with self.assertRaises(ValueError):
                sample_pattern(invalid, 3)
        self.assertEqual(vram_summary(False, [], [0, 1])["nvml_samples"], 0)
        self.assertIsNone(vram_summary(False, [], [0, 1])["nvml_max_mib_by_gpu"])
        self.assertIsNone(vram_summary(True, [], [0, 1])["nvml_max_mib_by_gpu"])

    def test_sampler_is_started_only_when_requested(self):
        started = threading.Event()
        def sampler(stop, samples):
            samples.append({"monotonic": 1.25, "gpus": {0: 73}})
            started.set()
        with collect_vram_samples(False, sampler) as samples:
            self.assertEqual(samples, [])
        self.assertFalse(started.is_set())
        with collect_vram_samples(True, sampler) as samples:
            self.assertTrue(started.wait(2))
        self.assertEqual(vram_summary(True, samples, [0, 1])["nvml_max_mib_by_gpu"],
                         {"0": 73, "1": None})

    def test_result_collector_accepts_a_run_without_samples(self):
        collector = Path(__file__).parents[1]/"scripts"/"collect_pack4_8gpu_results.py"
        code = compile(collector.read_text(), str(collector), "exec")
        real_path = Path
        with tempfile.TemporaryDirectory() as tmp:
            workspace = real_path(tmp)
            repo = workspace/"mgbfs-pack4"
            evidence = workspace/"eight-gpu-evidence"
            results = workspace/"mgbfs-pack4-evidence"
            (repo/"docs/acceptance").mkdir(parents=True)
            evidence.mkdir()
            for name in ("topology.txt", "gpus.csv", "dependency-sha256.txt",
                         "s14-s16-mahonian-calculation.jsonl",
                         "cyclic-rank-capacity.jsonl"):
                (evidence/name).write_text("")
            (evidence/"mahonian-window-oracle.json").write_text("{}")
            (evidence/"cyclic-rank-capacity-receipt-check.json").write_text("{}")
            source = results/"benchmark-s12-max_capacity-8gpu-30623"
            source.mkdir(parents=True)
            (source/"summary.json").write_text(json.dumps([{
                "n":12, "run":0, "codec":"pack4-v1", "nvml_enabled":False,
                "nvml_sample_interval_seconds":None, "nvml_max_mib_by_gpu":None,
                "nvml_samples":0}]))
            def mapped_path(path):
                text = str(path)
                if text.startswith("/workspace/"):
                    return workspace/text[len("/workspace/"):]
                return real_path(path)
            fake_pathlib = types.ModuleType("pathlib")
            fake_pathlib.Path = mapped_path
            with mock.patch.dict(sys.modules, {"pathlib":fake_pathlib}):
                exec(code, {"__name__":"__main__"})
            artifact = json.loads((repo/"docs/acceptance/2026-11-22-pack4-8gpu-results.json").read_text())
            self.assertEqual(len(artifact["benchmarks"]), 1)
            self.assertEqual(artifact["benchmarks"][0]["nvml_samples"], 0)
            self.assertIsNone(artifact["benchmarks"][0]["nvml_max_mib_by_gpu"])

    def test_benchmark_consumer_preserves_layers_and_sample_association(self):
        sample_called = threading.Event()
        queries = []
        reported = [1, 2, 2, 1]
        class Graph:
            @staticmethod
            def permutation(*args):
                return args
        worker = types.SimpleNamespace(process=types.SimpleNamespace(pid=123))
        @contextlib.contextmanager
        def workers():
            yield types.SimpleNamespace(workers=[worker, worker])
        def fake_query(args, **kw):
            if "--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in args:
                return ""
            self.assertIn("--query-gpu=index,memory.used", args)
            queries.append(tuple(args))
            sample_called.set()
            return "0, 73\n1, 83\n"
        def fake_run(graph, output, **kw):
            output = Path(output)
            output.mkdir(parents=True)
            if output.name.startswith("warm-"):
                return {"status": "COMPLETE"}
            run = int(output.name.split("-")[1])
            if run == 0:
                self.assertFalse(sample_called.is_set())
            else:
                self.assertTrue(sample_called.wait(2), "enabled sampler did not run")
            (output/"states.json").write_text(json.dumps({
                "current": [[2,1,0]], "previous_small": [[1,2,0],[2,0,1]]
            }))
            codec = output.name.rsplit("-", 1)[1]
            rows = 4
            rank = {"rank": 0, "banks": 2, "actual_queue_capacity": 8,
                    "owner_lanes": 4, "queue_batch_ceiling": 4,
                    "segment_rows_per_bank_by_shard": [[rows]],
                    "segment_state_allocated_bytes": rows*2*(8 if codec=="1" else 3),
                    "segment_counter_allocated_bytes": 1, "storage_peak_requested_bytes": 1,
                    "bitmap_bytes": 1, "peak_live_rows": 1, "queues_allocated_bytes": 1,
                    "reserve_bytes": 1, "resident_codec": "pack4-v1" if codec=="1" else "unpacked-soa"}
            return {"status":"COMPLETE", "layer_sizes":reported, "layer_seconds":[.1,.2,.3],
                    "segment_memory_by_rank":[rank], "bfs_seconds":1, "launch_wall_seconds":1,
                    "setup_seconds":0, "source_retries":0}
        m = types.ModuleType("multigpubfs")
        m.GraphDefinition = Graph
        launch = types.ModuleType("multigpubfs.launch")
        launch.run_graph = fake_run
        session = types.ModuleType("multigpubfs.generic_session")
        session.persistent_graph_workers = workers
        modules = {"multigpubfs":m, "multigpubfs.launch":launch, "multigpubfs.generic_session":session}
        script = Path(__file__).with_name("bitmap_pack4_benchmark_gpu.py")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, modules), \
             mock.patch.dict(os.environ, {"MGBFS_EXECUTABLE":"fake", "MGBFS_TEST_DEVICE_COUNT":"2",
                    "MGBFS_BENCH_SAMPLE_PATTERN":"0,1", "MGBFS_PACK4_EVIDENCE_ROOT":tmp}), \
             mock.patch.object(sys, "argv", [str(script),"3","balanced","0,1"]), \
             mock.patch("subprocess.check_output", side_effect=fake_query):
            runpy.run_path(str(script), run_name="__main__")
            root = next(Path(tmp).iterdir())
            records = json.loads((root/"summary.json").read_text())
            self.assertEqual([r["nvml_enabled"] for r in records], [False, True])
            self.assertEqual(records[0]["nvml_samples"], 0)
            self.assertIsNone(records[0]["nvml_max_mib_by_gpu"])
            self.assertEqual(json.loads((root/"samples-0-0.json").read_text()), [])
            self.assertEqual(records[1]["nvml_max_mib_by_gpu"], {"0":73,"1":83})
            self.assertTrue(all(r["all_layer_counts_mahonian"] and r["terminal_full_sets"] for r in records))
            self.assertTrue(all(abs(r["window_seconds"]-.6)<1e-9 for r in records))
            self.assertEqual(len(queries), 1)

if __name__ == "__main__":
    unittest.main()
