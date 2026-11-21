"""CPU-only contract checks for the eight-GPU benchmark harness.

These stubs test the command/receipt plumbing, not CUDA or BFS correctness.
"""
import contextlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_segmented_eight.py"
observed = []

class GraphDefinition:
    @staticmethod
    def permutation(generators, initial):
        return {"generators": generators, "initial": initial}

@contextlib.contextmanager
def workers():
    yield

def fake_run_graph(graph, output, **kw):
    observed.append((graph, kw))
    assert kw["devices"] == list(range(8))
    assert kw["autotune"] is False
    assert kw["shards"] == 8
    output.mkdir(parents=True)
    (output / "states.json").write_text(json.dumps({
        "current": [[2, 1, 0]], "previous_small": [[1, 2, 0], [2, 0, 1]]
    }))
    return {
        "layer_sizes": [1, 2, 2, 1],
        "layer_seconds": [.1, .1, .1],
        "segment_memory_by_rank": [{"segment_strategy": "atomic"} for _ in range(8)],
        "status": "COMPLETE",
        "reason": None,
        "bfs_seconds": .3,
        "setup_seconds": .01,
        "launch_wall_seconds": .4,
        "source_retries": 0,
        "plan": {"batch": kw["_batch"]},
        "runtime_identity": {"fixture": "CPU stub, not a GPU test"},
    }

def fake_subprocess(command, **kw):
    if command[0] == "nvidia-smi":
        value = "0, 12\n" if "--query-gpu=index,memory.used" in command else "stub GPUs\n"
        return subprocess.CompletedProcess(command, 0, value, "")
    if command[0] == "git":
        if "--others" in command or "--binary" in command:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(command, 0, "stub-commit\n", "")
    raise AssertionError(f"unexpected command: {command}")

def execute(root, batch):
    native = root / "mgbfs"
    library = root / "libmgbfs_cuda.so"
    native.write_bytes(b"stub executable")
    library.write_bytes(b"stub library")
    output = root / f"case-{batch}"
    modules = {
        "multigpubfs": types.ModuleType("multigpubfs"),
        "multigpubfs.launch": types.ModuleType("multigpubfs.launch"),
        "multigpubfs.generic_session": types.ModuleType("multigpubfs.generic_session"),
    }
    modules["multigpubfs"].GraphDefinition = GraphDefinition
    modules["multigpubfs.launch"].run_graph = fake_run_graph
    modules["multigpubfs.generic_session"].persistent_graph_workers = workers
    env = {"MGBFS_EXECUTABLE": str(native), "MGBFS_BENCH_CUDA_LIBRARY": str(library),
           "MGBFS_BENCH_REPO": str(root)}
    argv = [str(SCRIPT), "--backend", "permutation_bitmap", "--strategy", "atomic",
            "--n", "3", "--batch", str(batch), "--base", str(output)]
    with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, env), \
         mock.patch.object(sys, "argv", argv), mock.patch.object(subprocess, "run", fake_subprocess):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    case = output / "S3-permutation_bitmap-atomic-1"
    initial = json.loads((case / "invocation.json").read_text())
    result = json.loads((case / "summary.json").read_text())
    assert initial["batch"] == batch
    assert result["plan"]["batch"] == batch
    assert result["checks"]["final_two_layers_full_exact_sets_no_duplicates"]
    assert json.loads((case / "sampled_peak_mib.json").read_text())["samples"] >= 1

def invalid_batch(root):
    modules = {
        "multigpubfs": types.ModuleType("multigpubfs"),
        "multigpubfs.launch": types.ModuleType("multigpubfs.launch"),
        "multigpubfs.generic_session": types.ModuleType("multigpubfs.generic_session"),
    }
    modules["multigpubfs"].GraphDefinition = GraphDefinition
    modules["multigpubfs.launch"].run_graph = fake_run_graph
    modules["multigpubfs.generic_session"].persistent_graph_workers = workers
    argv = [str(SCRIPT), "--backend", "permutation_bitmap", "--strategy", "atomic",
            "--batch", "0", "--base", str(root / "bad")]
    with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "argv", argv):
        try:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("zero batch was accepted")
    assert not (root / "bad").exists()

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    execute(root, 8192)
    execute(root, 32768)
    invalid_batch(root)
assert [kwargs["_batch"] for _, kwargs in observed] == [8192, 32768]
print("PASS CPU_STUB benchmark argument and receipts; no GPU execution")
