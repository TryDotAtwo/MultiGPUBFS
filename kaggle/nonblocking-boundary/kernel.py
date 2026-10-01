"""Pinned two-T4 full-state/failure gate for the nonblocking transport."""
import importlib.util
import os
from pathlib import Path
import subprocess

SOURCE = "64670c2"
os.environ["NCCL_DEBUG"] = "INFO"
repo = Path("/tmp/mgbfs-nonblocking-boundary")
subprocess.run(["git", "clone", "-q", "https://github.com/TryDotAtwo/MultiGPUBFS.git", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", SOURCE], check=True)
actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert actual.startswith(SOURCE)
binary = repo / "nccl-wrapper-test"
subprocess.run(["g++", "-std=c++17", "-Itests/nccl_stubs", "tests/nccl_transport_failure.cpp",
                "-o", str(binary)], cwd=repo, check=True)
subprocess.run([str(binary)], cwd=repo, check=True)
spec = importlib.util.spec_from_file_location("gate", repo / "kaggle/host-sized-nccl-gate/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
# The runner pins the separately verified runtime. Do not relabel diagnostic
# harness changes as runtime changes or override its LSA/debugger admission.
gate.main()
