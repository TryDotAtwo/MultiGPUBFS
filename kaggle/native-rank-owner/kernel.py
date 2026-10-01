"""Exact-source native rank owner gate; honest one/two-GPU scope."""
import importlib.util
from pathlib import Path
import subprocess

SOURCE = "f29ae949b9051a9cc7a273d9f2bb48ed7c8a6b3e"
repo = Path("/tmp/mgbfs-native-rank-owner-source")
subprocess.run(["git", "clone", "-q", "https://github.com/TryDotAtwo/MultiGPUBFS.git", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", SOURCE], check=True)
actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert actual == SOURCE
spec = importlib.util.spec_from_file_location("gate", repo / "kaggle/lsa-bfs-gate/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SOURCE = actual
gate.MODE = "nccl_window_processes"
gate.main()
