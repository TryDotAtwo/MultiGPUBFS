"""Exact-source native rank owner gate; honest one/two-GPU scope."""
import importlib.util
from pathlib import Path
import subprocess

SOURCE = "546643f16f34d05e993fe344412513e0320c85f1"
repo = Path("/tmp/mgbfs-native-rank-owner-source")
subprocess.run(["git", "clone", "-q", "https://github.com/TryDotAtwo/MultiGPUBFS.git", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", SOURCE], check=True)
actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert actual == SOURCE
spec = importlib.util.spec_from_file_location("gate", repo / "kaggle/lsa-bfs-gate/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SOURCE = actual
gate.MODE = "native_rank_gate"
gate.main()
