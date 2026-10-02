"""Exact-source native rank owner gate; honest one/two-GPU scope."""
import importlib.util
from pathlib import Path
import subprocess

SOURCE = "12dee48d1e31e4c06d3e96f63ecb215fb307fb1a"
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
gate.NCCL_VARIANT = "minimum_arch_guard"
gate.main()
