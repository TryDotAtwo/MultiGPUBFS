"""Focused cuCO dynamic-shard-ref build and GPU contract gate."""
import importlib.util
from pathlib import Path
import urllib.request

SOURCE_COMMIT = "9a6b85e317ea35e947d1c28dc482fe2f57171ca3"

if __name__ == "__main__":
    script = Path("/tmp/mgbfs-library-gate.py")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
        f"{SOURCE_COMMIT}/kaggle/library-owner/kernel.py", script)
    spec = importlib.util.spec_from_file_location("mgbfs_library_gate", script)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SOURCE_COMMIT = SOURCE_COMMIT
    gate.FULL_BFS_GATE = False
    gate.LOAD_SCREEN = False
    gate.SANITIZER_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
    gate.main()
