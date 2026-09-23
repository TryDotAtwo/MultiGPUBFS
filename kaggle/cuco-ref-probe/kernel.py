"""Focused cuCO dynamic-shard-ref build and GPU contract gate."""
import importlib.util
import os
from pathlib import Path
import urllib.request

SOURCE_COMMIT = "62bec6155edae372b244309b2a044bbb13f226a6"

if __name__ == "__main__":
    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "180")
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
