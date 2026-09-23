"""Two-T4 full BFS gate for the pre-owner retirement failure vote."""
import importlib.util
import os
from pathlib import Path
import urllib.request

SOURCE_COMMIT = "b71c16b"

if __name__ == "__main__":
    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "180")
    script = Path("/tmp/mgbfs-library-gate.py")
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE_COMMIT}/kaggle/library-owner/kernel.py",
        script,
    )
    spec = importlib.util.spec_from_file_location("mgbfs_library_gate", script)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SOURCE_COMMIT = SOURCE_COMMIT
    gate.FULL_BFS_GATE = True
    gate.LOAD_SCREEN = False
    gate.SANITIZER_TOOLS = ()
    gate.main()
