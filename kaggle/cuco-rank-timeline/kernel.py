"""Diagnostic Nsight timeline for the real two-rank CUCO_RANK BFS."""
import importlib.util
import os
from pathlib import Path
import urllib.request

SOURCE_COMMIT = "d95ef218d321cd35b601c7db6440ec22038e966d"

if __name__ == "__main__":
    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "180")
    script = Path("/tmp/mgbfs-library-timeline.py")
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE_COMMIT}/kaggle/library-owner/kernel.py",
        script,
    )
    spec = importlib.util.spec_from_file_location("mgbfs_library_timeline", script)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SOURCE_COMMIT = SOURCE_COMMIT
    gate.FULL_BFS_GATE = True
    gate.LOAD_SCREEN = True
    gate.PROFILE_SCREEN = True
    gate.SCREEN_OWNERS = ("CUCO_RANK",)
    gate.SCREEN_WORLDS = (2,)
    gate.SCREEN_REPEATS = 1
    gate.SCREEN_POOL_BYTES_BY_WORLD = {2: 512 << 20}
    gate.NATIVE_COMPARISON = False
    gate.SANITIZER_TOOLS = ()
    gate.main()
