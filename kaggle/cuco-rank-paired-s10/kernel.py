"""Paired unprofiled S10 archive benchmark: CUCO_RANK versus native CUB."""
import importlib.util
import os
from pathlib import Path
import urllib.request

SOURCE_COMMIT = "4b68552b0c0c865862d896a2ca1913043110e575"

if __name__ == "__main__":
    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "180")
    script = Path("/tmp/mgbfs-library-paired.py")
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE_COMMIT}/kaggle/library-owner/kernel.py",
        script,
    )
    spec = importlib.util.spec_from_file_location("mgbfs_library_paired", script)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SOURCE_COMMIT = SOURCE_COMMIT
    gate.FULL_BFS_GATE = True
    gate.LOAD_SCREEN = True
    gate.PROFILE_SCREEN = False
    gate.SCREEN_OWNERS = ("CUCO_RANK",)
    gate.SCREEN_WORLDS = (2,)
    gate.SCREEN_REPEATS = 5
    gate.SCREEN_POOL_BYTES_BY_WORLD = {2: 512 << 20}
    gate.NATIVE_COMPARISON = True
    gate.SANITIZER_TOOLS = ()
    gate.main()
