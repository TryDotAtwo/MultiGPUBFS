"""Paired S10 archive benchmark with a 96 MiB fixed CUCO_RANK pool."""
import importlib.util
import os
from pathlib import Path
import urllib.request

SOURCE_COMMIT = "fb8b9f4669199d0b95c766f051b4ad6d1d7e4892"

if __name__ == "__main__":
    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "180")
    os.environ["MGBFS_INSTALL_TIMEOUT_SEC"] = "2700"
    script = Path("/tmp/mgbfs-library-paired-pool96.py")
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE_COMMIT}/kaggle/library-owner/kernel.py",
        script,
    )
    spec = importlib.util.spec_from_file_location("mgbfs_library_paired_pool96", script)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SOURCE_COMMIT = SOURCE_COMMIT
    gate.FULL_BFS_GATE = True
    gate.LOAD_SCREEN = True
    gate.PROFILE_SCREEN = False
    gate.SCREEN_OWNERS = ("CUCO_RANK",)
    gate.SCREEN_WORLDS = (2,)
    gate.SCREEN_REPEATS = 5
    gate.SCREEN_POOL_BYTES_BY_WORLD = {2: 96 << 20}
    gate.NATIVE_COMPARISON = True
    gate.SANITIZER_TOOLS = ()
    gate.main()
