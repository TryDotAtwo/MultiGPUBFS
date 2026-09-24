"""Five-repeat archive-verified S10 LSA versus host-sized NCCL screen."""
import importlib.util
import os
from pathlib import Path
import urllib.request

SOURCE_COMMIT = "6c74077988114da59043181bd3e49c6b984fe5c5"

if __name__ == "__main__":
    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "180")
    script = Path("/tmp/mgbfs-lsa-paired-gate.py")
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE_COMMIT}/kaggle/lsa-bfs-gate/kernel.py",
        script,
    )
    spec = importlib.util.spec_from_file_location("mgbfs_lsa_paired", script)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SOURCE = SOURCE_COMMIT
    gate.MODE = "benchmark"
    gate.main()
