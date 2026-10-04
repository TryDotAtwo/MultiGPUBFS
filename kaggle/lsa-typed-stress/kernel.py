"""Launch the existing pinned full-BFS gate; no second runtime implementation."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile

SOURCE = "a136204ab4a377e0655216d9a9e11640c51b5526"


def main():
    checkout = Path(tempfile.mkdtemp(prefix="mgbfs-stress-bootstrap-"))
    subprocess.run(["git", "init", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1",
                    "https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE:
        raise RuntimeError("SOURCE_COMMIT_MISMATCH")
    spec = importlib.util.spec_from_file_location("pinned_bfs_gate",
        checkout / "kaggle/lsa-bfs-gate/kernel.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SOURCE = SOURCE
    gate.MODE = "typed_sanitizer_version_gate"
    gate.main()


if __name__ == "__main__":
    main()
