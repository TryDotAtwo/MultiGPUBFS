"""Pinned full two-T4 gate, preserving the previous sanitizer notebook outputs."""
import importlib.util
from pathlib import Path
import tempfile
import urllib.request

SOURCE = "013ed5c979f4225db273e0015fa9ed72fd230c90"


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-lookahead-gate-", dir="/tmp"))
    path = root / "gate.py"
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE}/"
        "kaggle/distributed-sanitizer/kernel.py", path)
    spec = importlib.util.spec_from_file_location("lookahead_gate", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.SOURCE = SOURCE
    require_fixture = gate.require_fixture

    def require_current_fixture(output, expected):
        # Only the distributed_archive suite changed: one new full-state
        # lookahead test. Preserve strict counts for every other fixture.
        require_fixture(output, 13 if expected == 12 else expected)

    gate.require_fixture = require_current_fixture
    gate.main()


if __name__ == "__main__":
    main()
