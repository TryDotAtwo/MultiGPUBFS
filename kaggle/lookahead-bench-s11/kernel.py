"""Pinned S11 profile comparison; submit only after lookahead correctness gate."""
import importlib.util
from pathlib import Path
import tempfile
import urllib.request

SOURCE = "013ed5c979f4225db273e0015fa9ed72fd230c90"


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-lookahead-panel-", dir="/tmp"))
    path = root / "panel.py"
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE}/"
        "kaggle/distributed-bench-s11/kernel.py", path)
    spec = importlib.util.spec_from_file_location("lookahead_panel", path)
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    panel.SOURCE = SOURCE
    panel.main()


if __name__ == "__main__":
    main()
