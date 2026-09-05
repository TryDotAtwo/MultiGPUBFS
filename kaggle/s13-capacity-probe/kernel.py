"""S13 compact capacity diagnostic; no archive or catalog artifact is produced."""
import runpy
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "0b2acdd5b9e4a6e27af13cc4f0c8c79190f23dcf"

def main():
    with tempfile.TemporaryDirectory(prefix="s13-harness-", dir="/tmp") as directory:
        path = Path(directory) / "probe.py"
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{SOURCE}/kaggle/s12-capacity-probe/kernel.py",
            path,
        )
        probe = runpy.run_path(str(path))
        config = probe["main"].__globals__
        config.update(SOURCE=SOURCE, GROUP="s13", CARDINALITY=6_227_020_800,
                      CAPACITIES_PER_RANK=[220_000_000], RUN_TIMEOUT=7200)
        probe["main"]()

if __name__ == "__main__":
    main()
