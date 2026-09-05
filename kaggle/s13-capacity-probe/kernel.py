"""S13 compact capacity diagnostic; no archive or catalog artifact is produced."""
import runpy
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "f2482c4db18c4c6a2596a7344f857d667b442b3a"

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
                      CAPACITIES_PER_RANK=[178_000_000])
        probe["main"]()

if __name__ == "__main__":
    main()
