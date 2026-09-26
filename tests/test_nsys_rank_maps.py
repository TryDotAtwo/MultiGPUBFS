"""Process-map attribution for address-only Nsight CUDA callchains."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/nsys_rank_maps.py"


class RankMapAttributionTests(unittest.TestCase):
    def test_executable_address_selects_rank_and_uses_file_offset(self):
        # Catches applying an address from rank 1 to rank 0's ASLR base, or
        # forgetting the mapping's nonzero ELF file offset.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = root / "calls.json"
            rank0 = root / "rank-0.maps"
            rank1 = root / "rank-1.maps"
            output = root / "resolved.json"
            calls.write_text(json.dumps({"rows": [{
                "api": "cudaStreamSynchronize", "calls": 7,
                "api_duration_ns": 7000,
                "symbols": ["0x7010", "0x4123"],
            }]}))
            rank0.write_text(
                "1000-2000 r-xp 00001000 00:01 1 /tmp/build/mgbfs\n"
                "7000-8000 r-xp 00000000 00:01 2 /tmp/libcudart.so\n"
            )
            rank1.write_text(
                "4000-5000 r-xp 00002000 00:01 1 /tmp/build/mgbfs\n"
                "7000-8000 r-xp 00000000 00:01 2 /tmp/libcudart.so\n"
            )
            completed = subprocess.run([
                sys.executable, str(SCRIPT), str(calls), str(output),
                "--map", f"0:{rank0}", "--map", f"1:{rank1}",
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            row = json.loads(output.read_text())["rows"][0]
            self.assertEqual(row["rank"], 1)
            self.assertEqual(row["project_frames"], [{
                "module": "/tmp/build/mgbfs", "offset": "0x2123",
            }])


if __name__ == "__main__":
    unittest.main()
