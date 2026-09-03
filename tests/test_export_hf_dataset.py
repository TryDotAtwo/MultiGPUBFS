import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def frame(chain, sequence, kind, depth, count, payload):
    header = bytearray(80)
    header[:8] = b"MGBFSFR1"
    struct.pack_into("<QQQQQ", header, 8, kind, depth, count, len(payload), sequence)
    header[48:] = chain
    digest = hashlib.sha256(header + payload).digest()
    return bytes(header) + payload + digest, digest


def archive(path):
    header = b"MGBFSAR1" + struct.pack("<Q", 4) + bytes(range(32))
    chain = hashlib.sha256(header).digest()
    state_payload = bytes([1, 0, 0, 1, 1, 1, 0, 1])
    hash_payload = bytes(range(16)) + bytes(range(16, 32))
    first, chain = frame(chain, 0, 1, 0, 2, state_payload + hash_payload)
    layer, chain = frame(chain, 1, 2, 0, 2, b"")
    done, _ = frame(chain, 2, 3, 1, 2, b"")
    path.write_bytes(header + first + layer + done)


class ExportDataset(unittest.TestCase):
    def test_exports_every_unique_state_and_replay_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "rank0.mgbfsar1"
            archive(source)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps({"status": "COMPLETE", "group_id": "fixture", "config_digest": "ab" * 32}),
                encoding="utf-8",
            )
            output = root / "dataset"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/export_hf_dataset.py"), "--run-id", "r1",
                 "--summary", str(summary), "--archive", f"0={source}", "--output", str(output)],
                check=True,
            )
            states = pq.read_table(output / "states" / "rank-00000-part-00000.parquet")
            self.assertEqual(states.num_rows, 2)
            self.assertEqual(states.column("depth").to_pylist(), [0, 0])
            self.assertEqual(states.column("state").to_pylist(), [bytes([1, 0, 0, 1]), bytes([1, 1, 0, 1])])
            self.assertEqual(states.column("hash128_le").to_pylist(), [bytes(range(16)), bytes(range(16, 32))])
            layers = pq.read_table(output / "layers" / "part-00000.parquet").to_pylist()
            self.assertEqual(layers[0]["unique_states"], 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_unique_states"], 2)
            self.assertEqual(manifest["max_depth"], 0)
            self.assertEqual(len(manifest["files"][0]["sha256"]), 64)

    def test_rejects_corrupt_or_incomplete_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "bad.bin"
            archive(source)
            source.write_bytes(source.read_bytes()[:-1])
            summary = root / "summary.json"
            summary.write_text('{"status":"COMPLETE"}', encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/export_hf_dataset.py"), "--run-id", "r1",
                 "--summary", str(summary), "--archive", f"0={source}", "--output", str(root / "out")],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
