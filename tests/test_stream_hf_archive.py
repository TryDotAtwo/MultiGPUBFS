import hashlib
import io
import struct
import tempfile
import threading
import unittest
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stream_hf_archive import ArchiveStream, HubStagingSink, LocalStagingSink


def frame(chain, sequence, kind, depth, count, payload=b""):
    header = bytearray(80)
    header[:8] = b"MGBFSFR1"
    struct.pack_into("<QQQQQ", header, 8, kind, depth, count, len(payload), sequence)
    header[48:] = chain
    digest = hashlib.sha256(header + payload).digest()
    return bytes(header) + payload + digest, digest


def complete_archive(width=4):
    header = b"MGBFSAR1" + struct.pack("<Q", width) + bytes(range(32))
    chain = hashlib.sha256(header).digest()
    records = bytes([1, 0, 0, 1, 1, 1, 0, 1]) + bytes(range(32))
    a, chain = frame(chain, 0, 1, 0, 2, records)
    b, chain = frame(chain, 1, 2, 0, 2)
    c, _ = frame(chain, 2, 3, 1, 2)
    return header + a + b + c


class StreamArchive(unittest.TestCase):
    def test_complete_stream_emits_fixed_schema_shards_and_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = LocalStagingSink(Path(folder), rows_per_shard=1, slot_count=2)
            result = ArchiveStream("r1", "fixture", rank=0, sink=sink).consume(
                io.BytesIO(complete_archive())
            )
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["total_unique_states"], 2)
            files = sorted(Path(folder).glob("states/*.parquet"))
            self.assertEqual(len(files), 2)
            table = pq.read_table(files)
            self.assertEqual(table.column("state").to_pylist(), [bytes([1, 0, 0, 1]), bytes([1, 1, 0, 1])])
            self.assertTrue((Path(folder) / "stream-commit.json").is_file())

    def test_truncation_never_writes_complete_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = LocalStagingSink(Path(folder), rows_per_shard=2, slot_count=2)
            with self.assertRaisesRegex(ValueError, "ARCHIVE_TRUNCATED"):
                ArchiveStream("r1", "fixture", rank=0, sink=sink).consume(
                    io.BytesIO(complete_archive()[:-1])
                )
            self.assertFalse((Path(folder) / "stream-commit.json").exists())

    def test_slot_capacity_is_fixed_and_reuse_requires_upload_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = LocalStagingSink(Path(folder), rows_per_shard=1, slot_count=1, auto_receipt=False)
            stream = ArchiveStream("r1", "fixture", rank=0, sink=sink)
            with self.assertRaisesRegex(RuntimeError, "PARQUET_SLOT_RING_FATAL"):
                stream.consume(io.BytesIO(complete_archive()))
            self.assertEqual(sink.peak_live_slots, 1)
            self.assertFalse((Path(folder) / "stream-commit.json").exists())

    def test_hub_sink_uploads_only_to_staging_branch_and_reclaims_slots(self):
        class Api:
            def __init__(self):
                self.calls = []

            def upload_file(self, **kwargs):
                source = kwargs["path_or_fileobj"]
                if hasattr(source, "read"):
                    self.assert_buffered = isinstance(source, io.BufferedIOBase)
                    payload = source.read()
                    self.assert_payload = payload[:4] == b"PAR1" and payload[-4:] == b"PAR1"
                self.calls.append(kwargs)

        with tempfile.TemporaryDirectory() as folder:
            api = Api()
            sink = HubStagingSink(
                Path(folder) / "slots", rows_per_shard=1, slot_count=2,
                repo_id="TryDotAtwo/results", branch="run-r1", api=api,
                rank=0, max_slot_bytes=1_000_000,
            )
            ArchiveStream("r1", "fixture", rank=0, sink=sink).consume(io.BytesIO(complete_archive()))
            self.assertEqual(len(api.calls), 3)
            self.assertTrue(all(call["revision"] == "run-r1" for call in api.calls))
            self.assertTrue(all(call["path_in_repo"].startswith("pending/run-r1/") for call in api.calls))
            self.assertTrue(api.assert_payload)
            self.assertTrue(api.assert_buffered)
            self.assertFalse(list((Path(folder) / "slots").glob("slot-*.parquet")))

    def test_hub_upload_failure_never_emits_rank_commit(self):
        class Api:
            def __init__(self):
                self.calls = 0

            def upload_file(self, **_kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise OSError("injected")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "slots"
            sink = HubStagingSink(
                root, rows_per_shard=1, slot_count=2, repo_id="TryDotAtwo/results",
                branch="run-r1", api=Api(), rank=0, max_slot_bytes=1_000_000,
            )
            with self.assertRaisesRegex(RuntimeError, "PARQUET_UPLOAD_FAILED"):
                ArchiveStream("r1", "fixture", rank=0, sink=sink).consume(io.BytesIO(complete_archive()))
            self.assertFalse((root / "rank-00000-stream-commit.json").exists())

    def test_tail_may_wait_for_receipt_only_after_search_is_complete(self):
        class Api:
            def __init__(self):
                self.release = threading.Event()
                self.calls = []

            def upload_file(self, **kwargs):
                if hasattr(kwargs["path_or_fileobj"], "read"):
                    self.release.wait(timeout=2)
                self.calls.append(kwargs)

        with tempfile.TemporaryDirectory() as folder:
            api = Api()
            sink = HubStagingSink(
                Path(folder) / "slots", rows_per_shard=3, slot_count=2,
                repo_id="TryDotAtwo/results", branch="run-r1", api=api,
                rank=0, max_slot_bytes=1_000_000,
            )
            # A full shard occupies a slot. The terminal partial shard is
            # allowed to wait for a receipt because generation has ended.
            for ordinal in range(7):
                sink.add({
                    "run_id": "r1", "group_id": "fixture", "config_digest": "00" * 32,
                    "rank": 0, "depth": 0, "rank_ordinal": ordinal,
                    "state": bytes([ordinal]), "hash128_le": bytes(16),
                })
            api.release.set()
            manifest = sink.complete({"status": "COMPLETE"})
            self.assertEqual(len(manifest["files"]), 3)
            self.assertEqual(manifest["peak_live_slots"], 2)


if __name__ == "__main__":
    unittest.main()
