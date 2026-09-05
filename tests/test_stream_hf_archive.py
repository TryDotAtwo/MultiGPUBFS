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
    def test_consumer_metrics_survive_truncated_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = LocalStagingSink(folder, 8, 2)
            reader = ArchiveStream('r1', 'fixture', 0, sink)
            with self.assertRaisesRegex(ValueError, 'ARCHIVE_TRUNCATED'):
                reader.consume(io.BytesIO(complete_archive()[:-1]))
            metrics = getattr(reader, 'timings', {})
            self.assertEqual(metrics.get('records'), 2)
            self.assertEqual(metrics.get('record_frames'), 1)
            for key in ('read_seconds', 'checksum_seconds', 'arrow_seconds', 'sink_seconds'):
                self.assertGreaterEqual(metrics.get(key, -1), 0)

    def test_preuploaded_shards_commit_in_bounded_batches(self):
        class Api:
            def __init__(self):
                self.batches = []

            def preupload_lfs_files(self, **kwargs):
                for operation in kwargs['additions']:
                    operation.path_or_fileobj = b''

            def create_commit(self, **kwargs):
                self.batches.append(len(kwargs['operations']))

            def upload_file(self, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as folder:
            api = Api()
            sink = HubStagingSink(Path(folder) / 'slots', 1, 258,
                                  'TryDotAtwo/results', 'run-r1', api, 0, 32768)
            for ordinal in range(257):
                sink.add({'run_id': 'r1', 'group_id': 'fixture', 'config_digest': '00'*32,
                          'rank': 0, 'depth': 0, 'rank_ordinal': ordinal,
                          'state': b'x', 'hash128_le': bytes(16)})
            self.assertEqual(api.batches, [])
            sink.complete({'status': 'COMPLETE'})
            self.assertEqual(api.batches, [256, 1])
            self.assertGreater(getattr(sink, 'encode_seconds', 0), 0)

    def test_non_lfs_preupload_cannot_recycle_slot_or_commit(self):
        class Api:
            def preupload_lfs_files(self, **kwargs):
                pass  # Server classified as regular Git: bytes still needed.

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'slots'
            sink = HubStagingSink(root, 1, 2, 'TryDotAtwo/results', 'run-r1', Api(), 0, 32768)
            with self.assertRaisesRegex(RuntimeError, 'PARQUET_PREUPLOAD_NOT_RELEASED'):
                ArchiveStream('r1', 'fixture', 0, sink).consume(io.BytesIO(complete_archive()))
            self.assertFalse((root / 'rank-00000-stream-commit.json').exists())

    def test_failed_batch_commit_does_not_publish_complete_rank(self):
        class Api:
            def preupload_lfs_files(self, **kwargs):
                for operation in kwargs['additions']:
                    operation.path_or_fileobj = b''

            def create_commit(self, **kwargs):
                raise OSError('commit rejected')

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'slots'
            sink = HubStagingSink(root, 1, 2, 'TryDotAtwo/results', 'run-r1', Api(), 0, 32768)
            with self.assertRaisesRegex(OSError, 'commit rejected'):
                ArchiveStream('r1', 'fixture', 0, sink).consume(io.BytesIO(complete_archive()))
            self.assertFalse((root / 'rank-00000-stream-commit.json').exists())

    def test_corrupt_payload_is_rejected_before_any_batch_or_commit(self):
        class Sink:
            def add_batch(self, _table):
                raise AssertionError("corrupt payload escaped validation")

            def complete(self, _result):
                raise AssertionError("corrupt archive committed")

        # Both planes must be covered by the incremental checksum.
        for offset in (48 + 80, 48 + 80 + 8):
            wire = bytearray(complete_archive())
            wire[offset] ^= 1
            with self.assertRaisesRegex(ValueError, "ARCHIVE_CHECKSUM"):
                ArchiveStream("r1", "fixture", 0, Sink()).consume(io.BytesIO(wire))

    def test_fragmented_pipe_preserves_checksum_and_records(self):
        class Fragmented(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 7) if size >= 0 else 7)

        with tempfile.TemporaryDirectory() as folder:
            sink = LocalStagingSink(Path(folder), rows_per_shard=2, slot_count=2)
            result = ArchiveStream("r1", "fixture", 0, sink).consume(
                Fragmented(complete_archive()))
            self.assertEqual(result["total_unique_states"], 2)
            self.assertTrue((Path(folder) / "stream-commit.json").exists())

    def test_record_frames_are_delivered_as_columnar_batches(self):
        class BatchOnlySink:
            def __init__(self):
                self.batches = []

            def add(self, _row):
                raise AssertionError("row-at-a-time archive path")

            def add_batch(self, table):
                self.batches.append(table)

            def complete(self, _result):
                pass

        sink = BatchOnlySink()
        ArchiveStream("r1", "fixture", rank=0, sink=sink).consume(
            io.BytesIO(complete_archive())
        )
        self.assertEqual(len(sink.batches), 1)
        self.assertEqual(sink.batches[0].num_rows, 2)
        self.assertEqual(sink.batches[0].column("rank_ordinal").to_pylist(), [0, 1])
        self.assertEqual(
            sink.batches[0].column("state").to_pylist(),
            [bytes([1, 0, 0, 1]), bytes([1, 1, 0, 1])],
        )

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
                self.preuploads = []
                self.decoded = []
                self.unique_encodings = []
                self.hash_codecs = []

            def preupload_lfs_files(self, **kwargs):
                for operation in kwargs['additions']:
                    payload = operation.path_or_fileobj.read()
                    parquet = pq.ParquetFile(io.BytesIO(payload))
                    self.hash_codecs.append(parquet.metadata.row_group(0).column(7).compression)
                    self.decoded.extend(parquet.read().to_pylist())
                    self.unique_encodings.extend(
                        parquet.metadata.row_group(0).column(i).encodings for i in (5, 6, 7))
                    self.assert_payload = payload[:4] == b'PAR1' and payload[-4:] == b'PAR1'
                    self.assert_buffered = isinstance(operation.path_or_fileobj, io.BufferedIOBase)
                    operation.path_or_fileobj = b''
                self.preuploads.append(kwargs)

            def create_commit(self, **kwargs):
                self.calls.append(kwargs)

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
            self.assertEqual(len(api.preuploads), 2)
            self.assertEqual(len(api.calls), 2)
            self.assertEqual(len(api.calls[0]['operations']), 2)
            self.assertTrue(all(op.path_or_fileobj == b'' for op in api.calls[0]['operations']))
            self.assertTrue(all(call["revision"] == "run-r1" for call in api.calls))
            self.assertTrue(all(op.path_in_repo.startswith('pending/run-r1/') for op in api.calls[0]['operations']))
            self.assertTrue(api.assert_payload)
            self.assertTrue(api.assert_buffered)
            rows = sorted(api.decoded, key=lambda row: row['rank_ordinal'])
            self.assertEqual([row['state'] for row in rows], [bytes([1, 0, 0, 1]), bytes([1, 1, 0, 1])])
            self.assertEqual([row['hash128_le'] for row in rows], [bytes(range(16)), bytes(range(16, 32))])
            self.assertTrue(all('RLE_DICTIONARY' not in enc and 'PLAIN_DICTIONARY' not in enc
                                for enc in api.unique_encodings))
            self.assertEqual(api.hash_codecs, ['UNCOMPRESSED', 'UNCOMPRESSED'])
            self.assertFalse(list((Path(folder) / "slots").glob("slot-*.parquet")))

    def test_hub_upload_failure_never_emits_rank_commit(self):
        class Api:
            def __init__(self):
                self.calls = 0

            def preupload_lfs_files(self, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise OSError("injected")
                for operation in kwargs['additions']:
                    operation.path_or_fileobj = b''

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

            def preupload_lfs_files(self, **kwargs):
                self.release.wait(timeout=2)
                for operation in kwargs['additions']:
                    operation.path_or_fileobj = b''

            def create_commit(self, **kwargs):
                self.calls.append(kwargs)

            def upload_file(self, **kwargs):
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
