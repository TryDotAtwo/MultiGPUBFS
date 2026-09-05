#!/usr/bin/env python3
"""Incrementally validate MGBFSAR1 and emit bounded Parquet state shards.

The sink contract deliberately separates a closed local shard from its upload
receipt.  Production upload sinks may recycle a staging slot only after the
remote pre-upload completed; exhaustion is fatal rather than backpressure.
"""
import argparse
import hashlib
import io
import json
import os
import re
import struct
import sys
from concurrent.futures import wait, FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


STATE_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("group_id", pa.string()),
    ("config_digest", pa.string()),
    ("rank", pa.uint32()),
    ("depth", pa.uint32()),
    ("rank_ordinal", pa.uint64()),
    ("state", pa.binary()),
    ("hash128_le", pa.binary(16)),
])


def _read_exact(source, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = source.read(remaining)
        if not chunk:
            raise ValueError("ARCHIVE_TRUNCATED")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class LocalStagingSink:
    """Test/reference sink with an explicit fixed staging-slot ledger."""

    def __init__(self, root, rows_per_shard, slot_count, auto_receipt=True):
        if rows_per_shard <= 0 or slot_count <= 0:
            raise ValueError("PARQUET_RING_SHAPE")
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError("PARQUET_OUTPUT_NOT_EMPTY")
        (self.root / "states").mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.slot_count = slot_count
        self.auto_receipt = auto_receipt
        self.rows = []
        self.part = 0
        self.live_slots = set()
        self.peak_live_slots = 0
        self.files = []

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_shard:
            self.flush()

    def add_batch(self, table):
        # Reference/test sink; production uses HubStagingSink's columnar path.
        for batch in table.to_batches(max_chunksize=self.rows_per_shard):
            for row in batch.to_pylist():
                self.add(row)

    def flush(self):
        if not self.rows:
            return
        available = next((slot for slot in range(self.slot_count) if slot not in self.live_slots), None)
        if available is None:
            raise RuntimeError("PARQUET_SLOT_RING_FATAL")
        self.live_slots.add(available)
        self.peak_live_slots = max(self.peak_live_slots, len(self.live_slots))
        path = self.root / "states" / f"part-{self.part:08d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(self.rows, STATE_SCHEMA),
            path,
            compression="zstd",
            row_group_size=min(self.rows_per_shard, 131072),
        )
        self.files.append(path)
        self.rows.clear()
        self.part += 1
        if self.auto_receipt:
            self.receipt(available)

    def receipt(self, slot):
        if slot not in self.live_slots:
            raise RuntimeError("PARQUET_DUPLICATE_RECEIPT")
        self.live_slots.remove(slot)

    def complete(self, result):
        self.flush()
        if self.live_slots:
            raise RuntimeError("PARQUET_UPLOADS_INCOMPLETE")
        manifest = dict(result)
        manifest["schema"] = "MGBFS_HF_STREAM_COMMIT_V1"
        manifest["files"] = [
            {
                "path": path.relative_to(self.root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in self.files
        ]
        target = self.root / "stream-commit.json"
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


class HubStagingSink:
    """Bounded local Parquet slots with asynchronous uploads to a run branch.

    This sink never publishes to the dataset's default branch.  A separate
    global finalizer must validate every rank commit and promote the immutable
    objects with server-side copies.
    """

    def __init__(
        self,
        root,
        rows_per_shard,
        slot_count,
        repo_id,
        branch,
        api,
        rank,
        max_slot_bytes,
    ):
        if rows_per_shard <= 0 or slot_count < 2 or max_slot_bytes <= 0:
            raise ValueError("PARQUET_RING_SHAPE")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.rows_per_shard = rows_per_shard
        self.slot_count = slot_count
        self.repo_id = repo_id
        self.branch = branch
        self.api = api
        self.rank = rank
        self.max_slot_bytes = max_slot_bytes
        self.tables = []
        self.buffered_rows = 0
        self.part = 0
        self.free_slots = list(range(slot_count))
        try:
            self.slot_buffers = [bytearray(max_slot_bytes) for _ in range(slot_count)]
        except MemoryError as error:
            raise RuntimeError("PARQUET_SLOT_ALLOCATION_FATAL") from error
        self.inflight = {}
        self.executor = ThreadPoolExecutor(max_workers=slot_count, thread_name_prefix="mgbfs-hf")
        self.files = []
        self.peak_live_slots = 0

    def add(self, row):
        self.add_batch(pa.Table.from_pylist([row], STATE_SCHEMA))

    def add_batch(self, table):
        if table.schema != STATE_SCHEMA:
            table = table.cast(STATE_SCHEMA)
        offset = 0
        while offset < table.num_rows:
            room = self.rows_per_shard - self.buffered_rows
            take = min(room, table.num_rows - offset)
            self.tables.append(table.slice(offset, take))
            self.buffered_rows += take
            offset += take
            if self.buffered_rows == self.rows_per_shard:
                self.flush()

    def _upload(self, slot, size, remote_path):
        # huggingface_hub intentionally accepts BufferedIOBase, not a bare
        # RawIOBase. The wrapper remains bounded and reads the live slot
        # without making a second bytes-sized copy.
        with io.BufferedReader(_MemoryViewReader(self.slot_buffers[slot], size)) as reader:
            self.api.upload_file(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.branch,
                path_or_fileobj=reader,
                path_in_repo=remote_path,
                commit_message=f"Stage {remote_path}",
            )
        return slot

    def _reap(self):
        for slot, future in list(self.inflight.items()):
            if not future.done():
                continue
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(f"PARQUET_UPLOAD_FAILED: {error}") from error
            self.free_slots.append(slot)
            del self.inflight[slot]

    def _drain_one(self):
        """Wait for one receipt only during terminal archive finalization."""
        if not self.inflight:
            raise RuntimeError("PARQUET_SLOT_RING_FATAL")
        wait(tuple(self.inflight.values()), return_when=FIRST_COMPLETED)
        self._reap()

    def flush(self, final=False):
        if not self.tables:
            return
        self._reap()
        if final and not self.free_slots:
            self._drain_one()
        if not self.free_slots:
            raise RuntimeError("PARQUET_SLOT_RING_FATAL")
        slot = self.free_slots.pop()
        writer = pa.FixedSizeBufferWriter(pa.py_buffer(self.slot_buffers[slot]))
        try:
            table = self.tables[0] if len(self.tables) == 1 else pa.concat_tables(self.tables)
            pq.write_table(
                table,
                writer,
                compression="zstd",
                row_group_size=min(table.num_rows, 131072),
            )
            size = writer.tell()
        except Exception as error:
            self.free_slots.append(slot)
            raise RuntimeError(f"PARQUET_SLOT_BYTES_FATAL_{self.max_slot_bytes}: {error}") from error
        finally:
            writer.close()
        remote_path = (
            f"pending/{self.branch}/states/"
            f"rank-{self.rank:05d}-part-{self.part:08d}.parquet"
        )
        self.files.append({
            "path": remote_path,
            "bytes": size,
            "sha256": hashlib.sha256(memoryview(self.slot_buffers[slot])[:size]).hexdigest(),
        })
        self.inflight[slot] = self.executor.submit(self._upload, slot, size, remote_path)
        self.peak_live_slots = max(self.peak_live_slots, len(self.inflight))
        self.tables.clear()
        self.buffered_rows = 0
        self.part += 1

    def complete(self, result):
        # Search is finished here, so waiting for one upload receipt is a
        # durability wait, not runtime backpressure.
        self.flush(final=True)
        for future in list(self.inflight.values()):
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(f"PARQUET_UPLOAD_FAILED: {error}") from error
        self._reap()
        self.executor.shutdown(wait=True)
        manifest = dict(result)
        manifest.update(
            schema="MGBFS_HF_STREAM_COMMIT_V1",
            branch=self.branch,
            files=self.files,
            peak_live_slots=self.peak_live_slots,
            slot_count=self.slot_count,
            max_slot_bytes=self.max_slot_bytes,
        )
        commit_path = self.root / f"rank-{self.rank:05d}-stream-commit.json"
        commit_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.api.upload_file(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.branch,
            path_or_fileobj=str(commit_path),
            path_in_repo=f"pending/{self.branch}/rank-{self.rank:05d}-stream-commit.json",
            commit_message=f"Complete staged rank {self.rank}",
        )
        return manifest


class _MemoryViewReader(io.RawIOBase):
    """Seekable, zero-copy reader over one immutable live staging slot."""

    def __init__(self, buffer, size):
        self.view = memoryview(buffer)[:size]
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = len(self.view) + offset
        else:
            raise ValueError("SEEK_WHENCE")
        if target < 0:
            raise ValueError("SEEK_NEGATIVE")
        self.position = min(target, len(self.view))
        return self.position

    def readinto(self, target):
        count = min(len(target), len(self.view) - self.position)
        if count <= 0:
            return 0
        target[:count] = self.view[self.position:self.position + count]
        self.position += count
        return count


class ArchiveStream:
    def __init__(self, run_id, group_id, rank, sink):
        self.run_id = run_id
        self.group_id = group_id
        self.rank = rank
        self.sink = sink

    def consume(self, source):
        header = _read_exact(source, 48)
        if header[:8] != b"MGBFSAR1":
            raise ValueError("ARCHIVE_HEADER")
        width = struct.unpack_from("<Q", header, 8)[0]
        if not 0 < width <= 33025:
            raise ValueError("ARCHIVE_WIDTH")
        config_digest = header[16:48].hex()
        chain = hashlib.sha256(header).digest()
        sequence = depth = layer_count = total = ordinal = 0
        layer_counts = []
        while True:
            frame = _read_exact(source, 80)
            kind, frame_depth, count, size, frame_sequence = struct.unpack_from("<QQQQQ", frame, 8)
            if (
                frame[:8] != b"MGBFSFR1"
                or frame[48:] != chain
                or frame_sequence != sequence
                or frame_depth != depth
            ):
                raise ValueError("ARCHIVE_CHAIN")
            payload = _read_exact(source, size)
            digest = _read_exact(source, 32)
            # Keep the wire checksum unchanged without allocating/copying a
            # second complete payload for every archive frame.
            checksum = hashlib.sha256(frame)
            checksum.update(payload)
            chain = checksum.digest()
            if digest != chain:
                raise ValueError("ARCHIVE_CHECKSUM")
            if kind == 1:
                if count == 0 or count * (width + 16) != size:
                    raise ValueError("ARCHIVE_RECORD_SHAPE")
                state_bytes = count * width
                states = pa.FixedSizeBinaryArray.from_buffers(
                    pa.binary(width), count, [None, pa.py_buffer(memoryview(payload)[:state_bytes])]
                )
                hashes = pa.FixedSizeBinaryArray.from_buffers(
                    pa.binary(16), count, [None, pa.py_buffer(memoryview(payload)[state_bytes:])]
                )
                table = pa.Table.from_arrays(
                    [
                        pa.repeat(pa.scalar(self.run_id), count),
                        pa.repeat(pa.scalar(self.group_id), count),
                        pa.repeat(pa.scalar(config_digest), count),
                        pa.repeat(pa.scalar(self.rank, pa.uint32()), count),
                        pa.repeat(pa.scalar(depth, pa.uint32()), count),
                        pa.array(range(ordinal, ordinal + count), type=pa.uint64()),
                        states,
                        hashes,
                    ],
                    schema=STATE_SCHEMA,
                )
                self.sink.add_batch(table)
                ordinal += count
                layer_count += count
            elif kind == 2:
                if size or count != layer_count:
                    raise ValueError("ARCHIVE_LAYER_COUNT")
                layer_counts.append(layer_count)
                total += layer_count
                layer_count = 0
                depth += 1
            elif kind == 3:
                if size or layer_count or count != total or depth == 0:
                    raise ValueError("ARCHIVE_RUN_COUNT")
                result = {
                    "status": "COMPLETE",
                    "run_id": self.run_id,
                    "group_id": self.group_id,
                    "config_digest": config_digest,
                    "rank": self.rank,
                    "state_bytes": width,
                    "total_unique_states": total,
                    "max_depth": depth - 1,
                    "layer_counts": layer_counts,
                    "archive_chain_sha256": chain.hex(),
                }
                self.sink.complete(result)
                return result
            else:
                raise ValueError("ARCHIVE_FRAME_KIND")
            sequence += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--input", default="-", help="MGBFSAR1 stream path or '-' for stdin")
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--slot-count", type=int, default=3)
    parser.add_argument("--max-slot-bytes", type=int, required=True)
    parser.add_argument("--create-branch", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id):
        raise ValueError("RUN_ID_PATH")
    if args.rank < 0:
        raise ValueError("RANK")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN_MISSING")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if args.create_branch:
        api.create_branch(
            repo_id=args.repo_id,
            repo_type="dataset",
            branch=args.branch,
            exist_ok=False,
        )
    sink = HubStagingSink(
        args.staging_dir,
        rows_per_shard=args.rows_per_shard,
        slot_count=args.slot_count,
        repo_id=args.repo_id,
        branch=args.branch,
        api=api,
        rank=args.rank,
        max_slot_bytes=args.max_slot_bytes,
    )
    if args.input == "-":
        result = ArchiveStream(args.run_id, args.group_id, args.rank, sink).consume(sys.stdin.buffer)
    else:
        with open(args.input, "rb", buffering=0) as source:
            result = ArchiveStream(args.run_id, args.group_id, args.rank, sink).consume(source)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
