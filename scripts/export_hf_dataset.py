#!/usr/bin/env python3
"""Stream complete MGBFS archives into a replayable Parquet dataset."""
import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


STATE_SCHEMA = pa.schema([
    ("run_id", pa.string()), ("group_id", pa.string()), ("config_digest", pa.string()),
    ("rank", pa.uint32()), ("depth", pa.uint32()), ("rank_ordinal", pa.uint64()),
    ("logical_owner", pa.uint32()), ("shard", pa.uint32()), ("bucket", pa.uint32()),
    ("state", pa.binary()), ("hash128_le", pa.binary(16)),
])
LAYER_SCHEMA = pa.schema([
    ("run_id", pa.string()), ("group_id", pa.string()), ("config_digest", pa.string()),
    ("depth", pa.uint32()), ("unique_states", pa.uint64()), ("state_bytes", pa.uint32()),
    ("archive_payload_bytes", pa.uint64()), ("content_sha256", pa.string()),
    ("generated_candidates", pa.uint64()), ("source_duplicates", pa.uint64()),
    ("visited_duplicates", pa.uint64()), ("same_depth_duplicates", pa.uint64()),
    ("routed_records", pa.uint64()), ("routed_bytes", pa.uint64()),
    ("generation_ms", pa.float64()), ("hash_ms", pa.float64()), ("sort_ms", pa.float64()),
    ("exchange_ms", pa.float64()), ("owner_ms", pa.float64()), ("materialize_ms", pa.float64()),
    ("archive_d2h_ms", pa.float64()), ("wall_ms", pa.float64()),
    ("peak_future_records", pa.uint64()), ("owner_imbalance", pa.float64()),
    ("metrics_json", pa.large_string()),
])
RUN_SCHEMA = pa.schema([
    ("run_id", pa.string()), ("group_id", pa.string()), ("config_digest", pa.string()),
    ("status", pa.string()), ("total_unique_states", pa.uint64()), ("max_depth", pa.uint32()),
    ("summary_json", pa.large_string()),
])


def u64(data, offset):
    return struct.unpack_from("<Q", data, offset)[0]


def frames(path):
    with path.open("rb") as source:
        header = source.read(48)
        if len(header) != 48 or header[:8] != b"MGBFSAR1":
            raise ValueError("ARCHIVE_HEADER")
        width = u64(header, 8)
        if not 0 < width <= 33025:
            raise ValueError("ARCHIVE_WIDTH")
        chain = hashlib.sha256(header).digest()
        sequence = depth = layer_count = total = 0
        while True:
            frame = source.read(80)
            if len(frame) != 80:
                raise ValueError("ARCHIVE_TRUNCATED")
            kind, frame_depth, count, size, frame_sequence = struct.unpack_from("<QQQQQ", frame, 8)
            if frame[:8] != b"MGBFSFR1" or frame[48:] != chain or frame_sequence != sequence or frame_depth != depth:
                raise ValueError("ARCHIVE_CHAIN")
            payload = source.read(size)
            digest = source.read(32)
            if len(payload) != size or len(digest) != 32:
                raise ValueError("ARCHIVE_TRUNCATED")
            chain = hashlib.sha256(frame + payload).digest()
            if digest != chain:
                raise ValueError("ARCHIVE_CHECKSUM")
            if kind == 1:
                if count == 0 or count * (width + 16) != size:
                    raise ValueError("ARCHIVE_RECORD_SHAPE")
                layer_count += count
                yield depth, width, count, payload
            elif kind == 2:
                if size or count != layer_count:
                    raise ValueError("ARCHIVE_LAYER_COUNT")
                total += layer_count
                layer_count = 0
                depth += 1
            elif kind == 3:
                if size or layer_count or count != total or depth == 0:
                    raise ValueError("ARCHIVE_RUN_COUNT")
                return
            else:
                raise ValueError("ARCHIVE_FRAME_KIND")
            sequence += 1


class StateShards:
    def __init__(self, root, rows_per_shard):
        self.root, self.limit = root, rows_per_shard
        self.rows, self.part, self.files = [], 0, []
        root.mkdir(parents=True, exist_ok=True)

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.limit:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        rank = self.rows[0]["rank"]
        path = self.root / f"rank-{rank:05d}-part-{self.part:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows, STATE_SCHEMA), path, compression="zstd", row_group_size=min(self.limit, 131072))
        self.files.append(path)
        self.rows.clear()
        self.part += 1


def write_table(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema), path, compression="zstd")


def location(hash128, topology):
    if not topology:
        return None, None, None
    world = int(topology["world_size"]); shards = int(topology["shards_per_rank"])
    buckets = int(topology["buckets_per_shard"]); rank_map = topology["logical_owner_to_rank"]
    if any(value <= 0 or value & (value - 1) for value in (world, shards, buckets)):
        raise ValueError("TOPOLOGY_POWER_OF_TWO")
    owner_bits, shard_bits, bucket_bits = world.bit_length() - 1, shards.bit_length() - 1, buckets.bit_length() - 1
    high = int.from_bytes(hash128[8:16], "little")
    prefix = high >> (64 - owner_bits - shard_bits - bucket_bits) if owner_bits + shard_bits + bucket_bits else 0
    bucket = prefix & (buckets - 1); shard = (prefix >> bucket_bits) & (shards - 1)
    logical_owner = prefix >> (bucket_bits + shard_bits)
    return logical_owner, shard, bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--archive", action="append", required=True, help="rank=path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.rows_per_shard <= 0 or args.output.exists():
        raise ValueError("EXPORT_OUTPUT_OR_SHARD_SIZE")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise ValueError("RUN_NOT_COMPLETE")
    group_id = str(summary.get("group_id", "unknown"))
    config_digest = str(summary.get("config_digest", ""))
    topology = summary.get("topology") or summary.get("config", {}).get("topology")
    archives = []
    for value in args.archive:
        rank, separator, path = value.partition("=")
        if not separator:
            raise ValueError("ARCHIVE_ARGUMENT")
        archives.append((int(rank), Path(path)))
    if len({rank for rank, _ in archives}) != len(archives):
        raise ValueError("DUPLICATE_RANK")
    archives.sort()
    args.output.mkdir(parents=True)
    counts, widths, payload_bytes = defaultdict(int), {}, defaultdict(int)
    digests = defaultdict(hashlib.sha256)
    state_files = []
    for rank, path in archives:
        writer = StateShards(args.output / "states", args.rows_per_shard)
        ordinal = 0
        for depth, width, count, payload in frames(path):
            widths.setdefault(depth, width)
            if widths[depth] != width:
                raise ValueError("CROSS_RANK_STATE_WIDTH")
            states_size = count * width
            for index in range(count):
                state = payload[index * width:(index + 1) * width]
                hash128 = payload[states_size + index * 16:states_size + (index + 1) * 16]
                logical_owner, shard, bucket = location(hash128, topology)
                if topology and int(topology["logical_owner_to_rank"][logical_owner]) != rank:
                    raise ValueError("ARCHIVE_OWNER_MISMATCH")
                writer.add(dict(run_id=args.run_id, group_id=group_id, config_digest=config_digest,
                                rank=rank, depth=depth, rank_ordinal=ordinal, logical_owner=logical_owner,
                                shard=shard, bucket=bucket, state=state, hash128_le=hash128))
                digests[depth].update(struct.pack("<I", rank)); digests[depth].update(state); digests[depth].update(hash128)
                ordinal += 1
            counts[depth] += count
            payload_bytes[depth] += len(payload)
        writer.flush()
        state_files.extend(writer.files)
    if not counts or sorted(counts) != list(range(max(counts) + 1)):
        raise ValueError("NONCONTIGUOUS_OR_EMPTY_DEPTHS")
    layer_metrics = summary.get("layers", {})
    numeric_metrics = ("generated_candidates", "source_duplicates", "visited_duplicates", "same_depth_duplicates",
                       "routed_records", "routed_bytes", "generation_ms", "hash_ms", "sort_ms", "exchange_ms",
                       "owner_ms", "materialize_ms", "archive_d2h_ms", "wall_ms", "peak_future_records", "owner_imbalance")
    layer_rows = [dict(run_id=args.run_id, group_id=group_id, config_digest=config_digest, depth=depth,
                       unique_states=counts[depth], state_bytes=widths[depth],
                       archive_payload_bytes=payload_bytes[depth], content_sha256=digests[depth].hexdigest(),
                       **{name: layer_metrics.get(str(depth), {}).get(name) for name in numeric_metrics},
                       metrics_json=json.dumps(layer_metrics.get(str(depth), {}), sort_keys=True, separators=(",", ":")))
                  for depth in sorted(counts)]
    write_table(args.output / "layers" / "part-00000.parquet", layer_rows, LAYER_SCHEMA)
    total = sum(counts.values()); maximum = max(counts)
    run_row = dict(run_id=args.run_id, group_id=group_id, config_digest=config_digest,
                   status="COMPLETE", total_unique_states=total, max_depth=maximum,
                   summary_json=json.dumps(summary, sort_keys=True, separators=(",", ":")))
    write_table(args.output / "runs" / "part-00000.parquet", [run_row], RUN_SCHEMA)
    all_files = state_files + [args.output / "layers" / "part-00000.parquet", args.output / "runs" / "part-00000.parquet"]
    manifest = dict(schema="MGBFS_HF_DATASET_V1", run_id=args.run_id, group_id=group_id,
                    config_digest=config_digest, total_unique_states=total, max_depth=maximum,
                    files=[dict(path=str(path.relative_to(args.output)).replace("\\", "/"), bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(all_files)])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
