#!/usr/bin/env python3
"""Bounded-memory integrity and exact state-uniqueness gate for HF export."""
import argparse
import hashlib
import heapq
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def records(path, width):
    with path.open("rb") as source:
        while True:
            value = source.read(width)
            if not value:
                return
            if len(value) != width:
                raise ValueError("SORT_CHUNK_TRUNCATED")
            yield value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sort-memory-records", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.sort_memory_records <= 0:
        raise ValueError("SORT_MEMORY_RECORDS")
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "MGBFS_HF_DATASET_V1":
        raise ValueError("DATASET_SCHEMA")
    for item in manifest["files"]:
        path = args.dataset / item["path"]
        if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("PARQUET_MANIFEST_MISMATCH")
    layer_table = pq.read_table(args.dataset / "layers")
    expected = {row["depth"]: row["unique_states"] for row in layer_table.to_pylist()}
    widths = {row["state_bytes"] for row in layer_table.to_pylist()}
    if len(widths) != 1:
        raise ValueError("STATE_WIDTH_MISMATCH")
    width = widths.pop()
    counts = defaultdict(int)
    total = 0
    with tempfile.TemporaryDirectory(prefix="mgbfs-unique-") as folder:
        folder = Path(folder); chunks = []; pending = []

        def flush():
            if not pending:
                return
            pending.sort()
            path = folder / f"chunk-{len(chunks):06d}.bin"
            with path.open("wb") as target:
                target.writelines(pending)
            chunks.append(path); pending.clear()

        for parquet in sorted((args.dataset / "states").glob("*.parquet")):
            source = pq.ParquetFile(parquet)
            for batch in source.iter_batches(columns=["depth", "state"], batch_size=131072):
                for depth, state in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                    if len(state) != width:
                        raise ValueError("STATE_WIDTH")
                    counts[depth] += 1; total += 1; pending.append(state)
                    if len(pending) >= args.sort_memory_records:
                        flush()
        flush()
        previous = None; unique = 0
        streams = [records(path, width) for path in chunks]
        for state in heapq.merge(*streams):
            if state == previous:
                raise ValueError("DUPLICATE_STATE")
            previous = state; unique += 1
    if dict(counts) != expected or total != manifest["total_unique_states"] or max(counts) != manifest["max_depth"]:
        raise ValueError("DATASET_COUNTS")
    verification = {
        "schema": "MGBFS_HF_VERIFY_V1", "status": "PASS", "unique_states": unique,
        "layers": len(counts), "max_depth": max(counts), "manifest_sha256": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()).hexdigest(),
    }
    (args.dataset / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
