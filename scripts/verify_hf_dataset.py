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
import numpy as np

PRIME=4_294_967_291

def gemm_hash_contract(width,seed):
    shake=hashlib.shake_256(b"MGBFS/GEMM_U8_P32X4/V1\0"+width.to_bytes(4,"little")+seed.to_bytes(16,"little"))
    raw=shake.digest((width*4+4)*4+4096);values=[]
    for at in range(0,len(raw),4):
        value=int.from_bytes(raw[at:at+4],"little")
        if value<PRIME:values.append(value)
        if len(values)==width*4+4:break
    if len(values)!=width*4+4:raise ValueError("HASH_XOF_EXHAUSTED")
    return np.asarray(values[:width*4],dtype=np.uint64).reshape(width,4),np.asarray(values[-4:],dtype=np.uint64)


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
    run_rows=pq.read_table(args.dataset / "runs").to_pylist()
    if len(run_rows)!=1:raise ValueError("RUN_INVENTORY")
    run_summary=json.loads(run_rows[0]["summary_json"]);hash_spec=run_summary.get("hash",{})
    validate_hash=hash_spec.get("algorithm")=="GEMM_U8_P32X4_V1"
    if validate_hash:
        coefficients,offsets=gemm_hash_contract(width,int(hash_spec["seed_u128"]))
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
            for batch in source.iter_batches(columns=["depth", "state", "hash128_le"], batch_size=131072):
                states=batch.column(1).to_pylist()
                if validate_hash and states:
                    matrix=np.frombuffer(b"".join(states),dtype=np.uint8).reshape(len(states),width).astype(np.uint64)
                    expected_hashes=(matrix@coefficients+offsets)%PRIME
                    actual=np.frombuffer(b"".join(batch.column(2).to_pylist()),dtype="<u4").reshape(len(states),4)
                    if not np.array_equal(expected_hashes,actual):raise ValueError("HASH_STATE_MISMATCH")
                for depth, state in zip(batch.column(0).to_pylist(), states):
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
        "layers": len(counts), "max_depth": max(counts), "hash_state_pairs_verified":validate_hash, "manifest_sha256": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()).hexdigest(),
    }
    (args.dataset / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
