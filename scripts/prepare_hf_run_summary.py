#!/usr/bin/env python3
"""Normalize one completed native benchmark into the catalog RunSummaryV1."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def group_metadata(spec):
    if re.fullmatch(r"s[2-9][0-9]*", spec):
        degree = int(spec[1:])
        return {
            "group_id": f"S_{degree} cycle-inverse-transposition matrix Cayley graph over F2",
            "state_bytes": degree * degree,
            "modulus": 2,
            "base_generators": 3,
        }
    match = re.fullmatch(r"u4-([2-9][0-9]*)", spec)
    if match:
        modulus = int(match.group(1))
        if modulus > 256:
            raise ValueError("GROUP_MODULUS")
        return {
            "group_id": f"U_4(Z/{modulus}Z) matrix Cayley graph",
            "state_bytes": 16,
            "modulus": modulus,
            "base_generators": 6,
        }
    raise ValueError("GROUP_SPEC")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id):
        raise ValueError("RUN_ID_PATH")
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    layers = raw.get("layer_sizes", [])
    times = raw.get("per_depth_seconds", [])
    if raw.get("status") != "COMPLETE" or raw.get("unique_states") != sum(layers) or len(times) != len(layers):
        raise ValueError("INCOMPLETE_OR_COUNT_MISMATCH")
    meta = group_metadata(str(raw.get("group", "")))
    macro_moves = int(raw.get("macro_move_count", 0))
    if macro_moves <= 0:
        raise ValueError("MACRO_MOVE_COUNT")
    layer_capacity = int(raw.get("layer_capacity", raw["unique_states"]))
    future_capacity = int(raw.get("future_capacity_per_depth", layer_capacity))
    description = (
        f"macro-native-v1;group={raw['group']};batch={raw['batch']};k={raw['macro_depth']};"
        f"layer={layer_capacity};future={future_capacity};pre={str(raw['prededup']).lower()};seed=20260828"
    )
    digest = hashlib.sha256(description.encode()).hexdigest()
    layer_metrics = {
        str(depth): {"unique_states": count, "generated_candidates": count * macro_moves,
                     "wall_ms": times[depth] * 1000}
        for depth, count in enumerate(layers)
    }
    topology = {"world_size": 1, "logical_owner_to_rank": [0], "shards_per_rank": 64,
                "buckets_per_shard": 256}
    summary = {
        "schema": "RunSummaryV1", "status": "COMPLETE", "run_id": args.run_id,
        "group_id": meta["group_id"], "group_spec": raw["group"], "config_digest": digest,
        "source_commit": args.source_commit, "hardware": args.hardware,
        "hash": {"algorithm": "GEMM_U8_P32X4_V1", "seed_u128": 20260828, "byte_order": "little-endian"},
        "config": {
            "profile": "DENSE", "owner_backend": "CUB_SORT_MERGE", "pre_dedup": raw["prededup"],
            "macro_depth": raw["macro_depth"], "macro_move_count": macro_moves, "batch": raw["batch"],
            "layer_capacity": layer_capacity, "future_capacity_per_depth": future_capacity,
            "state_layout": "canonical row-major u8", "state_bytes": meta["state_bytes"],
            "base_generators": meta["base_generators"], "modulus": meta["modulus"], "topology": topology,
        },
        "topology": topology, "search_complete_seconds": raw["search_complete_seconds"],
        "durable_run_commit_seconds": raw["durable_run_commit_seconds"],
        "total_unique_states": raw["unique_states"], "layer_sizes": layers,
        "layer_sha256": raw.get("layer_sha256", []),
        "memory": {"cuda_context_used_bytes": raw.get("cuda_context_used_bytes"),
                   "cuda_allocated_used_bytes": raw.get("cuda_allocated_used_bytes"),
                   "pinned_bytes": raw.get("pinned_bytes"), "disk_reserved_bytes": raw.get("disk_reserved_bytes")},
        "layers": layer_metrics, "raw_benchmark": raw,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
