#!/usr/bin/env python3
"""Validate per-rank stream receipts before server-side Hub promotion."""
import re


def combine_rank_commits(commits, expected_world):
    if expected_world <= 0 or len(commits) != expected_world:
        raise ValueError("RANK_SET")
    commits = sorted(commits, key=lambda item: item.get("rank", -1))
    if [item.get("rank") for item in commits] != list(range(expected_world)):
        raise ValueError("RANK_SET")
    first = commits[0]
    keys = ("run_id", "group_id", "config_digest", "branch", "state_bytes", "max_depth")
    if (
        first.get("schema") != "MGBFS_HF_STREAM_COMMIT_V1"
        or first.get("status") != "COMPLETE"
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(first.get("run_id", "")))
    ):
        raise ValueError("STREAM_SCHEMA")
    layer_count = first["max_depth"] + 1
    layers = [0] * layer_count
    files = []
    source_paths = set()
    destinations = set()
    total = 0
    rank_chains = []
    for item in commits:
        if item.get("schema") != first["schema"] or item.get("status") != "COMPLETE":
            raise ValueError("STREAM_SCHEMA")
        if any(item.get(key) != first.get(key) for key in keys):
            raise ValueError("STREAM_CONFIG")
        counts = item.get("layer_counts")
        if not isinstance(counts, list) or len(counts) != layer_count or any(
            not isinstance(value, int) or value < 0 for value in counts
        ):
            raise ValueError("STREAM_LAYERS")
        if sum(counts) != item.get("total_unique_states"):
            raise ValueError("STREAM_TOTAL")
        total += sum(counts)
        layers = [left + right for left, right in zip(layers, counts)]
        chain = str(item.get("archive_chain_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", chain):
            raise ValueError("STREAM_CHAIN")
        rank_chains.append(chain)
        prefix = f"pending/{first['branch']}/states/"
        for source in item.get("files", []):
            source_path = str(source.get("path", ""))
            checksum = str(source.get("sha256", ""))
            if (
                not source_path.startswith(prefix)
                or not source_path.endswith(".parquet")
                or source_path in source_paths
                or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                or not isinstance(source.get("bytes"), int)
                or source["bytes"] <= 0
            ):
                raise ValueError("STREAM_FILE")
            destination = f"states/{first['run_id']}-{source_path.rsplit('/', 1)[-1]}"
            if destination in destinations:
                raise ValueError("STREAM_FILE")
            source_paths.add(source_path)
            destinations.add(destination)
            files.append({
                "source_path": source_path,
                "path": destination,
                "bytes": source["bytes"],
                "sha256": checksum,
            })
    files.sort(key=lambda item: item["path"])
    return {
        "schema": "MGBFS_HF_STREAM_GLOBAL_V1",
        "status": "COMPLETE",
        "run_id": first["run_id"],
        "group_id": first["group_id"],
        "config_digest": first["config_digest"],
        "branch": first["branch"],
        "world_size": expected_world,
        "state_bytes": first["state_bytes"],
        "max_depth": first["max_depth"],
        "layer_counts": layers,
        "total_unique_states": total,
        "rank_archive_chains": rank_chains,
        "files": files,
    }
