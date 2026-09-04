#!/usr/bin/env python3
"""Atomically promote validated per-rank staging objects on Hugging Face."""
import argparse
import hashlib
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq


def combine_rank_commits(commits, expected_world):
    if expected_world <= 0 or len(commits) != expected_world:
        raise ValueError("RANK_SET")
    commits = sorted(commits, key=lambda item: item.get("rank", -1))
    if [item.get("rank") for item in commits] != list(range(expected_world)):
        raise ValueError("RANK_SET")
    first = commits[0]
    keys = ("run_id", "group_id", "config_digest", "state_bytes", "max_depth")
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
    branches = []
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
        branch = str(item.get("branch", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", branch):
            raise ValueError("STREAM_BRANCH")
        branches.append(branch)
        prefix = f"pending/{branch}/states/"
        for source in item.get("files", []):
            source_path = str(source.get("path", ""))
            checksum = str(source.get("sha256", ""))
            if (
                not source_path.startswith(prefix)
                or not source_path.endswith(".parquet")
                or (branch, source_path) in source_paths
                or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                or not isinstance(source.get("bytes"), int)
                or source["bytes"] <= 0
            ):
                raise ValueError("STREAM_FILE")
            destination = f"states/{first['run_id']}-{source_path.rsplit('/', 1)[-1]}"
            if destination in destinations:
                raise ValueError("STREAM_FILE")
            source_paths.add((branch, source_path))
            destinations.add(destination)
            files.append({
                "source_path": source_path,
                "source_revision": branch,
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
        # `branch` is retained for V1 single-branch readers.  New multi-rank
        # producers use an isolated branch per rank so concurrent Hub commits
        # cannot race on one Git ref.
        "branch": branches[0],
        "branches": branches,
        "world_size": expected_world,
        "state_bytes": first["state_bytes"],
        "max_depth": first["max_depth"],
        "layer_counts": layers,
        "total_unique_states": total,
        "rank_archive_chains": rank_chains,
        "files": files,
    }


def _metadata_payloads(global_commit):
    run_id = global_commit["run_id"]
    layers = pa.table({
        "run_id": pa.array([run_id] * len(global_commit["layer_counts"]), pa.string()),
        "group_id": pa.array(
            [global_commit["group_id"]] * len(global_commit["layer_counts"]), pa.string()
        ),
        "depth": pa.array(range(len(global_commit["layer_counts"])), pa.uint32()),
        "unique_states": pa.array(global_commit["layer_counts"], pa.uint64()),
    })
    output = pa.BufferOutputStream()
    pq.write_table(layers, output, compression="zstd")
    layer_bytes = output.getvalue().to_pybytes()
    manifest = dict(global_commit)
    manifest["files"] = [dict(item) for item in global_commit["files"]]
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    verification = {
        "schema": "MGBFS_HF_VERIFICATION_V1",
        "run_id": run_id,
        "config_digest": global_commit["config_digest"],
        "rank_archive_chains": global_commit["rank_archive_chains"],
        "objects": [
            {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
            for item in global_commit["files"]
        ],
        "metadata_sha256": {
            f"layers/{run_id}.parquet": hashlib.sha256(layer_bytes).hexdigest(),
            f"runs/{run_id}.json": hashlib.sha256(manifest_bytes).hexdigest(),
        },
    }
    verification_bytes = json.dumps(verification, indent=2, sort_keys=True).encode("utf-8")
    return {
        f"layers/{run_id}.parquet": layer_bytes,
        f"runs/{run_id}.json": manifest_bytes,
        f"verification/{run_id}.json": verification_bytes,
    }


def promote(api, repo_id, commits, expected_world, copy_cls=None, add_cls=None):
    """Create one default-branch commit; no state object is visible before it."""
    if copy_cls is None or add_cls is None:
        from huggingface_hub import CommitOperationAdd, CommitOperationCopy
        copy_cls = CommitOperationCopy
        add_cls = CommitOperationAdd
    combined = combine_rank_commits(commits, expected_world)
    operations = [
        copy_cls(
            src_path_in_repo=item["source_path"],
            path_in_repo=item["path"],
            src_revision=item["source_revision"],
        )
        for item in combined["files"]
    ]
    operations.extend(
        add_cls(path_in_repo=path, path_or_fileobj=payload)
        for path, payload in _metadata_payloads(combined).items()
    )
    receipt = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Publish complete BFS run {combined['run_id']}",
    )
    return combined, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("commits", nargs="+")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN_MISSING")
    from huggingface_hub import HfApi
    commits = []
    for path in args.commits:
        with open(path, "r", encoding="utf-8") as source:
            commits.append(json.load(source))
    combined, receipt = promote(HfApi(token=token), args.repo_id, commits, args.world_size)
    print(json.dumps({
        "status": combined["status"],
        "run_id": combined["run_id"],
        "commit_url": getattr(receipt, "commit_url", None),
    }), flush=True)


if __name__ == "__main__":
    main()
