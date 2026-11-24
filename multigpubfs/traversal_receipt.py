"""Validate and merge exact traversal results from local or external ranks."""

import hashlib
import json


def merge_rank_traversal(reports, snapshots, config, *, error_prefix, segmented, bitmap,
                         sample_limit_field=False, require_admitted_cuts=False):
    devices = config["devices"]
    world = len(devices)
    plans = config["rank_plans"]
    if world == 0 or len(reports) != world or len(snapshots) != world or len(plans) != world:
        raise RuntimeError(f"{error_prefix}_RANK_GEOMETRY_MISMATCH")

    first = reports[0]
    seconds = first["layer_seconds"]
    for rank, (report, device, plan) in enumerate(zip(reports, devices, plans)):
        if (report["rank"] != rank or report["world"] != world
                or report["device"] != device or report["plan"] != plan):
            raise RuntimeError(f"{error_prefix}_RANK_GEOMETRY_MISMATCH")
        if (any(report[key] != first[key] for key in ("status", "reason", "layer_sizes"))
                or len(report["layer_seconds"]) != len(seconds)):
            raise RuntimeError(f"{error_prefix}_LAYER_CONSENSUS_MISMATCH")

    current = [row for snapshot in snapshots for row in snapshot["current"]]
    previous = [row for snapshot in snapshots for row in snapshot["previous_small"]]
    if len(current) > 1000 or len(previous) >= 1000:
        raise RuntimeError(f"{error_prefix}_RETENTION_BOUND")
    states = {
        "schema": 2, "state_encoding": "signed_int64_vectors",
        "current": current, "previous_small": previous,
    }
    if sample_limit_field:
        states["current_sample_limit"] = 1000

    result = dict(first)
    result.pop("rank")
    result.pop("device")
    result.update(
        devices=devices,
        rank_plans=plans,
        owner_cuts=config.get("owner_cuts"),
        layer_seconds=[max(values) for values in zip(*(part["layer_seconds"] for part in reports))],
        bfs_seconds=max(part["bfs_seconds"] for part in reports),
        setup_seconds=max(part["setup_seconds"] for part in reports),
        states_sha256=hashlib.sha256(json.dumps(states).encode()).hexdigest(),
    )
    if segmented:
        memory = [dict(part["segment_memory"], rank=i, device=devices[i])
                  for i, part in enumerate(reports)]
        result["segment_memory_by_rank"] = memory
        result["segment_memory_scope"] = "rank 0 in segment_memory; see segment_memory_by_rank for all devices"
        if not bitmap:
            cuts = [part.get("admitted_owner_cuts") for part in memory]
            if require_admitted_cuts or any(value is not None for value in cuts):
                active = cuts[0]
                if (not isinstance(active, list) or len(active) != world + 1
                        or any(value != active for value in cuts)):
                    raise RuntimeError(f"{error_prefix}_SEGMENTED_OWNER_CUTS_MISMATCH")
                result["planned_owner_cuts"] = result["owner_cuts"]
                result["owner_cuts"] = active
    return result, states
