"""Validate and merge exact traversal results from local or external ranks."""

import hashlib
import json


def read_terminal_evidence(directory, report):
    digest = report.get("terminal_states_sha256")
    if digest is None:
        return None
    with (directory / "terminal_states.json").open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError("TERMINAL_EVIDENCE_DIGEST_MISMATCH")
    value = json.loads(raw)
    if (value.get("schema") != 1 or value.get("graph_digest") != report["graph_digest"]
            or value.get("depth") != len(report["layer_sizes"]) - 1):
        raise RuntimeError("TERMINAL_EVIDENCE_IDENTITY_MISMATCH")
    return value


def merge_terminal_evidence(report, rank_evidence):
    if all(part is None for part in rank_evidence):
        return None
    if (any(part is None for part in rank_evidence) or report["status"] != "COMPLETE"
            or report["reason"] != "FRONTIER_EXHAUSTED"):
        raise RuntimeError("TERMINAL_EVIDENCE_RANK_CONSENSUS_MISMATCH")
    sizes = report["layer_sizes"]
    value = {"schema": 1, "graph_digest": report["graph_digest"], "depth": len(sizes) - 1}
    for field, length in (("current", sizes[-1]), ("previous_small", sizes[-2] if len(sizes) > 1 else 0)):
        rows = [row for part in rank_evidence for row in part[field]]
        if len(rows) != length or len({tuple(row) for row in rows}) != length:
            raise RuntimeError("TERMINAL_EVIDENCE_LAYER_COVERAGE_MISMATCH")
        value[field] = rows
    text = json.dumps(value, separators=(",", ":"))
    report["terminal_states_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text


def write_terminal_evidence(directory, text):
    if text is not None:
        temporary = directory / "terminal_states.json.tmp"
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(directory / "terminal_states.json")


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
