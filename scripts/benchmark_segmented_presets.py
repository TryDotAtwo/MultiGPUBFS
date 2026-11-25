"""Compare BFS presets or transport variants on the same single-host workload."""

import argparse
from contextlib import nullcontext
from functools import cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from .segmented_acceptance_oracle import verify_output
    from .segmented_benchmark_gpu import BenchmarkGpus, MIB, _nvidia_smi
else:
    from segmented_acceptance_oracle import verify_output
    from segmented_benchmark_gpu import BenchmarkGpus, MIB, _nvidia_smi


S13_WINDOW = (34, 44)
BUILD_KEYS = ("native_sha256", "cuda_library_sha256", "python_runtime_sha256")
UNVERIFIED = "UNVERIFIED"
WITHIN_BOUND = "VERIFIED_WITHIN_4X"
EXCLUDED = "EXCLUDED_ABOVE_4X"


def _time(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value) and value > 0


def _same_context(reference, candidate, tolerance_bytes):
    if "error" in reference or "error" in candidate:
        return "a run or its independent output check failed"
    left, right = (run["runtime_identity"] for run in (reference, candidate))
    if any(not left.get(key) or left.get(key) != right.get(key) for key in BUILD_KEYS):
        return "different or unidentified native, CUDA or Python builds"
    if any(reference[key] != candidate[key] for key in ("graph_digest", "pressure", "request")):
        return "different graph, pressure method or workload request"
    if not reference["graph_digest"]:
        return "unidentified graph"
    if reference["communication_policy"] != candidate["communication_policy"]:
        return "different NCCL preflight policy"
    snapshots = [run[key] for run in (reference, candidate) for key in ("before", "after")]
    for snapshot in snapshots:
        devices = snapshot["devices"]
        if not devices or not snapshot["topology"] or any(not gpu["uuid"] for gpu in devices):
            return "missing GPU identity or topology"
        first = snapshots[0]
        if snapshot["topology"] != first["topology"] or [
            {k: v for k, v in gpu.items() if k != "free_bytes"} for gpu in devices
        ] != [{k: v for k, v in gpu.items() if k != "free_bytes"} for gpu in first["devices"]]:
            return "different GPU IDs, driver or topology"
    if any(
        max(free) - min(free) > tolerance_bytes
        for free in zip(*[[gpu["free_bytes"] for gpu in snap["devices"]] for snap in snapshots])
    ):
        return "starting or post-teardown free VRAM differs beyond the tolerance"
    if not all(run["checks"]["all_completed_layer_counts_match_mahonian"] for run in (reference, candidate)):
        return "independent layer count verification failed"
    return None


def _window(run, n):
    first, last = S13_WINDOW
    if n != 13 or len(run["layer_sizes"]) <= last:
        return None
    ranks = len(run["before"]["devices"])
    commit_times = run.get("layer_committed_seconds_by_rank")
    if commit_times is not None:
        clocks = [clock[first - 1:last + 1] for clock in commit_times]
        if len(clocks) != ranks or any(len(clock) != last - first + 2 or not all(map(_time, clock))
                             or any(a >= b for a, b in zip(clock, clock[1:])) for clock in clocks):
            return None
        duration = max(clock[-1] - clock[0] for clock in clocks)
        through_last = max(clock[-1] for clock in clocks)
        timing = "maximum per-rank monotonic interval from the depth 33 commit through the depth 44 commit"
    else:
        rank_times = [times[first - 1:last] for times in run["layer_seconds_by_rank"]]
        if len(rank_times) != ranks or any(len(times) != last - first + 1 or not all(map(_time, times)) for times in rank_times):
            return None
        duration = max(map(sum, rank_times))
        through_last = None
        timing = "maximum per-rank sum of layer durations (legacy timing without commit clock)"
    sizes = run["layer_sizes"]
    new_states = sum(sizes[first:last + 1])
    observations = run.get("transport_high_water_observed_by_rank")
    depth_rows = []
    sync_by_rank = None
    if observations and len(observations) == ranks:
        sync_by_rank = [
            [rank.get("global_sync_by_depth", {}).get(str(depth))
             for depth in range(first, last + 1)]
            for rank in observations
        ]
        if any(not isinstance(sync, dict) for rank in sync_by_rank for sync in rank):
            sync_by_rank = None
        for rank in observations:
            by_depth = rank.get("transport_by_depth", {})
            depth_rows.extend(by_depth.get(str(depth), {}) for depth in range(first, last + 1))
    submitted_bytes = _sum_rank(depth_rows, "remote_submitted_nccl_bytes")
    send_calls = _sum_rank(depth_rows, "remote_nccl_send_calls")
    return {
        "remote_submitted_nccl_bytes": submitted_bytes,
        "remote_nccl_send_calls": send_calls,
        "remote_submitted_nccl_bytes_per_new_state": submitted_bytes / new_states if submitted_bytes is not None and new_states else None,
        "remote_nccl_send_calls_per_new_state": send_calls / new_states if send_calls is not None and new_states else None,
        "global_sync_by_rank_and_depth": sync_by_rank,
        "seconds": duration,
        "seconds_from_bfs_start_through_last_commit": through_last,
        "timing": timing,
        "new_states": new_states,
        "generated_edges": (n - 1) * sum(sizes[first - 1:last]),
    }


def _ratio(reference_seconds, candidate_seconds, reason, limit=4.0):
    verified = reason is None and _time(reference_seconds) and _time(candidate_seconds)
    result = {
        "status": UNVERIFIED, "slowdown": None,
        "reason": reason or "missing or invalid timing",
    }
    if verified:
        slowdown = candidate_seconds / reference_seconds
        result.update(
            status="VERIFIED" if limit is None else EXCLUDED if slowdown > limit else WITHIN_BOUND,
            slowdown=slowdown, reason=None,
        )
    return result


def _stopped_on_vram(run):
    if run.get("status") != "INCOMPLETE" or run.get("reason") != "RESOURCE_10":
        return False
    vram_limit_reasons = {
        "segment_budget", "cuda_allocation", "descriptor_budget", "descriptor_cuda_allocation",
        "hash_directory_budget", "hash_directory_growth_budget", "hash_view_budget",
        "hash_directory_budget_cuda_allocation", "hash_directory_growth_budget_cuda_allocation",
        "hash_view_budget_cuda_allocation",
    }
    details = [item for item in run.get("resource_detail_by_rank", []) if item is not None]
    return bool(details) and all(item.get("reason") in vram_limit_reasons for item in details)


def _capacity_bracket(completions, failures):
    completed = max(completions, key=lambda row: row["held_mib_per_gpu"], default=None)
    failed = None
    if completed is not None:
        usable = (row for row in failures if row["held_mib_per_gpu"] > completed["held_mib_per_gpu"]
                  and all(a < b for a, b in zip(row["before_free_bytes_by_rank"],
                                                completed["before_free_bytes_by_rank"], strict=True)))
        failed = min(usable, key=lambda row: row["held_mib_per_gpu"], default=None)
    return {"complete": completed, "vram_limited": failed}


def compare_pair(reference, candidate, n, tolerance_bytes, *, labels=("balanced", "maxmemory")):
    reason = _same_context(reference, candidate, tolerance_bytes)
    limit = 4.0 if labels == ("balanced", "maxmemory") else None
    runs = (reference, candidate)
    full_reason = reason
    if full_reason is None and not all(run["checks"]["final_two_layers_full_exact_sets_no_duplicates"] for run in runs):
        full_reason = "both runs must exhaust the BFS and verify the two terminal layers"
    if full_reason is None and reference["layer_sizes"] != candidate["layer_sizes"]:
        full_reason = "completed full layer sequences differ"
    windows = [None, None] if reason else [_window(run, n) for run in runs]
    window_reason = reason
    if window_reason is None and n != 13:
        window_reason = "preregistered window applies to S13"
    if window_reason is None and not all(windows):
        window_reason = "both runs must complete and time every depth from 34 through 44"
    if window_reason is None:
        through_window = S13_WINDOW[1] + 1
        if reference["layer_sizes"][:through_window] != candidate["layer_sizes"][:through_window]:
            window_reason = "committed layer counts differ through depth 44"
    if window_reason is None and windows[0]["timing"] != windows[1]["timing"]:
        window_reason = "the completed windows use different timing sources"
    if window_reason is None and labels == ("runs", "snapshot") and not any(
        row and row.get("tiles_with_snapshot", 0) > 0
        for row in candidate.get("hash_timing_by_rank", ())
    ):
        window_reason = "no timed lookups used a snapshot in depths 34-44"

    full = _ratio(*(run.get("bfs_seconds") for run in runs), full_reason, limit=limit)
    # An unfinished run has no full-run timing. Its measured time can still
    # exclude a recipe which has already used more than the full 4× allowance.
    if (limit is not None and reason is None and full["status"] == UNVERIFIED
            and reference["checks"]["final_two_layers_full_exact_sets_no_duplicates"]
            and candidate.get("status") == "INCOMPLETE"
            and candidate.get("reason") == "DEADLINE"
            and reference["layer_sizes"][:len(candidate["layer_sizes"])] == candidate["layer_sizes"]
            and len(candidate["layer_sizes"]) < len(reference["layer_sizes"])
            and all(_time(run.get("bfs_seconds")) for run in runs)):
        lower_bound = candidate["bfs_seconds"] / reference["bfs_seconds"]
        if lower_bound > limit:
            full["exclusion"] = {
                "status": EXCLUDED,
                "slowdown_lower_bound": lower_bound,
                "basis": "the verified candidate prefix already exceeded the full balanced time allowance",
            }
    capacity = {"status": UNVERIFIED}
    if reason is None and limit is not None:
        for winner, limited, winner_label in (
            (reference, candidate, labels[0]), (candidate, reference, labels[1]),
        ):
            prefix = limited["layer_sizes"]
            if (winner["checks"]["final_two_layers_full_exact_sets_no_duplicates"]
                    and _stopped_on_vram(limited)
                    and 0 < len(prefix) < len(winner["layer_sizes"])
                    and winner["layer_sizes"][:len(prefix)] == prefix):
                capacity = {
                    "status": "VERIFIED_COMPLETION_AT_MATCHED_PRESSURE",
                    "completed_preset": winner_label,
                    "resource_limited_depth": len(prefix) - 1,
                }
                break
    return {
        "comparable_context": reason is None, "context_reason": reason,
        "full": full,
        "capacity_at_matched_pressure": capacity,
        "s13_depths_34_44": {
            **_ratio(*(window and window["seconds"] for window in windows), window_reason, limit=limit),
            "depth_first": S13_WINDOW[0], "depth_last": S13_WINDOW[1],
            "timing": windows[0]["timing"] if window_reason is None else None,
            labels[0]: windows[0], labels[1]: windows[1],
        },
    }


def _aggregate_comparisons(comparisons, verified):
    aggregate = {}
    for scope in ("full", "s13_depths_34_44"):
        evidence = [pair[scope] for pair in comparisons]
        if any(
            item["status"] == EXCLUDED or item.get("exclusion", {}).get("status") == EXCLUDED
            for item in evidence
        ):
            aggregate[scope] = EXCLUDED
        elif evidence and all(item["status"] == verified for item in evidence):
            aggregate[scope] = verified
        else:
            aggregate[scope] = UNVERIFIED
    return aggregate


def _sum_rank(ranks, key):
    values = [rank.get(key) for rank in ranks]
    return sum(values) if values and all(type(value) is int and value >= 0 for value in values) else None


def metrics(n, report, rank_reports):
    sizes = report["layer_sizes"]
    unique = sum(sizes)
    new_states = sum(sizes[1:])
    ranks = report["segment_memory_by_rank"]
    storage = _sum_rank(ranks, "storage_allocated_bytes")
    bitmap = _sum_rank(ranks, "bitmap_bytes")
    records = None
    if ranks and all(rank.get("transport_trace") is True for rank in ranks):
        records = _sum_rank(ranks, "remote_records_sent")
    packet_receipts = [rank.get("packet_transport") for rank in ranks]
    compact = bool(packet_receipts) and all(isinstance(receipt, dict) for receipt in packet_receipts)
    # Exact local exchange sends one 32-byte record per traced row. Generic
    # signed byte states through 24 elements are already embedded in that record.
    payload_bytes = _sum_rank(ranks, "remote_candidate_payload_bytes")
    plans = [rank["plan"] for rank in rank_reports]
    if compact:
        payload_bytes = _sum_rank(packet_receipts, "submitted_payload_bytes")
    elif payload_bytes is None and records is not None and len(plans) == len(ranks) and all(rank.get("exact_exchange") is True for rank in ranks):
        dimensions = {(plan.get("elements"), plan.get("state_bytes"), plan.get("parent_transport")) for plan in plans}
        if dimensions == {(n, 1, False)} and n <= 24:
            payload_bytes = records * 32
    transfers = (_sum_rank(packet_receipts, "submitted_packets") if compact
                 else _sum_rank(ranks, "remote_payload_transfers"))
    submitted_bytes = _sum_rank(ranks, "remote_submitted_nccl_bytes")
    send_calls = _sum_rank(ranks, "remote_nccl_send_calls")
    result = {
        "remote_submitted_nccl_bytes": submitted_bytes,
        "remote_submitted_nccl_metadata_bytes": _sum_rank(ranks, "remote_submitted_nccl_metadata_bytes"),
        "remote_nccl_send_calls": send_calls,
        "remote_submitted_nccl_bytes_per_committed_new_state": (
            submitted_bytes / new_states if submitted_bytes is not None and new_states else None
        ),
        "remote_nccl_send_calls_per_committed_new_state": (
            send_calls / new_states if send_calls is not None and new_states else None
        ),
        "committed_depth": len(sizes) - 1, "global_completed_unique": unique,
        "global_committed_new_states": new_states,
        "global_peak_frontier": max(sizes),
        "logical_edges_into_completed_layers": (n - 1) * sum(sizes[:-1]),
        "global_segment_storage_bytes_at_end": storage, "global_bitmap_bytes_at_end": bitmap,
        "global_queue_allocated_bytes_at_end": _sum_rank(ranks, "queues_allocated_bytes"),
        "sum_of_rank_storage_request_high_water_bytes": _sum_rank(ranks, "storage_peak_requested_bytes"),
        "sum_of_rank_peak_live_rows": _sum_rank(ranks, "peak_live_rows"),
        "hash_reclaimed_segments": _sum_rank(ranks, "hash_reclaimed_segments"),
        "hash_reclaimed_device_bytes": _sum_rank(ranks, "hash_reclaimed_device_bytes"),
        "sum_of_rank_source_retries": _sum_rank(rank_reports, "source_retries"),
        "storage_plus_bitmap_bytes_per_completed_unique_at_end": (
            (storage + bitmap) / unique if storage is not None and bitmap is not None else None
        ),
        "remote_candidate_records": records,
        "remote_candidate_records_per_completed_unique": records / unique if records is not None else None,
        "remote_candidate_application_payload_bytes": payload_bytes,
        "remote_candidate_payload_bytes_per_completed_unique": payload_bytes / unique if payload_bytes is not None else None,
        "remote_payload_transfers": transfers,
        "remote_payload_transfers_per_committed_new_state": (
            transfers / new_states if transfers is not None and new_states else None
        ),
        "remote_payload_bytes_per_committed_new_state": (
            payload_bytes / new_states if payload_bytes is not None and new_states else None
        ),
        "packet_bytes_including_shard_descriptors": _sum_rank(packet_receipts, "submitted_bytes") if compact else None,
        "packet_hub_peak_active_receive_credits": (
            packet_receipts[0].get("hub_peak_active_receive_credits") if compact else None
        ),
        "packet_split_destination_windows": (
            _sum_rank(packet_receipts, "split_destination_windows") if compact else None
        ),
        "packet_pinned_route_metadata_by_rank": (
            [receipt.get("pinned_route_metadata") for receipt in packet_receipts] if compact else None
        ),
        "transport_allocated_bytes_by_rank": [rank.get("transport_allocated_bytes") for rank in ranks],
        "transport_device_peak_bytes_by_rank": [rank.get("transport_device_peak_bytes") for rank in ranks],
        "transport_device_peak_breakdown_by_rank": [rank.get("transport_device_peak_breakdown") for rank in ranks],
        "admitted_batch_by_rank": [rank.get("admitted_batch") for rank in ranks],
        "configured_owner_lanes_by_rank": [rank.get("owner_lanes") for rank in ranks],
        "peak_active_owner_lanes_by_rank": [rank.get("peak_owner_lanes") for rank in rank_reports],
        "global_sync_by_rank": [rank.get("global_sync") for rank in ranks],
        "batch_hash_timing_by_rank": [rank.get("batch_hash_timing") for rank in ranks],
        "scope": {
            "storage": "sum of segment storage and bitmap allocations at final snapshot",
            "high_water": "sum of independent per-rank maxima, an upper bound on simultaneous usage",
            "transport": "sent remote candidate records including retries or unfinished work; application payload excludes count exchange, collectives and protocol overhead",
            "nccl_sends": "explicit remote P2P sends including padding and count or descriptor metadata, excluding collectives, control TCP and NCCL wire overhead",
        },
    }
    return result


def transport_observations(output, devices):
    peaks = []
    for rank, _ in enumerate(devices):
        entries = []
        with (output / f"rank-{rank}.log").open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("SEGMENT_LAYER_STATS "):
                    entries.append(json.loads(line.removeprefix("SEGMENT_LAYER_STATS ")))
        if not entries:
            return None
        peaks.append({
            "rank": rank, "observed_depth_min": min(row["depth"] for row in entries),
            "observed_depth_max": max(row["depth"] for row in entries),
            "global_sync_by_depth": {row["depth"]: row["global_sync"] for row in entries if "global_sync" in row},
            "transport_by_depth": {
                str(row["depth"]): {
                    field: row.get(field) for field in (
                        "remote_candidate_payload_bytes", "remote_submitted_nccl_bytes",
                        "remote_submitted_nccl_metadata_bytes", "remote_nccl_send_calls",
                    )
                } for row in entries
            },
            **{field: max(row[field] for row in entries) for field in (
                "peak_send_bucket", "peak_receive_bucket", "global_peak_queue",
            )},
        })
    return peaks


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run_leg(graph, output, gpus, native, env, recipe, request, pressure, *,
             transport_mode="exact", registry_evidence=False):
    from multigpubfs.generic_session import persistent_graph_workers
    from multigpubfs.launch import run_graph

    before = gpus.snapshot()
    started_unix_ns = time.time_ns()
    started = time.monotonic()
    sampler = gpus.sample_memory(before) if registry_evidence else nullcontext()
    with sampler as memory, persistent_graph_workers():
        report = run_graph(
            graph, output, executable=native, backend="generic", autotune=False,
            devices=request["devices"], shards=request["shards"], max_seconds=request["max_seconds"],
            segment_rows=request["segment_rows"], _batch=request["batch"],
            _profile_layers=request["max_depth"], _native_env=env, **recipe,
        )
    wall = time.monotonic() - started
    ended_unix_ns = time.time_ns()
    after = gpus.snapshot()
    if report["devices"] != request["devices"] or report["graph_digest"] != graph.digest():
        raise ValueError("native graph digest or ordered GPU ordinals differ from the request")
    n = request["degree"]
    rank_reports = [
        json.loads((output / name).read_text(encoding="utf-8")) for name in report["rank_receipts"]
    ]
    expected_codec = "pack4-v1" if recipe["bitmap_pack4"] else "unpacked-soa"
    ranks = report["segment_memory_by_rank"]
    if len(rank_reports) != len(ranks) or len(ranks) != len(request["devices"]) or any(
        rank["device"] != device or rank["segment_strategy"] != recipe["segment_strategy"]
        or rank["resident_codec"] != expected_codec
        for rank, device in zip(ranks, request["devices"])
    ):
        raise ValueError("wrong GPU, segment strategy or resident codec in native rank receipt")
    if transport_mode == "compact" and any(
        not isinstance(rank.get("packet_transport"), dict) for rank in ranks
    ):
        raise ValueError("compact packet mode missing from a native rank receipt")
    if transport_mode == "exact" and any(
        rank.get("exact_exchange") is not True or rank.get("packet_transport") is not None
        for rank in ranks
    ):
        raise ValueError("exact exchange missing from a native rank receipt")
    history = request.get("hash_series_history")
    if history:
        expected_banks = 1 if history == "append" else 3
        if any(rank.get("banks") != expected_banks for rank in ranks):
            raise ValueError(f"hash-series comparison requires {expected_banks} history banks in both legs")
    terminal = None
    if registry_evidence and report["status"] == "COMPLETE":
        from multigpubfs.traversal_receipt import read_terminal_evidence

        terminal = read_terminal_evidence(output, report)
        if terminal is None:
            raise ValueError(f"native terminal evidence missing: {report.get('terminal_evidence_status')}")
    checks = verify_output(n, report, output / "states.json", terminal=terminal)
    preflight = report.get("communication_preflight", {})
    profile_evidence = {}
    if registry_evidence:
        from multigpubfs.profile_selection import nccl_profile_network

        profile_evidence["profile_vram_sampling"] = memory
        same_visibility = all(env.get(key) == os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES",
        ))
        if same_visibility and preflight.get("status") in ("VERIFIED", "SINGLE_GPU"):
            profile_evidence["profile_network"] = nccl_profile_network(dict(env, **(preflight.get("policy") or {})))
            if len(request["devices"]) > 1:
                from multigpubfs.native_distribution import runtime_identity

                provenance = runtime_identity(native, env, profile_nccl=True)
                if all(provenance.get(key) == report["runtime_identity"].get(key) for key in BUILD_KEYS):
                    profile_evidence["profile_nccl_library_sha256"] = provenance.get("nccl_library_sha256")
    return {
        "runtime_identity": report["runtime_identity"], "graph_digest": report["graph_digest"],
        "recipe": recipe, "transport_mode": transport_mode,
        "hash_series_requested": env.get("MGBFS_SEGMENT_HASH_SERIES") == "1",
        "hash_segment_reuses_by_rank": [rank.get("hash_segment_reuses") for rank in ranks],
        "hash_series_by_rank": [rank.get("batch_hash_runs_by_shard") for rank in ranks],
        "hash_snapshot_bytes_by_rank": [rank.get("hash_snapshot_bytes") for rank in ranks],
        "hash_snapshot_rebuild_peak_bytes_by_rank": [rank.get("hash_snapshot_rebuild_peak_bytes") for rank in ranks],
        "hash_snapshot_installations_by_rank": [rank.get("hash_snapshot_installations") for rank in ranks],
        "hash_snapshot_storage_evictions_by_rank": [
            rank.get("hash_snapshot_storage_evictions") for rank in ranks
        ],
        "hash_timing_by_rank": [rank.get("batch_hash_timing") for rank in ranks],
        "request": request, "pressure": pressure, "before": before, "after": after,
        "checks": checks, "status": report["status"], "reason": report["reason"],
        "resource_detail_by_rank": [rank.get("resource_detail") for rank in ranks],
        "communication_policy": {key: preflight.get(key) for key in ("status", "selected", "policy")},
        "layer_sizes": report["layer_sizes"], "bfs_seconds": report["bfs_seconds"],
        "layer_seconds_by_rank": [rank["layer_seconds"] for rank in rank_reports],
        "layer_committed_seconds_by_rank": (
            [rank["layer_committed_seconds"] for rank in rank_reports]
            if all("layer_committed_seconds" in rank for rank in rank_reports) else None
        ),
        "measurement_unix_ns": [started_unix_ns, ended_unix_ns],
        "wall_seconds_with_setup_and_teardown": wall, "setup_seconds": report["setup_seconds"],
        "metrics": metrics(n, report, rank_reports),
        "transport_high_water_observed_by_rank": (
            transport_observations(output, request["devices"]) if request["transport_high_water"] else None
        ),
        **profile_evidence,
    }


def _physical_indices_by_uuid():
    try:
        output = _nvidia_smi("--query-gpu=index,uuid", "--format=csv,noheader,nounits")
        indices = {}
        for line in output.splitlines():
            index, uuid = (field.strip() for field in line.split(","))
            if not index.isascii() or not index.isdecimal() or not uuid.startswith("GPU-") or uuid in indices:
                return {}
            indices[uuid] = int(index)
        return indices if len(set(indices.values())) == len(indices) else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def _registry_topology(legs, resolve_indices):
    from multigpubfs.profile_registry import local_profile_topology

    reference = legs["balanced"]
    snapshot = reference["before"]
    topology = snapshot["topology"]
    network = reference.get("profile_network")
    if (network is None or resolve_indices is None or
            not all(gpu.get("pci_bus_id") for gpu in snapshot["devices"])):
        return topology
    indices = resolve_indices()
    slots = [indices.get(gpu["uuid"]) for gpu in snapshot["devices"]]
    return local_profile_topology(topology, slots, network) or topology


def _export_registry_pair(graph, pair_path, legs, comparison, *, resolve_indices=None):
    from multigpubfs.profile_registry import export_profiles

    full = comparison["full"]
    if full["status"] != WITHIN_BOUND:
        return {"status": full["status"], "reason": full["reason"] or "paired full traversal exceeded 4×"}
    if legs["balanced"].get("profile_network") != legs["maxmemory"].get("profile_network"):
        return {"status": UNVERIFIED, "reason": "effective NCCL overrides differ across preset runs"}
    nccl = legs["balanced"].get("profile_nccl_library_sha256")
    if nccl != legs["maxmemory"].get("profile_nccl_library_sha256"):
        return {"status": UNVERIFIED, "reason": "NCCL library provenance differs across preset runs"}
    topology = _registry_topology(legs, resolve_indices)
    receipts = {}
    inventory = None
    pressure = legs["balanced"]["pressure"]
    held = pressure["warmed_allocation_bytes_per_gpu"]
    namespace = f"cuda_warmed_allocation:{held}" if held else "ambient"
    for label, leg in legs.items():
        memory = leg["profile_vram_sampling"]
        baseline, peak = memory["baseline_bytes_by_rank"], memory["peak_bytes_by_rank"]
        if (memory.get("error") or memory["samples"] == 0 or len(peak) != len(leg["before"]["devices"])
                or len(baseline) != len(peak) or any(a <= b for a, b in zip(peak, baseline))):
            return {"status": UNVERIFIED, "reason": f"{label}: NVML did not observe live allocation on every rank"}
        output = pair_path / label
        launch = json.loads((output / "launch.json").read_text(encoding="utf-8"))
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        devices = leg["before"]["devices"]
        if (launch["graph_digest"] != report["graph_digest"] or
                launch["devices"] != report["devices"] or
                [rank["device"] for rank in launch["inventory"]] != [gpu["ordinal"] for gpu in devices]):
            raise ValueError(f"{label}: startup inventory does not match the native rank reports")
        fields = ("device", "compute_major", "compute_minor", "sm_count", "total_bytes")
        identity = {
            "devices": report["devices"],
            "driver_versions": [gpu["driver"] for gpu in devices],
            "inventory": [{key: rank[key] for key in fields} for rank in launch["inventory"]],
        }
        if inventory is not None and inventory != identity:
            raise ValueError("owning-rank admission hardware differs across preset runs")
        inventory = identity
        report.update(
            profile_pressure_namespace=namespace,
            profile_peak_vram_bytes_by_rank=peak,
            profile_admission_by_rank=[
                {"free_bytes_at_admission": rank.get("free_bytes"),
                 "additional_reserve_bytes": rank.get("additional_reserve_bytes", launch.get("additional_reserve_bytes"))}
                for rank in launch["inventory"]
            ],
        )
        if nccl and isinstance(topology, dict):
            report["runtime_identity"]["nccl_library_sha256"] = nccl
        receipts[label] = report
    reference, candidate = (legs[label]["profile_vram_sampling"] for label in ("balanced", "maxmemory"))
    free_at_boundaries = [[gpu["free_bytes"] for gpu in leg[boundary]["devices"]]
                          for leg in legs.values() for boundary in ("before", "after")]
    drift = [max(max(samples) - min(samples), abs(left - right))
             for samples, left, right in zip(zip(*free_at_boundaries, strict=True),
                                             reference["baseline_bytes_by_rank"],
                                             candidate["baseline_bytes_by_rank"], strict=True)]
    reference_peak = max(reference["peak_bytes_by_rank"])
    candidate_peak = max(candidate["peak_bytes_by_rank"])
    # Device-wide peaks include unrelated resident bytes. Give the candidate
    # the largest measured boundary drift on each physical GPU before comparing maxima.
    guarded_peak = max(peak + extra for peak, extra in zip(candidate["peak_bytes_by_rank"], drift, strict=True))
    drift_evidence = {"boundary_vram_drift_bytes_by_rank": drift,
                      "maxmemory_peak_gap_after_drift_bytes": reference_peak - guarded_peak}
    if reference_peak > candidate_peak and reference_peak <= guarded_peak:
        return {"status": UNVERIFIED, "reason": "measured idle VRAM drift can explain the lower device peak",
                **drift_evidence}
    for label, report in receipts.items():
        target = pair_path / label / "report.json"
        temporary = target.with_suffix(".json.tmp")
        _write(temporary, report)
        temporary.replace(target)
    destination = pair_path / "profile"
    destination.mkdir()
    _write(destination / "graph.json", graph.to_dict())
    _write(destination / "inventory.json", inventory)
    (destination / "topology.txt").write_text(legs["balanced"]["before"]["topology"] + "\n", encoding="utf-8")
    topology_path = "profile/topology.txt"
    physical_slots = isinstance(topology, dict)
    if physical_slots:
        (destination / "topology.json").write_text(
            json.dumps(topology, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8",
        )
        topology_path = "profile/topology.json"
    catalog = destination / "registry.json"
    entry = export_profiles(graph.to_dict(), inventory, topology, pair_path / "balanced",
                            pair_path / "maxmemory", catalog, destination / "evidence")
    return {
        "status": "EXPORTED", "capacity_verification": entry["capacity_verification"],
        "pressure_namespace": namespace, "catalog": "profile/registry.json",
        "topology": topology_path, "topology_verification": "PHYSICAL_SLOTS" if physical_slots else UNVERIFIED,
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "evidence": f"profile/evidence/{entry['evidence_sha256']}.json",
        **drift_evidence,
    }


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _devices(value):
    selected = [int(device) for device in value.split(",")]
    if min(selected) < 0 or len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError("expected distinct nonnegative CUDA ordinals")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="artifact directory for this run")
    parser.add_argument("--devices", type=_devices, default=[0])
    parser.add_argument("--executable")
    parser.add_argument("--n", type=int, choices=range(2, 17), default=13)
    parser.add_argument("--repeat", type=_positive, default=1, help="paired repeats, alternating which preset runs first")
    parser.add_argument("--max-depth", type=_positive)
    parser.add_argument("--max-seconds", type=_positive)
    parser.add_argument("--registry-evidence", action="store_true",
                        help="sample NVML peaks and export a verified full paired preset to pair-N/profile")
    parser.add_argument("--batch", type=_positive, default=8192)
    parser.add_argument("--shards", type=_positive, help="defaults to the number of GPUs")
    parser.add_argument("--segment-rows", type=_positive, help="same maximum rows for both modes, otherwise automatic")
    for label, default in (("balanced", "warp"), ("maxmemory", "atomic")):
        parser.add_argument(f"--{label}-strategy", choices=("atomic", "warp", "batch"), default=default)
        parser.add_argument(f"--{label}-queue-cap", type=_positive, help="explicit candidate rows per transport queue")
        parser.add_argument(f"--{label}-owner-lanes", type=_positive, help="explicit owner lane count per rank")
    for label, default in (("balanced", "segmented24"), ("maxmemory", "permutation_bitmap")):
        parser.add_argument(f"--{label}-backend", choices=("segmented24", "permutation_bitmap"), default=default)
    for label in ("balanced", "maxmemory"):
        parser.add_argument(
            f"--{label}-hash-probe", choices=("scalar", "prefix"),
            help=f"explicit lookup kernel for the {label} sorted hash recipe",
        )
    parser.add_argument("--unpacked-bitmap", action="store_true")
    parser.add_argument("--hash-timing", action="store_true",
                        help="time sorted hash probe and merge kernels on comparable layers")
    parser.add_argument("--transport-high-water", action="store_true", help="trace native per-layer bucket occupancy in both legs")
    packet_args = parser.add_mutually_exclusive_group()
    packet_args.add_argument("--preset-packet-control-addr", help="use compact transport for both presets")
    packet_args.add_argument("--maxmemory-packet-control-addr",
                             help="use exact exchange for balanced and compact transport for maxmemory")
    packet_args.add_argument("--compare-hash-strategies", action="store_true", help="compare warp vs bounded batch with the same balanced settings")
    packet_args.add_argument("--compare-hash-eager-tail", metavar="RUNS", type=int,
                             help="compare default batch snapshot timing with a different eager threshold (0 disables eager rebuilds)")
    packet_args.add_argument("--compare-hash-reuse", action="store_true", help="compare free vs reuse on the same rolling hash recipe")
    parser.add_argument("--hash-reuse-strategy", choices=("batch", "warp", "atomic"), default="batch",
                        help="hash strategy for both legs of --compare-hash-reuse")
    packet_args.add_argument("--compare-hash-series", action="store_true", help="compare legacy and snapshot lookup with identical batch hash geometry")
    packet_args.add_argument("--packet-control-addr", help="compare exact and compact on the same recipe; provide a free address such as 127.0.0.1:29761")
    parser.add_argument("--hash-series-history", choices=("rolling", "append"), default="rolling")
    parser.add_argument("--transport-recipe", choices=("balanced", "maxmemory"), default="balanced",
                        help="recipe used by both legs of --packet-control-addr")
    parser.add_argument("--free-tolerance-mib", type=_positive, default=128)
    pressure_args = parser.add_mutually_exclusive_group()
    pressure_args.add_argument("--pressure-mib", type=_positive, action="append",
                               help="hold warmed CUDA memory on each GPU; repeat for a capacity sweep")
    pressure_args.add_argument("--segment-budget-bytes", type=_positive, help="cap only segment storage per GPU; leaves total physical VRAM unchanged")
    args = parser.parse_args()
    if args.output is None:
        from multigpubfs.artifact_paths import new_benchmark_directory

        args.output = new_benchmark_directory()
    preset_labels = (args.transport_recipe,) if args.packet_control_addr else ("balanced", "maxmemory")
    compare_eager_tail = args.compare_hash_eager_tail is not None
    if compare_eager_tail and args.compare_hash_eager_tail not in {0, *range(2, 65)}:
        parser.error("--compare-hash-eager-tail must be 0 or between 2 and 64")
    hash_comparison = (args.compare_hash_series or args.compare_hash_reuse
                       or args.compare_hash_strategies or compare_eager_tail)
    if args.registry_evidence and (args.max_depth or hash_comparison or args.packet_control_addr
                                   or args.segment_budget_bytes or args.hash_timing
                                   or args.transport_high_water):
        parser.error("--registry-evidence requires full presets without test-only caps or tracing; "
                     "use --pressure-mib for physical pressure")
    if hash_comparison and args.balanced_backend != "segmented24":
        parser.error("hash comparison requires a segmented24 balanced backend")
    if not hash_comparison:
        for label in preset_labels:
            if (getattr(args, f"{label}_backend") == "permutation_bitmap"
                    and not args.unpacked_bitmap and getattr(args, f"{label}_strategy") != "atomic"):
                parser.error(f"Pack4 requires --{label}-strategy atomic")
    packet_addr = args.packet_control_addr or args.preset_packet_control_addr or args.maxmemory_packet_control_addr
    if packet_addr and len(args.devices) < 2:
        parser.error("compact transport requires at least two local GPUs")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        parser.error("use a single host process and pass all local CUDA ordinals via --devices")

    from multigpubfs import GraphDefinition
    from multigpubfs.native_distribution import native_runtime

    n = args.n
    generators = [list(range(i)) + [i + 1, i] + list(range(i + 2, n)) for i in range(n - 1)]
    graph = GraphDefinition.permutation(generators, list(range(n)))
    selected = args.devices
    gpus = BenchmarkGpus(selected)
    native, inherited = native_runtime(args.executable)
    env = {k: v for k, v in inherited.items() if not k.startswith(("MGBFS_SEGMENT_", "MGBFS_BITMAP_"))}
    env.update(MGBFS_SEGMENT_EXACT_EXCHANGE="1")
    if args.registry_evidence:
        from multigpubfs.launch import _native_key_info

        if _native_key_info(native, env).get("graph_profile_terminal_evidence_v1") is not True:
            parser.error("native executable does not support full terminal evidence capture")
        env["MGBFS_PROFILE_TERMINAL_EVIDENCE_DEPTH"] = str(n * (n - 1) // 2)
    else:
        env["MGBFS_SEGMENT_TRANSPORT_TRACE"] = "1"
    if args.transport_high_water:
        env["MGBFS_SEGMENT_LAYER_STATS"] = "1"
    if args.compare_hash_series and args.hash_series_history == "append":
        env["MGBFS_SEGMENT_FULL_HISTORY"] = "1"
    if args.hash_timing or args.compare_hash_series:
        first, last = S13_WINDOW if n == 13 else (1, n * (n - 1) // 2)
        env.update(MGBFS_SEGMENT_HASH_TIMING="1",
                   MGBFS_SEGMENT_HASH_TIMING_MIN_DEPTH=str(first),
                   MGBFS_SEGMENT_HASH_TIMING_MAX_DEPTH=str(last))
    if args.segment_budget_bytes:
        env["MGBFS_SEGMENT_TEST_BYTES"] = str(args.segment_budget_bytes)
    pressure_levels = sorted(set(args.pressure_mib or [0]))
    preset_environment = {
        label: {key: str(value) for key, value in (
            ("MGBFS_SEGMENT_QUEUE_CAP", getattr(args, f"{label}_queue_cap")),
            ("MGBFS_SEGMENT_OWNER_LANES", getattr(args, f"{label}_owner_lanes")),
        ) if value is not None}
        for label in ("balanced", "maxmemory")
    }
    for label in ("balanced", "maxmemory"):
        probe = getattr(args, f"{label}_hash_probe")
        if probe is not None:
            preset_environment[label]["MGBFS_SEGMENT_HASH_PREFIX_PROBE"] = str(int(probe == "prefix"))
    request = {
        "degree": n, "devices": selected, "shards": args.shards or len(selected),
        "batch": args.batch, "segment_rows": args.segment_rows, "max_depth": args.max_depth,
        "max_seconds": args.max_seconds, "transport_high_water": args.transport_high_water,
        "hash_timing": args.hash_timing or args.compare_hash_series,
        "hash_series_history": args.hash_series_history if args.compare_hash_series else None,
        "hash_eager_tail_experiment": args.compare_hash_eager_tail,
        "preset_transport": {
            "balanced": "compact" if args.preset_packet_control_addr else "exact",
            "maxmemory": "compact" if args.preset_packet_control_addr or args.maxmemory_packet_control_addr else "exact",
        },
        "preset_environment": preset_environment,
    }
    recipes = {}
    for label in ("balanced", "maxmemory"):
        backend = getattr(args, f"{label}_backend")
        recipes[label] = dict(
            memory_backend=backend, memory_preset=label,
            segment_strategy=getattr(args, f"{label}_strategy"),
            bitmap_pack4=backend == "permutation_bitmap" and not args.unpacked_bitmap,
        )
    if args.packet_control_addr:
        recipes = dict.fromkeys(("exact", "compact"), recipes[args.transport_recipe])
    if args.compare_hash_series:
        recipes = dict.fromkeys(("runs", "snapshot"), {**recipes["balanced"], "segment_strategy": "batch"})
    if args.compare_hash_reuse:
        recipes = dict.fromkeys(("free", "reuse"), {**recipes["balanced"], "segment_strategy": args.hash_reuse_strategy})
    if args.compare_hash_strategies:
        recipes = {strategy: {**recipes["balanced"], "segment_strategy": strategy}
                   for strategy in ("warp", "batch")}
    if compare_eager_tail:
        recipes = dict.fromkeys(("default", "tuned"), {**recipes["balanced"], "segment_strategy": "batch"})
    labels = tuple(recipes)
    args.output.mkdir(parents=True, exist_ok=False)
    _write(args.output / "invocation.json", {
        "graph_digest": graph.digest(), "native": native, "pressure_levels_mib_per_gpu": pressure_levels,
        "request": request, "recipes": recipes, "repeat": args.repeat, "s13_window": S13_WINDOW,
        "packet_control_addr": args.packet_control_addr,
        "preset_packet_control_addr": args.preset_packet_control_addr,
        "maxmemory_packet_control_addr": args.maxmemory_packet_control_addr,
        "transport_recipe": args.transport_recipe if args.packet_control_addr else None,
        **({"registry_evidence": True} if args.registry_evidence else {}),
    })
    observed = {label: [] for label in labels}
    limited = {label: [] for label in labels}
    resolve_indices = cache(_physical_indices_by_uuid)
    sweep = len(pressure_levels) > 1
    for held_mib in pressure_levels:
        held = held_mib * MIB
        pressure = {
            "method": "cuda_warmed_allocation" if held else "segment_storage_cap" if args.segment_budget_bytes else "ambient",
            "warmed_allocation_bytes_per_gpu": held,
            "segment_storage_cap_bytes_per_gpu": args.segment_budget_bytes,
            "observed_free_tolerance_bytes": args.free_tolerance_mib * MIB,
        }
        level_output = args.output
        if sweep:
            level_output = args.output / f"pressure-{held_mib}-mib"
            level_output.mkdir()
        comparisons = []
        with gpus.pressure(held):
            for index in range(args.repeat):
                pair_path = level_output / f"pair-{index + 1}"
                pair_path.mkdir()
                order = tuple(recipes) if index % 2 == 0 else tuple(reversed(recipes))
                legs = {}
                for label in order:
                    output = pair_path / label
                    try:
                        preset = "balanced" if hash_comparison else args.transport_recipe if args.packet_control_addr else label
                        leg_env = dict(env, **preset_environment[preset])
                        if args.compare_hash_series:
                            leg_env["MGBFS_SEGMENT_HASH_SERIES"] = "1" if label == "snapshot" else "0"
                        if args.compare_hash_reuse:
                            leg_env["MGBFS_SEGMENT_HASH_REUSE"] = "1" if label == "reuse" else "0"
                        if compare_eager_tail and label == "tuned":
                            leg_env["MGBFS_SEGMENT_HASH_EAGER_TAIL"] = str(args.compare_hash_eager_tail)
                        use_packets = (label == "compact" or args.preset_packet_control_addr is not None
                                       or (label == "maxmemory" and args.maxmemory_packet_control_addr is not None))
                        if use_packets:
                            leg_env["MGBFS_SEGMENT_PACKET_CONTROL_ADDR"] = packet_addr
                        leg = _run_leg(graph, output, gpus, native, leg_env, recipes[label], request, pressure,
                                       transport_mode="compact" if use_packets else "exact",
                                       registry_evidence=args.registry_evidence)
                    except (OSError, ValueError, RuntimeError) as exc:
                        leg = {"error": str(exc)}
                    legs[label] = leg
                    _write(pair_path / f"{label}.summary.json", leg)
                comparison = compare_pair(*(legs[label] for label in labels), n,
                                          pressure["observed_free_tolerance_bytes"], labels=labels)
                if args.registry_evidence:
                    try:
                        comparison["registry_evidence"] = _export_registry_pair(
                            graph, pair_path, legs, comparison, resolve_indices=resolve_indices,
                        )
                    except (KeyError, OSError, ValueError, RuntimeError) as error:
                        comparison["registry_evidence"] = {"status": UNVERIFIED, "reason": str(error)}
                    print("REGISTRY_EVIDENCE", json.dumps(comparison["registry_evidence"]))
                if args.compare_hash_series:
                    snapshots = legs["snapshot"].get("hash_snapshot_installations_by_rank", [])
                    comparison["snapshot_observed"] = any(count for count in snapshots if count is not None)
                    if not comparison["snapshot_observed"]:
                        comparison = compare_pair(
                            legs["runs"], {"error": "snapshot was never installed"}, n,
                            pressure["observed_free_tolerance_bytes"], labels=labels,
                        ) | {"snapshot_observed": False}
                if args.compare_hash_reuse:
                    free_counts = legs["free"].get("hash_segment_reuses_by_rank", [])
                    reuse_counts = legs["reuse"].get("hash_segment_reuses_by_rank", [])
                    comparison["reuse_observed"] = free_counts == [0] * len(selected) and any(reuse_counts)
                    if comparison["comparable_context"] and not comparison["reuse_observed"]:
                        comparison = compare_pair(
                            legs["free"], {"error": "expected zero reuses in free and positive reuses in reuse"}, n,
                            pressure["observed_free_tolerance_bytes"], labels=labels,
                        ) | {"reuse_observed": False}
                if comparison["comparable_context"]:
                    for label, leg in legs.items():
                        point = {
                            "held_mib_per_gpu": held_mib,
                            "before_free_bytes_by_rank": [gpu["free_bytes"] for gpu in leg["before"]["devices"]],
                            "pair": index + 1,
                        }
                        if leg["status"] == "COMPLETE" and leg["checks"]["final_two_layers_full_exact_sets_no_duplicates"]:
                            observed[label].append(point)
                        elif _stopped_on_vram(leg):
                            limited[label].append({**point, "completed_depth": len(leg["layer_sizes"]) - 1})
                comparisons.append({"pair": index + 1, "execution_order": order, **comparison})
                _write(pair_path / "comparison.json", comparisons[-1])
        verified = "VERIFIED" if args.packet_control_addr or hash_comparison else WITHIN_BOUND
        aggregate = _aggregate_comparisons(comparisons, verified)
        result = {"schema": 1, "aggregate": aggregate, "pairs": comparisons}
        _write(level_output / "comparison.json", result)
        print(json.dumps(aggregate))
    if sweep:
        _write(args.output / "capacity-sweep.json", {
            "tested_pressure_mib_per_gpu": pressure_levels,
            "completions": observed,
            "vram_limited": limited,
            "capacity_brackets": {
                label: _capacity_bracket(observed[label], limited[label]) for label in labels
            },
            "highest_tested_pressure_with_verified_completion": {
                label: max(records, key=lambda row: row["held_mib_per_gpu"], default=None)
                for label, records in observed.items()
            },
        })



if __name__ == "__main__":
    main()
