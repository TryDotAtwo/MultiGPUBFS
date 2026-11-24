from contextlib import contextmanager
from copy import deepcopy
import ctypes as ct
import hashlib
import json
import sys

import pytest

from scripts import benchmark_segmented_presets as bench
from scripts import segmented_benchmark_gpu as gpu
from scripts.segmented_acceptance_oracle import mahonian, verify_output


def _leg(depth=78, *, layer_seconds=1, bfs_seconds=100):
    snapshot = {
        "devices": [{"ordinal": 2, "uuid": "GPU-a", "driver": "570", "free_bytes": 1234}],
        "topology": "a through NVLink",
    }
    return dict(
        runtime_identity={key: key for key in bench.BUILD_KEYS}, graph_digest="same-S13",
        pressure={"method": "cuda_warmed_allocation"}, request={"max_depth": None},
        communication_policy={"policy": {"NCCL_P2P_DISABLE": "0"}},
        before=deepcopy(snapshot), after=deepcopy(snapshot),
        checks=dict(all_completed_layer_counts_match_mahonian=True,
                    final_two_layers_full_exact_sets_no_duplicates=depth == 78),
        layer_sizes=mahonian(13)[:depth + 1], layer_seconds_by_rank=[[layer_seconds] * depth],
        bfs_seconds=bfs_seconds,
    )


def test_exactly_four_is_allowed_and_full_and_window_independently_exclude():
    reference = _leg()
    result = bench.compare_pair(reference, _leg(layer_seconds=4, bfs_seconds=401), 13, 0)
    assert result["full"]["status"] == bench.EXCLUDED
    heavy = result["s13_depths_34_44"]
    assert heavy["status"] == bench.WITHIN_BOUND
    assert heavy["slowdown"] == 4
    assert heavy["balanced"]["seconds"] == 11
    assert heavy["balanced"]["new_states"] == sum(mahonian(13)[34:45])
    assert heavy["balanced"]["generated_edges"] == 12 * sum(mahonian(13)[33:44])

    result = bench.compare_pair(reference, _leg(layer_seconds=4.01, bfs_seconds=400), 13, 0)
    assert result["full"]["status"] == bench.WITHIN_BOUND
    assert result["full"]["slowdown"] == 4
    assert result["s13_depths_34_44"]["status"] == bench.EXCLUDED


def test_capacity_completion_is_independent_of_throughput_evidence():
    reference = _leg(depth=20)
    reference.update(status="INCOMPLETE", reason="RESOURCE_10",
                     resource_detail_by_rank=[{"reason": "segment_budget"}])
    candidate = _leg()
    comparison = bench.compare_pair(reference, candidate, 13, 0)
    assert comparison["full"]["status"] == bench.UNVERIFIED
    assert comparison["capacity_at_matched_pressure"] == {
        "status": "VERIFIED_COMPLETION_AT_MATCHED_PRESSURE",
        "completed_preset": "maxmemory",
        "resource_limited_depth": 20,
    }
    for details in ([], [{"reason": "user_capacity"}], [{"reason": "cuda_allocation_test"}]):
        reference["resource_detail_by_rank"] = details
        comparison = bench.compare_pair(reference, candidate, 13, 0)
        assert comparison["capacity_at_matched_pressure"]["status"] == bench.UNVERIFIED
    reference["reason"] = "DEADLINE"
    comparison = bench.compare_pair(reference, candidate, 13, 0)
    assert comparison["capacity_at_matched_pressure"]["status"] == bench.UNVERIFIED


def test_short_prefix_and_missing_rank_timing_are_unverified():
    reference = _leg(depth=44)
    result = bench.compare_pair(reference, _leg(depth=45, layer_seconds=2), 13, 0)
    assert result["full"]["status"] == bench.UNVERIFIED
    assert result["s13_depths_34_44"]["slowdown"] == 2
    short = bench.compare_pair(reference, _leg(depth=43), 13, 0)
    assert short["s13_depths_34_44"]["status"] == bench.UNVERIFIED
    reference["layer_seconds_by_rank"].append([1] * 43)
    no_rank = bench.compare_pair(reference, _leg(depth=44), 13, 0)
    assert no_rank["s13_depths_34_44"]["slowdown"] is None


def test_s13_window_rejects_mismatched_committed_prefix():
    reference = _leg(depth=44)
    candidate = _leg(depth=45, layer_seconds=2)
    # Equal window work alone is insufficient: the search must start from the same prefix.
    candidate["layer_sizes"][17] += 1
    window = bench.compare_pair(reference, candidate, 13, 0)["s13_depths_34_44"]
    assert window["status"] == bench.UNVERIFIED
    assert window["slowdown"] is None
    assert "committed layer counts differ" in window["reason"]


def test_s13_window_requires_every_rank_and_uses_only_requested_transport_depths():
    run = _leg(depth=45)
    for snapshot in (run["before"], run["after"]):
        snapshot["devices"].append(dict(snapshot["devices"][0], ordinal=3, uuid="GPU-b"))
    assert bench._window(run, 13) is None
    run["layer_seconds_by_rank"].append([1] * 45)
    run["transport_high_water_observed_by_rank"] = [
        {
            "transport_by_depth": {
                str(depth): {"remote_submitted_nccl_bytes": 100, "remote_nccl_send_calls": 2}
                for depth in range(33, 46)
            },
            "global_sync_by_depth": {
                str(depth): {"legacy_chunk_votes": depth, "nccl_layer_count_gathers": 1}
                for depth in range(33, 46)
            },
        } for _ in range(2)
    ]
    window = bench._window(run, 13)
    assert window["remote_submitted_nccl_bytes"] == 2200
    for rank in window["global_sync_by_rank_and_depth"]:
        assert [entry["legacy_chunk_votes"] for entry in rank] == list(range(34, 45))

    assert window["remote_nccl_send_calls_per_new_state"] == 44 / window["new_states"]
    del run["transport_high_water_observed_by_rank"][1]["transport_by_depth"]["44"]
    del run["transport_high_water_observed_by_rank"][1]["global_sync_by_depth"]["44"]

    window = bench._window(run, 13)
    assert window["remote_submitted_nccl_bytes"] is None
    assert window["remote_nccl_send_calls"] is None
    assert window["global_sync_by_rank_and_depth"] is None


def test_s13_commit_clock_measures_a_contiguous_interval():
    reference = _leg(depth=44)
    reference["layer_committed_seconds_by_rank"] = [[2.0 * i for i in range(45)]]
    candidate = _leg(depth=44)
    candidate["layer_committed_seconds_by_rank"] = [[3.0 * i for i in range(45)]]
    heavy = bench.compare_pair(reference, candidate, 13, 0)["s13_depths_34_44"]
    assert heavy["slowdown"] == 1.5
    assert heavy["balanced"]["seconds"] == 22.0
    assert heavy["balanced"]["seconds_from_bfs_start_through_last_commit"] == 88.0
    candidate["layer_committed_seconds_by_rank"][0][42] = 0
    assert bench.compare_pair(reference, candidate, 13, 0)["s13_depths_34_44"]["status"] == bench.UNVERIFIED
    candidate.pop("layer_committed_seconds_by_rank")
    assert bench.compare_pair(reference, candidate, 13, 0)["s13_depths_34_44"]["status"] == bench.UNVERIFIED


def test_slow_verified_prefix_can_exclude_without_promising_a_full_timing():
    reference = _leg()
    candidate = _leg(depth=40, bfs_seconds=401)
    candidate.update(status="INCOMPLETE", reason="DEADLINE")
    full = bench.compare_pair(reference, candidate, 13, 0)["full"]
    assert full["status"] == bench.UNVERIFIED and full["slowdown"] is None
    assert full["exclusion"]["status"] == bench.EXCLUDED
    assert full["exclusion"]["slowdown_lower_bound"] == 4.01
    comparison = bench.compare_pair(reference, candidate, 13, 0)
    aggregate = bench._aggregate_comparisons([comparison], bench.WITHIN_BOUND)
    assert aggregate["full"] == bench.EXCLUDED
    assert aggregate["s13_depths_34_44"] == comparison["s13_depths_34_44"]["status"]


    candidate["bfs_seconds"] = 400
    assert "exclusion" not in bench.compare_pair(reference, candidate, 13, 0)["full"]
    candidate.update(bfs_seconds=401, reason="PROFILE_LAYER_LIMIT")
    assert "exclusion" not in bench.compare_pair(reference, candidate, 13, 0)["full"]


@pytest.mark.parametrize("change", [
    lambda r: r["runtime_identity"].update(cuda_library_sha256="another artifact"),
    lambda r: r.update(graph_digest="other graph"),
    lambda r: r["pressure"].update(method="segment_storage_cap"),
    lambda r: r["request"].update(max_depth=80),
    lambda r: r["communication_policy"].update(policy={"NCCL_P2P_DISABLE": "1"}),
    lambda r: r["before"].update(topology="PCIe"),
    lambda r: r["after"]["devices"][0].update(uuid="GPU-b"),
    lambda r: r["after"]["devices"][0].update(free_bytes=1111),
    lambda r: r["checks"].update(all_completed_layer_counts_match_mahonian=False),
])
def test_different_context_cannot_promise_speed(change):
    reference = _leg()
    candidate = deepcopy(reference)
    change(candidate)
    result = bench.compare_pair(reference, candidate, 13, 32)
    assert not result["comparable_context"]
    for scope in ("full", "s13_depths_34_44"):
        assert result[scope]["status"] == bench.UNVERIFIED
        assert result[scope]["slowdown"] is None


def test_oracle_detects_wrong_counts_and_duplicate_terminal_rows(tmp_path):
    states = tmp_path / "states.json"
    states.write_text('{"current":[[2,1,0]],"previous_small":[[2,0,1],[1,2,0]]}')
    report = dict(status="COMPLETE", layer_sizes=[1, 2, 2, 1])
    assert verify_output(3, report, states)["final_two_layers_full_exact_sets_no_duplicates"]
    states.write_text('{"current":[[2,1,0]],"previous_small":[[2,0,1],[2,0,1]]}')
    with pytest.raises(ValueError, match="sets or multiplicities"):
        verify_output(3, report, states)
    with pytest.raises(ValueError, match="layer mismatch"):
        verify_output(3, dict(status="INCOMPLETE", layer_sizes=[1, 3]), states)

    prefix = dict(status="INCOMPLETE", layer_sizes=[1, 2])
    states.write_text('{"current":[[0,2,1],[1,0,2]],"previous_small":[[0,1,2]]}')
    checked = verify_output(3, prefix, states)
    assert checked["retained_samples_match_committed_layers_no_duplicates"]
    assert not checked["final_two_layers_full_exact_sets_no_duplicates"]
    states.write_text('{"current":[[0,2,1],[2,0,1]],"previous_small":[[0,1,2]]}')
    with pytest.raises(ValueError, match="Invalid sampled state for depth 1"):
        verify_output(3, prefix, states)


def test_native_metrics_are_scoped_and_do_not_invent_missing_values():
    rank = dict(transport_trace=True, exact_exchange=True, remote_records_sent=48,
                storage_allocated_bytes=100, bitmap_bytes=20, queues_allocated_bytes=80)
    part = dict(plan=dict(elements=3, state_bytes=1, parent_transport=False), source_retries=1)
    report = dict(layer_sizes=[1, 2, 2, 1], segment_memory_by_rank=[rank])
    actual = bench.metrics(3, report, [part])
    assert actual["global_completed_unique"] == 6 and actual["global_peak_frontier"] == 2
    assert actual["storage_plus_bitmap_bytes_per_completed_unique_at_end"] == 20
    assert actual["remote_candidate_payload_bytes_per_completed_unique"] == 256
    assert actual["sum_of_rank_storage_request_high_water_bytes"] is None
    rank["transport_trace"] = False
    assert bench.metrics(3, report, [part])["remote_candidate_records"] is None


@pytest.mark.parametrize("compare_transport, compact_presets, registry_export", [
    (None, False, False), ("balanced", False, False), ("maxmemory", False, False),
    (None, True, False), (None, False, True),
])
def test_paired_cli_reaches_explicit_run_graph_configs(monkeypatch, tmp_path, compare_transport, compact_presets,
                                                     registry_export):
    from multigpubfs import launch, native_distribution

    calls, allocations = [], []

    class FakeGpuProbe:
        def __init__(self, ordinals):
            assert ordinals == [2, 0]

        def snapshot(self):
            return {"devices": [
                {"ordinal": ordinal, "uuid": f"GPU-{ordinal}", "free_bytes": 1000, "driver": "535.14"}
                for ordinal in (2, 0)
            ], "topology": "NVLink"}

        @contextmanager
        def pressure(self, held):
            allocations.append(held)
            yield

        @contextmanager
        def sample_memory(self, snapshot):
            yield {"baseline_bytes_by_rank": [10, 10], "samples": 10,
                   "peak_bytes_by_rank": [31, 24] if len(calls) == 0 else [20, 19]}

    def fake_run(graph, output, **options):
        calls.append(options)
        output.mkdir()
        (output / "states.json").write_text('{"current":[[2,1,0]],"previous_small":[[2,0,1],[1,2,0]]}')
        ranks, paths = [], []
        for index, device in enumerate(options["devices"]):
            name = f"rank-{index}.json"
            (output / name).write_text(json.dumps({"plan": {}, "layer_seconds": [1, 1, 1]}))
            paths.append(name)
            ranks.append({
                "device": device, "segment_strategy": options["segment_strategy"],
                "resident_codec": "pack4-v1" if options["bitmap_pack4"] else "unpacked-soa",
                "remote_candidate_payload_bytes": 80,
                "remote_submitted_nccl_bytes": 88,
                "remote_submitted_nccl_metadata_bytes": 8,
                "remote_nccl_send_calls": 3,
                "exact_exchange": options["_native_env"].get("MGBFS_SEGMENT_EXACT_EXCHANGE") == "1",
            })
            if "MGBFS_SEGMENT_PACKET_CONTROL_ADDR" in options["_native_env"]:
                ranks[-1]["packet_transport"] = {"submitted_packets": 2, "submitted_bytes": 128,
                                                  "split_destination_windows": index,
                                                  "pinned_route_metadata": index == 0,
                                                  "hub_peak_active_receive_credits": 1 if index == 0 else None,
                                                  "submitted_payload_bytes": 112}
                ranks[-1].update(remote_candidate_payload_bytes=112, remote_submitted_nccl_bytes=128,
                                 remote_submitted_nccl_metadata_bytes=16, remote_nccl_send_calls=2)
        result = dict(
            devices=options["devices"], graph_digest=graph.digest(), rank_receipts=paths,
            segment_memory_by_rank=ranks, layer_sizes=[1, 2, 2, 1], status="COMPLETE",
            reason="FRONTIER_EXHAUSTED", bfs_seconds=3, setup_seconds=0.5,
            runtime_identity={key: key for key in bench.BUILD_KEYS},
        )
        if registry_export:
            current = json.loads((output / "states.json").read_bytes())
            witness = json.dumps(dict(current, schema=1, graph_digest=graph.digest(), depth=3)).encode()
            (output / "terminal_states.json").write_bytes(witness)
            plan = {"shards": options["shards"], "capacity": 8192, "queue_capacity": 512,
                    "batch": 128, "state_bytes": 1}
            for rank, values in enumerate(ranks):
                values.update(rank=rank, owner_lanes=1, actual_queue_capacity=512,
                              banks=2 if options["bitmap_pack4"] else 1,
                              compact_positions=False, configured_segment_rows=1024, batch_hash_prefix_probe=False,
                              admitted_batch=128, rows_per_bank=1024, peak_live_rows=2)
            result.update(
                backend="EXPERIMENTAL_PERMUTATION_BITMAP" if options["bitmap_pack4"] else "EXPERIMENTAL_SEGMENTED24",
                states_sha256=hashlib.sha256((output / "states.json").read_bytes()).hexdigest(),
                terminal_states_sha256=hashlib.sha256(witness).hexdigest(), size_profile_count=0,
                runtime_identity={key: "a" * 64 for key in bench.BUILD_KEYS}, plan=plan, rank_plans=[plan, plan],
            )
            (output / "report.json").write_text(json.dumps(result))
            (output / "launch.json").write_text(json.dumps({
                "devices": options["devices"], "graph_digest": graph.digest(), "additional_reserve_bytes": 678,
                "inventory": [{"device": device, "compute_major": 8, "compute_minor": 6, "sm_count": 28,
                               "total_bytes": 12 << 30, "free_bytes": 123 + device} for device in options["devices"]],
            }))
        return result

    monkeypatch.setattr(bench, "BenchmarkGpus", FakeGpuProbe)
    monkeypatch.setattr(native_distribution, "native_runtime", lambda _: ("native", {
        "MGBFS_SEGMENT_TEST_HASH_BITS": "1",
    }))
    monkeypatch.setattr(launch, "run_graph", fake_run)
    if registry_export:
        from multigpubfs.profile_registry import read_profile_registry, verify_publication

        monkeypatch.setattr(launch, "_native_key_info", lambda *args: {"graph_profile_terminal_evidence_v1": True})
        monkeypatch.setattr(sys, "argv", [
            "benchmark", "--output", str(tmp_path / "run"), "--n", "3", "--devices", "2,0",
            "--registry-evidence",
        ])
        bench.main()
        assert len(calls) == 2 and allocations == [0]
        assert all(call["_native_env"]["MGBFS_PROFILE_TERMINAL_EVIDENCE_DEPTH"] == "3" for call in calls)
        pair = tmp_path / "run" / "pair-1"
        summary = json.loads((pair / "comparison.json").read_text())
        exported = summary["registry_evidence"]
        assert exported["status"] == "EXPORTED"
        catalog = pair / exported["catalog"]
        verify_publication(catalog, pair / "profile" / "evidence")
        entry = read_profile_registry(catalog, sha256=exported["catalog_sha256"])["entries"][0]
        assert entry["capacity_verification"] == "LOWER_PHYSICAL_PEAK"
        assert entry["hardware"]["driver_versions"] == ["535.14", "535.14"]
        balanced = entry["profiles"]["balanced"]
        assert balanced["peak_vram_bytes_by_rank"] == [31, 24]
        assert balanced["rank_geometry"][0]["free_bytes_at_admission"] == 125
        assert balanced["rank_geometry"][1]["additional_reserve_bytes"] == 678
        return
    extra = []
    if compare_transport:
        extra = ["--packet-control-addr", "127.0.0.1:29761", "--transport-recipe", compare_transport]
    if compact_presets:
        extra = ["--preset-packet-control-addr", "127.0.0.1:29761"]
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--output", str(tmp_path / "run"), "--n", "3", "--devices", "2,0",
        "--repeat", "2", "--segment-budget-bytes", "1000000", *extra,
        "--maxmemory-queue-cap", "8192", "--maxmemory-owner-lanes", "4",
        "--balanced-hash-probe", "prefix", "--hash-timing",
    ])
    bench.main()
    assert all(call["_native_env"]["MGBFS_SEGMENT_HASH_TIMING"] == "1"
               and call["_native_env"]["MGBFS_SEGMENT_HASH_TIMING_MIN_DEPTH"] == "1"
               and call["_native_env"]["MGBFS_SEGMENT_HASH_TIMING_MAX_DEPTH"] == "3"
               for call in calls)
    prefix = [call["_native_env"].get("MGBFS_SEGMENT_HASH_PREFIX_PROBE") for call in calls]
    if compare_transport is None:
        assert prefix == ["1", None, None, "1"]
    else:
        assert prefix == (["1"] * 4 if compare_transport == "balanced" else [None] * 4)
    if compare_transport:
        pack4 = compare_transport == "maxmemory"
        expected_backend = "permutation_bitmap" if pack4 else "segmented24"
        assert all(call["memory_backend"] == expected_backend and call["bitmap_pack4"] == pack4 for call in calls)
        assert all(call["_native_env"].get("MGBFS_SEGMENT_QUEUE_CAP") == ("8192" if pack4 else None)
                   for call in calls)
        assert [call["_native_env"].get("MGBFS_SEGMENT_PACKET_CONTROL_ADDR") for call in calls] == [
            None, "127.0.0.1:29761", "127.0.0.1:29761", None,
        ]
        compact = json.loads((tmp_path / "run" / "pair-1" / "compact.summary.json").read_text())
        assert compact["metrics"]["packet_bytes_including_shard_descriptors"] == 256
        assert compact["metrics"]["remote_payload_transfers"] == 4
        assert compact["metrics"]["packet_split_destination_windows"] == 1
        assert compact["metrics"]["packet_hub_peak_active_receive_credits"] == 1
        assert compact["metrics"]["packet_pinned_route_metadata_by_rank"] == [True, False]
        assert compact["metrics"]["remote_candidate_application_payload_bytes"] == 224
        assert compact["metrics"]["global_committed_new_states"] == 5
        assert compact["metrics"]["remote_payload_transfers_per_committed_new_state"] == 4 / 5
        assert compact["metrics"]["remote_payload_bytes_per_committed_new_state"] == 224 / 5
        assert compact["metrics"]["remote_submitted_nccl_bytes_per_committed_new_state"] == 256 / 5
        assert compact["metrics"]["remote_submitted_nccl_metadata_bytes"] == 32
        assert compact["metrics"]["remote_nccl_send_calls_per_committed_new_state"] == 4 / 5
        exact = json.loads((tmp_path / "run" / "pair-1" / "exact.summary.json").read_text())
        assert exact["metrics"]["remote_candidate_application_payload_bytes"] == 160
        assert exact["metrics"]["packet_split_destination_windows"] is None
        assert exact["metrics"]["packet_hub_peak_active_receive_credits"] is None
        assert exact["metrics"]["remote_submitted_nccl_bytes_per_committed_new_state"] == 176 / 5
        assert exact["metrics"]["remote_submitted_nccl_metadata_bytes"] == 16
        assert exact["metrics"]["remote_nccl_send_calls_per_committed_new_state"] == 6 / 5
    else:
        assert [call["memory_backend"] for call in calls] == [
            "segmented24", "permutation_bitmap", "permutation_bitmap", "segmented24",
        ]
        assert [call["bitmap_pack4"] for call in calls] == [False, True, True, False]
        expected_address = "127.0.0.1:29761" if compact_presets else None
        assert all(call["_native_env"].get("MGBFS_SEGMENT_PACKET_CONTROL_ADDR") == expected_address
                   for call in calls)
        assert [call["_native_env"].get("MGBFS_SEGMENT_QUEUE_CAP") for call in calls] == [None, "8192", "8192", None]
        assert [call["_native_env"].get("MGBFS_SEGMENT_OWNER_LANES") for call in calls] == [None, "4", "4", None]
    assert all(call["autotune"] is False and call["shards"] == 2 for call in calls)
    assert all(call["_native_env"]["MGBFS_SEGMENT_TEST_BYTES"] == "1000000" for call in calls)
    assert all("MGBFS_SEGMENT_TEST_HASH_BITS" not in call["_native_env"] for call in calls)
    actual = json.loads((tmp_path / "run" / "comparison.json").read_text())
    assert actual["aggregate"] == {"full": "VERIFIED" if compare_transport else bench.WITHIN_BOUND,
                                   "s13_depths_34_44": bench.UNVERIFIED}

    if not compare_transport:
        monkeypatch.setattr(sys, "argv", [
            "benchmark", "--output", str(tmp_path / "sweep"), "--n", "3", "--devices", "2,0",
            "--pressure-mib", "32", "--pressure-mib", "16",
        ])
        bench.main()
        sweep = json.loads((tmp_path / "sweep" / "capacity-sweep.json").read_text())
        assert allocations == [0, 16 * bench.MIB, 32 * bench.MIB]
        assert sweep["tested_pressure_mib_per_gpu"] == [16, 32]
        assert all(item["held_mib_per_gpu"] == 32
                   for item in sweep["highest_tested_pressure_with_verified_completion"].values())


def test_physical_holder_warms_every_gpu_and_unwinds_on_exit(monkeypatch):
    probe = gpu.BenchmarkGpus.__new__(gpu.BenchmarkGpus)
    probe.devices = {2: ct.c_int(2), 0: ct.c_int(0)}
    calls = []

    def driver_call(name, *args):
        calls.append((name, args))

    monkeypatch.setattr(probe, "_call", driver_call)
    with probe.pressure(1024):
        assert [args[1].value for name, args in calls if name == "cuDevicePrimaryCtxRetain"] == [2, 0]
        assert [args[2].value for name, args in calls if name == "cuMemsetD8_v2"] == [1024, 1024]
        assert [name for name, _ in calls].count("cuCtxSynchronize") == 2
        assert "cuMemFree_v2" not in [name for name, _ in calls]
    assert [args[0].value for name, args in calls if name == "cuDevicePrimaryCtxRelease_v2"] == [0, 2]
    assert [name for name, _ in calls].count("cuMemFree_v2") == 2

def test_capacity_bracket_requires_less_free_vram_on_every_rank():
    complete = {"held_mib_per_gpu": 16, "before_free_bytes_by_rank": [1000, 900]}
    uneven = {"held_mib_per_gpu": 32, "before_free_bytes_by_rank": [800, 950]}
    tight = {"held_mib_per_gpu": 48, "before_free_bytes_by_rank": [800, 700]}
    loose = {"held_mib_per_gpu": 64, "before_free_bytes_by_rank": [650, 600]}
    assert bench._capacity_bracket([complete], [uneven, loose, tight]) == {
        "complete": complete, "vram_limited": tight,
    }
    assert bench._capacity_bracket([complete], [uneven])["vram_limited"] is None

def test_s13_hash_series_window_requires_actual_snapshot_lookups():
    reference = _leg()
    candidate = _leg(layer_seconds=2)
    labels = ("runs", "snapshot")
    candidate["hash_timing_by_rank"] = [None, {"tiles": 9, "tiles_with_snapshot": 0}]
    result = bench.compare_pair(reference, candidate, 13, 0, labels=labels)
    assert result["full"]["status"] == "VERIFIED"
    assert result["s13_depths_34_44"]["status"] == bench.UNVERIFIED

    candidate["hash_timing_by_rank"][1]["tiles_with_snapshot"] = 3
    result = bench.compare_pair(reference, candidate, 13, 0, labels=labels)
    assert result["s13_depths_34_44"]["status"] == "VERIFIED"
    assert result["s13_depths_34_44"]["slowdown"] == 2
