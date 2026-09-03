"""Validate saved primitive-gate evidence, not production multi-rank BFS."""
import argparse
import json
import re
from pathlib import Path

def verify_inventory(text, test, tool):
    """Frozen fixture names: a successful subset must not certify a full gate."""
    expected = {
        "generate": {
            "allocation_query_matches_frozen_geometry_and_c_abi",
            "invalid_variant_and_legacy_grid_are_rejected_before_output_write",
            "large_batch_crosses_old_grid_y_boundary",
            "tensor_generation_then_hash_matches_full_state_oracle_without_intermediate_host_sync",
        },
        "hash": {
            "allocation_query_matches_frozen_hash_buffers_and_c_abi",
            "tensor_hash_matches_cpu_for_padding_unsigned_bytes_seeds_and_tail_counts",
        },
        "route": {"cub_routes_all_128_bits_stably_and_optionally_deduplicates"},
        "owner": {
            "owner_merges_old_layers_and_cross_epoch_duplicates_then_poison_on_overflow",
            "owner_accepts_max_hash_once_without_host_sync_and_rejects_bad_epochs_and_runs",
        },
        "pipeline": {"generated_routed_owner_survivors_match_full_state_layers_for_both_prededup_modes"},
        "materialize": {"appends_in_source_order_and_rejects_whole_invalid_batches"},
        "ping_pong": {
            "failure_with_both_slots_in_flight_is_sticky_and_drains_on_drop",
            "generation_variants_small_feedback",
            "reused_slots_and_partial_tails_preserve_every_layer",
        },
        "dense_device": {"capacity_failure_poisoning_does_not_publish_a_partial_layer"},
    }[test]
    if test == "ping_pong":
        if tool != "racecheck": expected.add("generation_variants_preserve_full_layers")
        if tool == "plain": expected.add("full_u4_pipelined_sweep")
    if test == "dense_device":
        expected.add("gpu_feedback_small_full_depth_sanitizer_fixture" if tool == "racecheck"
                     else "gpu_feedback_exhausts_exact_layers_without_cpu_supplied_frontiers")
    actual = re.findall(r"^test ([a-zA-Z0-9_]+) \.\.\. ok\s*$", text, re.MULTILINE)
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("FIXTURE_INVENTORY_MISMATCH")

def verify_summary(summary, source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}",source_commit):
        raise ValueError("IMMUTABLE_SOURCE_REQUIRED")
    if summary.get("status")!="PASS_PRIMITIVE_GATE" or summary.get("source_commit")!=source_commit:
        raise ValueError("INCOMPLETE_OR_WRONG_SOURCE")
    gpus=summary["gpus"]
    if len(gpus)!=2 or sorted(g["index"] for g in gpus)!=[0,1] or len({g["uuid"] for g in gpus})!=2:
        raise ValueError("TWO_DISTINCT_DEVICES_REQUIRED")
    if any(g["name"] not in ("Tesla T4","NVIDIA Tesla T4","NVIDIA T4") for g in gpus):
        raise ValueError("T4_REQUIRED")
    tests=("generate","hash","route","owner","pipeline","materialize","dense_device","ping_pong")
    tools=("plain","memcheck","racecheck","initcheck","synccheck")
    expected={(g["uuid"],test,tool) for g in gpus for test in tests for tool in tools}
    seen=set()
    indices={g["uuid"]:g["index"] for g in gpus}
    entries=[]
    for row in summary["results"]:
        key=(row["gpu"],row["test"],row["tool"])
        if key in seen or key not in expected or row["status"]!="PASS":
            raise ValueError("DUPLICATE_UNKNOWN_OR_FAILED_RESULT")
        seen.add(key)
        entries.append((f"gpu{indices[key[0]]}-{key[1]}-{key[2]}.log",key[2]))
    if seen!=expected:
        raise ValueError("MISSING_RESULT")
    return sorted(entries)

def verify_log(text, tool):
    if not re.search(r"test result: ok\. [1-9][0-9]* passed; 0 failed; 0 ignored;",text) or "test result: FAILED" in text:
        raise ValueError("RUST_TESTS_INCOMPLETE")
    if tool=="plain": return
    if tool not in ("memcheck","racecheck","initcheck","synccheck"):
        raise ValueError("UNKNOWN_SANITIZER")
    errors=re.findall(r"ERROR SUMMARY: ([0-9]+) errors",text)
    races=re.findall(r"RACECHECK SUMMARY: ([0-9]+) hazards displayed \(([0-9]+) errors, ([0-9]+) warnings\)",text)
    if any(int(x)!=0 for x in errors) or any(any(int(x)!=0 for x in row) for row in races):
        raise ValueError("SANITIZER_ERRORS")
    if tool=="racecheck" and not races or tool!="racecheck" and not errors:
        raise ValueError("SANITIZER_SUMMARY_MISSING")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--source",required=True)
    args=p.parse_args()
    summary=json.loads((args.directory/"summary.json").read_text(encoding="utf-8"))
    entries=verify_summary(summary,args.source)
    for name,tool in entries:
        log=(args.directory/name).read_text(encoding="utf-8")
        verify_log(log,tool)
        verify_inventory(log,name.split("-")[1],tool)
    query=(args.directory/"allocation-queries.log").read_text(encoding="utf-8")
    if "100% tests passed, 0 tests failed out of 1" not in query:
        raise ValueError("QUERY_CTEST_INCOMPLETE")
    print(json.dumps(dict(status="VERIFIED_PRIMITIVE_GATE",source=args.source,combinations=len(entries))))

if __name__=="__main__":
    main()
