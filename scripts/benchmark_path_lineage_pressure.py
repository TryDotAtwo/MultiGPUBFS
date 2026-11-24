"""Check a physical window traversal against a rank-zero RSS budget and disk CSR size."""

import argparse
import json
from pathlib import Path

from multigpubfs import GraphDefinition, open_session
from scripts.segmented_acceptance_oracle import mahonian


def increasing_counter_graph(degree):
    width = degree + 3
    counter, one, clean = range(degree, width)
    generators = []
    for at in range(degree - 1):
        rows = list(range(width))
        rows[at], rows[at + 1] = rows[at + 1], rows[at]
        matrix = [int(column == row) for row in rows for column in range(width)]
        matrix[counter * width + one] = 1
        if at:
            matrix[clean * width + clean] = 0
        generators.append((matrix, 0))
    return GraphDefinition.matrix(width, 1, generators, list(range(degree)) + [0, 1, 1])


def clean_target(graph, degree, length):
    target = list(graph.start)
    if length % 2:
        target[0], target[1] = target[1], target[0]
    target[degree] = length
    return target


def expected_csr(degree, max_length, ranks):
    counts = mahonian(degree)
    diameter = len(counts) - 1
    saturated = sum(counts) // 2
    states = [sum(counts[depth % 2:depth + 1:2]) for depth in range(min(max_length, diameter) + 1)]
    total = sum(states) + max(0, max_length - diameter) * saturated
    final = saturated if max_length > diameter else states[-1]
    # The all-zero word is the only clean word at each depth. At depth 1
    # its permutation is unique. Starting at depth 2 the same permutation
    # is also reachable by inserting two copies of another generator.
    total += max(0, max_length - 1)
    final += int(max_length >= 2)
    # Each distinct word ending at a child contributes through predecessors.
    # Here every edge raises the counter, and each layer contains exactly
    # the permutations reachable at that inversion parity within that depth.
    edges = (degree - 1) * (total - final)
    without_owner_sentinels = edges * 16 + (total - 1) * 8
    return without_owner_sentinels + max_length * 8, without_owner_sentinels + ranks * max_length * 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="artifact directory for this run")
    parser.add_argument("--executable", type=Path, help="matching native CUDA executable")
    parser.add_argument("--lineage-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--rank0-budget-bytes", type=int, required=True)
    parser.add_argument("--cache-bytes", type=int, default=8 << 20)
    args = parser.parse_args()
    try:
        import resource
    except ImportError:
        parser.error("this benchmark requires Linux RSS accounting")
    if args.output is None:
        from multigpubfs.artifact_paths import new_benchmark_directory

        args.output = new_benchmark_directory()
    if not (3 <= args.degree <= 12 and 0 < args.length < 1 << 32 and 0 < args.gpus <= 128
            and (64 << 10) <= args.cache_bytes < args.rank0_budget_bytes):
        parser.error("invalid graph or memory bounds")
    low, high = expected_csr(args.degree, args.length, args.gpus)
    if low < 2 * args.rank0_budget_bytes:
        parser.error(f"configured workload produces only {low}–{high} CSR bytes; increase degree or length")
    graph = increasing_counter_graph(args.degree)
    with open_session(gpus=args.gpus, tune="off", executable=args.executable) as session:
        result = session.solve(
            graph, targets=[graph.start, clean_target(graph, args.degree, args.length)],
            goal_mode="window", direction="forward",
            extra_steps=args.length, output=args.output, lineage_dir=args.lineage_dir,
            lineage_cache_bytes=args.cache_bytes,
        )
    paths = [{"target_index": row["target_index"], "generator_ids": row["generator_ids"]}
             for row in result.iter_paths()]
    receipt = result.report
    coordinator_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    worker_peaks = receipt.get("native_process_peak_rss_bytes_by_rank") or []
    worker_peak_rss = worker_peaks[0] if worker_peaks else None
    combined_peak_bound = (coordinator_peak_rss + worker_peak_rss
                           if worker_peak_rss else None)
    csr = receipt["lineage_committed_csr_bytes"]
    checks = {
        "exact_paths": paths == [{"target_index": 0, "generator_ids": []},
                                 {"target_index": 1, "generator_ids": [0] * args.length}],
        "full_coverage": (result.status == "COMPLETE" and receipt["d_min"] == 0
                          and receipt["completed_through_length"] == args.length),
        "csr_expected_size": low <= csr <= high,
        "cache_used_within_budget": 0 < receipt["lineage_cache_peak_bytes"] <= args.cache_bytes,
        "rank0_rss_within_budget": (combined_peak_bound is not None
                                    and combined_peak_bound <= args.rank0_budget_bytes),
        "csr_at_least_twice_rank0_budget": csr >= 2 * args.rank0_budget_bytes,
    }
    from multigpubfs.cluster import current_cluster

    remote_bytes = receipt.get("lineage_remote_read_bytes")
    if current_cluster() is not None:
        checks["reads_indexed_blocks_at_owners"] = (
            receipt.get("lineage_read_mode") == "rank_owned" and remote_bytes is not None
            and 0 < remote_bytes < csr
        )
    artifact = {"checks": checks, "csr_bytes": csr,
                "rank0_coordinator_peak_rss_bytes": coordinator_peak_rss,
                "rank0_native_worker_peak_rss_bytes": worker_peak_rss,
                "rank0_combined_rss_upper_bound_bytes": combined_peak_bound,
                "rank0_rss_budget_bytes": args.rank0_budget_bytes,
                "lineage_cache_peak_bytes": receipt["lineage_cache_peak_bytes"],
                "lineage_remote_read_bytes": remote_bytes,
                "note": "Sum of Linux process RSS high-water marks for the coordinator and native rank 0"}
    evidence = args.output / "lineage-pressure.json"
    evidence.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(f"window lineage acceptance failed: {json.dumps(artifact)}")
    print(json.dumps(artifact))


if __name__ == "__main__":
    main()
