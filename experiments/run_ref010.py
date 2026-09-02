"""Reproduce REF-010 distributed bidirectional owner-routing measurements."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

from multigpubfs.distributed_bidirectional import (
    distributed_bidirectional_breadth_first_search,
)
from multigpubfs.implicit import labeled_breadth_first_search, replay_moves


def shortest_distance(outgoing, start, target):
    distance = {start: 0}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == target:
            return distance[state]
        for _, child in outgoing[state]:
            if child not in distance:
                distance[child] = distance[state] + 1
                queue.append(child)
    return None


def directed_validation():
    edges = tuple((u, v) for u in range(4) for v in range(4) if u != v)
    configurations = tuple(
        (world_size, policy)
        for world_size in (1, 2, 4)
        for policy in ("smaller_frontier", "alternating")
    )
    summary = {
        f"p{world_size}_{policy}": {
            "pairs": 0,
            "distance_mismatches": 0,
            "replay_failures": 0,
            "round_accounting_failures": 0,
        }
        for world_size, policy in configurations
    }
    for mask in range(1 << len(edges)):
        outgoing = {state: [] for state in range(4)}
        incoming = {state: [] for state in range(4)}
        for bit, (parent, child) in enumerate(edges):
            if mask & (1 << bit):
                move = f"{parent}>{child}"
                outgoing[parent].append((move, child))
                incoming[child].append((move, parent))
        outgoing = {state: tuple(items) for state, items in outgoing.items()}
        incoming = {state: tuple(items) for state, items in incoming.items()}
        for start in range(4):
            for target in range(4):
                if start == target:
                    continue
                expected = shortest_distance(outgoing, start, target)
                for world_size, policy in configurations:
                    key = f"p{world_size}_{policy}"
                    result = distributed_bidirectional_breadth_first_search(
                        lambda state, g=outgoing: g[state],
                        lambda state, g=incoming: g[state],
                        start=start,
                        target=target,
                        owner=lambda state, p=world_size: state % p,
                        world_size=world_size,
                        expansion_policy=policy,
                    )
                    item = summary[key]
                    item["pairs"] += 1
                    item["distance_mismatches"] += result.distance != expected
                    cursor = start
                    for move in result.moves:
                        cursor = next(
                            (
                                child
                                for label, child in outgoing[cursor]
                                if label == move
                            ),
                            None,
                        )
                        if cursor is None:
                            break
                    item["replay_failures"] += bool(result.found and cursor != target)
                    for metrics in result.rounds:
                        valid = (
                            metrics.generated_transitions
                            == metrics.source_duplicate_occurrences
                            + metrics.source_unique_candidates
                            and metrics.source_unique_candidates
                            == metrics.local_after_source_dedup
                            + metrics.remote_after_source_dedup
                            == metrics.owner_duplicate_occurrences
                            + metrics.owner_unique_candidates
                            and metrics.owner_unique_candidates
                            == metrics.already_visited + metrics.newly_discovered
                        )
                        item["round_accounting_failures"] += not valid
    return {
        "graphs": 1 << len(edges),
        "ordered_distinct_pairs_per_configuration": 49152,
        "configurations": summary,
    }


def adjacent_transitions(state):
    result = []
    for index in range(len(state) - 1):
        child = list(state)
        child[index], child[index + 1] = child[index + 1], child[index]
        result.append((index, tuple(child)))
    return tuple(result)


def lehmer_rank(permutation):
    rank = 0
    for index, value in enumerate(permutation):
        smaller = sum(other < value for other in permutation[index + 1 :])
        rank = rank * (len(permutation) - index) + smaller
    return rank


def mix64(value):
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def s8_routing_sweep():
    start = tuple(range(8))
    complete = labeled_breadth_first_search(adjacent_transitions, sources=(start,))
    selected_depths = (2, 8, 14, 20, 28)
    rows = []
    for depth in selected_depths:
        target = complete.frontiers[depth][0]
        for strategy in ("direct", "mixed"):
            for world_size in (1, 2, 4, 8):
                def owner(state, strategy=strategy, world_size=world_size):
                    rank = lehmer_rank(state)
                    key = rank if strategy == "direct" else mix64(rank)
                    return key % world_size

                result = distributed_bidirectional_breadth_first_search(
                    adjacent_transitions,
                    adjacent_transitions,
                    start=start,
                    target=target,
                    owner=owner,
                    world_size=world_size,
                    expansion_policy="alternating",
                )
                assert result.distance == depth
                assert replay_moves(
                    start,
                    result.moves,
                    lambda state, move: adjacent_transitions(state)[move][1],
                ) == target
                rounds = result.rounds
                generated = sum(item.generated_transitions for item in rounds)
                source_unique = sum(item.source_unique_candidates for item in rounds)
                remote_generated = sum(
                    item.remote_generated_occurrences for item in rounds
                )
                remote_after = sum(item.remote_after_source_dedup for item in rounds)
                source_duplicates = sum(
                    item.source_duplicate_occurrences for item in rounds
                )
                owner_duplicates = sum(
                    item.owner_duplicate_occurrences for item in rounds
                )
                intersections = sum(item.intersections_discovered for item in rounds)
                first_intersection_round = next(
                    index
                    for index, item in enumerate(rounds, start=1)
                    if item.intersections_discovered
                )
                nonempty_skews = []
                for item in rounds:
                    average = item.frontier_states / world_size
                    nonempty_skews.append(max(item.frontier_by_owner) / average)
                peak_round = max(rounds, key=lambda item: item.frontier_states)
                peak_average = peak_round.frontier_states / world_size
                large_skews = [
                    max(item.frontier_by_owner) / (item.frontier_states / world_size)
                    for item in rounds
                    if item.frontier_states >= 128
                ]
                rows.append(
                    {
                        "depth": depth,
                        "target": "".join(map(str, target)),
                        "strategy": strategy,
                        "world_size": world_size,
                        "rounds": len(rounds),
                        "generated": generated,
                        "remote_generated": remote_generated,
                        "source_duplicates_removed": source_duplicates,
                        "source_unique": source_unique,
                        "remote_after_source_dedup": remote_after,
                        "owner_duplicates_removed": owner_duplicates,
                        "intersections_discovered": intersections,
                        "first_intersection_round": first_intersection_round,
                        "rounds_after_first_intersection": len(rounds)
                        - first_intersection_round,
                        "remote_generated_fraction": remote_generated / generated,
                        "remote_after_fraction": remote_after / source_unique,
                        "max_frontier_skew": max(nonempty_skews),
                        "peak_round_frontier": peak_round.frontier_states,
                        "peak_round_skew": max(peak_round.frontier_by_owner)
                        / peak_average,
                        "max_frontier_skew_ge_128": max(large_skews)
                        if large_skews
                        else "",
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = directed_validation()
    (args.output_dir / "REF-010-directed-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    rows = s8_routing_sweep()
    with (args.output_dir / "REF-010-s8-routing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(validation, indent=2))
    print(f"wrote {len(rows)} S8 routing rows")


if __name__ == "__main__":
    main()
