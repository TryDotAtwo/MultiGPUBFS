"""Reproduce REF-009 bidirectional expansion-policy measurements."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

from multigpubfs.bidirectional import bidirectional_breadth_first_search
from multigpubfs.implicit import labeled_breadth_first_search, replay_moves


POLICIES = ("smaller_frontier", "alternating", "estimated_work")


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


def directed_graph_corpus():
    edges = tuple((u, v) for u in range(4) for v in range(4) if u != v)
    totals = {
        policy: {
            "pairs": 0,
            "distance_mismatches": 0,
            "replay_failures": 0,
            "generated_transitions": 0,
            "expanded_states": 0,
            "expansion_rounds": 0,
            "forward_rounds": 0,
            "reverse_rounds": 0,
        }
        for policy in POLICIES
    }
    best_counts = {policy: 0 for policy in POLICIES}
    strict_winner_counts = {policy: 0 for policy in POLICIES}
    policy_disagreements = 0

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
                pair_results = {}
                for policy in POLICIES:
                    kwargs = {}
                    if policy == "estimated_work":
                        kwargs = {
                            "forward_work_estimate": lambda s, g=outgoing: len(g[s]),
                            "reverse_work_estimate": lambda s, g=incoming: len(g[s]),
                        }
                    result = bidirectional_breadth_first_search(
                        lambda state, g=outgoing: g[state],
                        lambda state, g=incoming: g[state],
                        start=start,
                        target=target,
                        expansion_policy=policy,
                        **kwargs,
                    )
                    pair_results[policy] = result
                    total = totals[policy]
                    total["pairs"] += 1
                    total["distance_mismatches"] += result.distance != expected
                    cursor = start
                    for move in result.moves:
                        edge = next(
                            (child for label, child in outgoing[cursor] if label == move),
                            None,
                        )
                        if edge is None:
                            cursor = None
                            break
                        cursor = edge
                    total["replay_failures"] += bool(result.found and cursor != target)
                    total["generated_transitions"] += result.generated_transitions
                    total["expanded_states"] += (
                        result.forward_expanded + result.reverse_expanded
                    )
                    total["expansion_rounds"] += len(result.expansion_trace)
                    total["forward_rounds"] += result.expansion_trace.count("forward")
                    total["reverse_rounds"] += result.expansion_trace.count("reverse")

                work = {
                    policy: result.generated_transitions
                    for policy, result in pair_results.items()
                }
                minimum = min(work.values())
                winners = [policy for policy, value in work.items() if value == minimum]
                for policy in winners:
                    best_counts[policy] += 1
                if len(winners) == 1:
                    strict_winner_counts[winners[0]] += 1
                policy_disagreements += len(set(work.values())) > 1

    return {
        "graph_count": 1 << len(edges),
        "ordered_distinct_pairs": 49152,
        "policy_disagreement_pairs": policy_disagreements,
        "best_including_ties": best_counts,
        "strict_winners": strict_winner_counts,
        "policies": totals,
    }


def adjacent_transitions(state):
    result = []
    for index in range(len(state) - 1):
        child = list(state)
        child[index], child[index + 1] = child[index + 1], child[index]
        result.append((index, tuple(child)))
    return tuple(result)


def s8_sweep():
    start = tuple(range(8))
    complete = labeled_breadth_first_search(adjacent_transitions, sources=(start,))
    rows = []
    for depth, frontier in enumerate(complete.frontiers):
        target = frontier[0]
        row = {"depth": depth, "target": "".join(map(str, target))}
        for policy in POLICIES:
            kwargs = {}
            if policy == "estimated_work":
                kwargs = {
                    "forward_work_estimate": lambda _state: 7,
                    "reverse_work_estimate": lambda _state: 7,
                }
            result = bidirectional_breadth_first_search(
                adjacent_transitions,
                adjacent_transitions,
                start=start,
                target=target,
                expansion_policy=policy,
                **kwargs,
            )
            assert result.distance == depth
            assert replay_moves(start, result.moves, lambda s, m: adjacent_transitions(s)[m][1]) == target
            prefix = policy
            row[f"{prefix}_generated"] = result.generated_transitions
            row[f"{prefix}_expanded"] = result.forward_expanded + result.reverse_expanded
            row[f"{prefix}_rounds"] = len(result.expansion_trace)
            row[f"{prefix}_forward_rounds"] = result.expansion_trace.count("forward")
            row[f"{prefix}_reverse_rounds"] = result.expansion_trace.count("reverse")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    directed = directed_graph_corpus()
    (args.output_dir / "REF-009-directed-summary.json").write_text(
        json.dumps(directed, indent=2) + "\n", encoding="utf-8"
    )
    rows = s8_sweep()
    with (args.output_dir / "REF-009-s8-policy-sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(directed, indent=2))
    print(f"wrote {len(rows)} S8 rows")


if __name__ == "__main__":
    main()
