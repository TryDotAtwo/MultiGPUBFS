"""Measure duplicate rejection by BFS depth on the 24-dimensional hypercube.

The graph is a Cayley graph of (Z/2Z)^24 with one bit-flip generator per move.
Duplicate categories follow the proposed pipeline order: reject against all prior
layers first, then unique the remaining candidates within the current layer.
"""

from __future__ import annotations

import argparse
import json


def collect(max_depth: int, dimensions: int = 24) -> list[dict[str, int | float]]:
    visited = {0}
    frontier = {0}
    rows: list[dict[str, int | float]] = []

    for depth in range(1, max_depth + 1):
        raw = len(frontier) * dimensions
        old_visited = 0
        unseen_occurrences = 0
        next_frontier: set[int] = set()

        for state in frontier:
            for bit in range(dimensions):
                child = state ^ (1 << bit)
                if child in visited:
                    old_visited += 1
                else:
                    unseen_occurrences += 1
                    next_frontier.add(child)

        same_layer = unseen_occurrences - len(next_frontier)
        survivors = len(next_frontier)
        rows.append(
            {
                "depth": depth,
                "frontier_in": len(frontier),
                "generated": raw,
                "old_visited_duplicates": old_visited,
                "same_layer_duplicates": same_layer,
                "duplicates_total": old_visited + same_layer,
                "survivors": survivors,
                "duplicate_fraction": (old_visited + same_layer) / raw,
                "survivor_fraction": survivors / raw,
            }
        )

        visited.update(next_frontier)
        frontier = next_frontier

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(collect(args.max_depth), indent=2))


if __name__ == "__main__":
    main()
