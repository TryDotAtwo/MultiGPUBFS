"""Reproduce REF-011 eager versus two-phase BFS wire-byte estimates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from multigpubfs.distributed_bidirectional import (
    distributed_bidirectional_breadth_first_search,
)
from multigpubfs.implicit import labeled_breadth_first_search
from multigpubfs.wire_model import WireFormat, estimate_wire_bytes


FORMATS = {
    "packed_rank16": WireFormat(key_bytes=2, parent_bytes=2, move_bytes=1),
    "packed_rank32": WireFormat(key_bytes=4, parent_bytes=4, move_bytes=1),
    "aligned_rank64": WireFormat(key_bytes=8, parent_bytes=8, move_bytes=8),
    "state128_parent128": WireFormat(key_bytes=16, parent_bytes=16, move_bytes=4),
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


def sweep():
    start = tuple(range(8))
    complete = labeled_breadth_first_search(adjacent_transitions, sources=(start,))
    rows = []
    for depth in (2, 8, 14, 20, 28):
        target = complete.frontiers[depth][0]
        for strategy in ("direct", "mixed"):
            for world_size in (2, 4, 8):
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
                remote_candidates = sum(
                    item.remote_after_source_dedup for item in result.rounds
                )
                remote_only_accepted = sum(
                    item.remote_only_newly_discovered for item in result.rounds
                )
                bitmap_bytes = sum(
                    item.remote_accept_bitmap_bytes for item in result.rounds
                )
                for format_name, wire_format in FORMATS.items():
                    estimate = estimate_wire_bytes(
                        remote_candidates=remote_candidates,
                        remote_only_accepted=remote_only_accepted,
                        accept_bitmap_bytes=bitmap_bytes,
                        wire_format=wire_format,
                    )
                    round_estimates = [
                        estimate_wire_bytes(
                            remote_candidates=item.remote_after_source_dedup,
                            remote_only_accepted=item.remote_only_newly_discovered,
                            accept_bitmap_bytes=item.remote_accept_bitmap_bytes,
                            wire_format=wire_format,
                        )
                        for item in result.rounds
                    ]
                    hybrid_bytes = sum(
                        min(item.eager_bytes, item.two_phase_total_bytes)
                        for item in round_estimates
                    )
                    two_phase_rounds = sum(
                        item.two_phase_total_bytes < item.eager_bytes
                        for item in round_estimates
                    )
                    rows.append(
                        {
                            "depth": depth,
                            "strategy": strategy,
                            "world_size": world_size,
                            "format": format_name,
                            "key_bytes": wire_format.key_bytes,
                            "parent_bytes": wire_format.parent_bytes,
                            "move_bytes": wire_format.move_bytes,
                            "remote_candidates": remote_candidates,
                            "remote_only_accepted": remote_only_accepted,
                            "accept_bitmap_bytes": bitmap_bytes,
                            "remote_only_accept_fraction": (
                                remote_only_accepted / remote_candidates
                                if remote_candidates
                                else 0.0
                            ),
                            "eager_bytes": estimate.eager_bytes,
                            "two_phase_bytes": estimate.two_phase_total_bytes,
                            "two_phase_reduction_fraction": (
                                estimate.two_phase_reduction_fraction
                            ),
                            "hybrid_bytes": hybrid_bytes,
                            "hybrid_reduction_fraction": (
                                0.0
                                if estimate.eager_bytes == 0
                                else 1.0 - hybrid_bytes / estimate.eager_bytes
                            ),
                            "hybrid_two_phase_rounds": two_phase_rounds,
                            "hybrid_eager_rounds": len(round_estimates)
                            - two_phase_rounds,
                        }
                    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = sweep()
    output = args.output_dir / "REF-011-wire-byte-sweep.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} wire estimates to {output}")


if __name__ == "__main__":
    main()
