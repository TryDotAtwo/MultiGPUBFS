"""Exact owner-computes simulation of distributed bidirectional BFS."""

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, Literal, TypeVar


State = TypeVar("State", bound=Hashable)
Move = TypeVar("Move")
ExpansionPolicy = Literal["smaller_frontier", "alternating"]


@dataclass(frozen=True)
class DistributedRoundMetrics:
    """Lossless accounting for one globally agreed complete-level superstep."""

    side: str
    depth: int
    frontier_states: int
    frontier_by_owner: tuple[int, ...]
    generated_transitions: int
    remote_generated_occurrences: int
    source_duplicate_occurrences: int
    source_unique_candidates: int
    local_after_source_dedup: int
    remote_after_source_dedup: int
    owner_duplicate_occurrences: int
    owner_unique_candidates: int
    already_visited: int
    newly_discovered: int
    remote_only_newly_discovered: int
    remote_accept_bitmap_bytes: int
    intersections_discovered: int


@dataclass(frozen=True)
class DistributedBidirectionalResult(Generic[State, Move]):
    found: bool
    distance: int | None
    moves: tuple[Move, ...]
    meeting_state: State | None
    forward_visited: int
    reverse_visited: int
    rounds: tuple[DistributedRoundMetrics, ...]


def distributed_bidirectional_breadth_first_search(
    forward_transitions: Callable[[State], Iterable[tuple[Move, State]]],
    reverse_transitions: Callable[[State], Iterable[tuple[Move, State]]],
    *,
    start: State,
    target: State,
    owner: Callable[[State], int],
    world_size: int,
    expansion_policy: ExpansionPolicy = "smaller_frontier",
) -> DistributedBidirectionalResult[State, Move]:
    """Simulate exact distributed BFS with authoritative owner-side visited.

    Reverse transitions follow the same contract as the single-process
    bidirectional reference: ``(forward_move, predecessor)``.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if expansion_policy not in {"smaller_frontier", "alternating"}:
        raise ValueError(f"unknown expansion policy: {expansion_policy!r}")

    def checked_owner(state: State) -> int:
        rank = owner(state)
        if not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError(
                f"owner({state!r}) returned {rank!r}, outside [0, {world_size})"
            )
        return rank

    checked_owner(start)
    checked_owner(target)
    if start == target:
        return DistributedBidirectionalResult(
            found=True,
            distance=0,
            moves=(),
            meeting_state=start,
            forward_visited=1,
            reverse_visited=1,
            rounds=(),
        )

    forward_distance = {start: 0}
    forward_parent: dict[State, State | None] = {start: None}
    forward_move: dict[State, Move | None] = {start: None}
    reverse_distance = {target: 0}
    reverse_next: dict[State, State | None] = {target: None}
    reverse_move: dict[State, Move | None] = {target: None}
    forward_frontier = [start]
    reverse_frontier = [target]
    forward_depth = 0
    reverse_depth = 0
    best_distance: int | None = None
    meeting: State | None = None
    rounds: list[DistributedRoundMetrics] = []

    while forward_frontier and reverse_frontier:
        if best_distance is not None and forward_depth + reverse_depth >= best_distance:
            break
        if expansion_policy == "alternating":
            expand_forward = len(rounds) % 2 == 0
        else:
            expand_forward = len(forward_frontier) <= len(reverse_frontier)

        frontier = forward_frontier if expand_forward else reverse_frontier
        side_depth = forward_depth if expand_forward else reverse_depth
        transitions = forward_transitions if expand_forward else reverse_transitions
        side_distance = forward_distance if expand_forward else reverse_distance
        opposite_distance = reverse_distance if expand_forward else forward_distance

        frontier_by_owner = [0] * world_size
        generated = []
        remote_generated = 0
        for parent in frontier:
            source_owner = checked_owner(parent)
            frontier_by_owner[source_owner] += 1
            for move, candidate in transitions(parent):
                destination_owner = checked_owner(candidate)
                generated.append(
                    (source_owner, destination_owner, move, candidate, parent)
                )
                remote_generated += source_owner != destination_owner

        source_seen: set[tuple[int, State]] = set()
        source_unique = []
        for record in generated:
            key = (record[0], record[3])
            if key in source_seen:
                continue
            source_seen.add(key)
            source_unique.append(record)

        local_after = sum(record[0] == record[1] for record in source_unique)
        remote_after = len(source_unique) - local_after
        remote_pair_counts: dict[tuple[int, int], int] = {}
        for record in source_unique:
            source_owner, destination_owner = record[:2]
            if source_owner != destination_owner:
                pair = (source_owner, destination_owner)
                remote_pair_counts[pair] = remote_pair_counts.get(pair, 0) + 1
        remote_bitmap_bytes = sum(
            (count + 7) // 8 for count in remote_pair_counts.values()
        )

        records_by_candidate: dict[
            State, list[tuple[int, int, Move, State, State]]
        ] = {}
        for record in source_unique:
            records_by_candidate.setdefault(record[3], []).append(record)
        owner_unique = [
            min(records, key=lambda item: (item[0] != item[1], item[0]))
            for records in records_by_candidate.values()
        ]

        next_frontier: list[State] = []
        already_visited = 0
        intersections = 0
        remote_only_new = 0
        for source_owner, destination_owner, move, candidate, parent in owner_unique:
            if candidate in side_distance:
                already_visited += 1
                continue
            candidate_depth = side_depth + 1
            side_distance[candidate] = candidate_depth
            next_frontier.append(candidate)
            remote_only_new += source_owner != destination_owner
            if expand_forward:
                forward_parent[candidate] = parent
                forward_move[candidate] = move
            else:
                reverse_next[candidate] = parent
                reverse_move[candidate] = move
            if candidate in opposite_distance:
                intersections += 1
                candidate_total = candidate_depth + opposite_distance[candidate]
                if best_distance is None or candidate_total < best_distance:
                    best_distance = candidate_total
                    meeting = candidate

        rounds.append(
            DistributedRoundMetrics(
                side="forward" if expand_forward else "reverse",
                depth=side_depth,
                frontier_states=len(frontier),
                frontier_by_owner=tuple(frontier_by_owner),
                generated_transitions=len(generated),
                remote_generated_occurrences=remote_generated,
                source_duplicate_occurrences=len(generated) - len(source_unique),
                source_unique_candidates=len(source_unique),
                local_after_source_dedup=local_after,
                remote_after_source_dedup=remote_after,
                owner_duplicate_occurrences=len(source_unique) - len(owner_unique),
                owner_unique_candidates=len(owner_unique),
                already_visited=already_visited,
                newly_discovered=len(next_frontier),
                remote_only_newly_discovered=remote_only_new,
                remote_accept_bitmap_bytes=remote_bitmap_bytes,
                intersections_discovered=intersections,
            )
        )
        if expand_forward:
            forward_frontier = next_frontier
            forward_depth += 1
        else:
            reverse_frontier = next_frontier
            reverse_depth += 1

    if meeting is None or best_distance is None:
        return DistributedBidirectionalResult(
            found=False,
            distance=None,
            moves=(),
            meeting_state=None,
            forward_visited=len(forward_distance),
            reverse_visited=len(reverse_distance),
            rounds=tuple(rounds),
        )

    prefix: list[Move] = []
    cursor = meeting
    while forward_parent[cursor] is not None:
        move = forward_move[cursor]
        if move is None:
            raise AssertionError("missing forward parent move")
        prefix.append(move)
        cursor = forward_parent[cursor]  # type: ignore[assignment]
    prefix.reverse()

    suffix: list[Move] = []
    cursor = meeting
    while reverse_next[cursor] is not None:
        move = reverse_move[cursor]
        if move is None:
            raise AssertionError("missing reverse move")
        suffix.append(move)
        cursor = reverse_next[cursor]  # type: ignore[assignment]
    moves = tuple(prefix + suffix)
    if len(moves) != best_distance:
        raise AssertionError("reconstructed path length disagrees with distance")

    return DistributedBidirectionalResult(
        found=True,
        distance=best_distance,
        moves=moves,
        meeting_state=meeting,
        forward_visited=len(forward_distance),
        reverse_visited=len(reverse_distance),
        rounds=tuple(rounds),
    )
