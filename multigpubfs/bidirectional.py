"""Deterministic exact bidirectional BFS reference for labeled graphs."""

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, Literal, TypeVar


State = TypeVar("State", bound=Hashable)
Move = TypeVar("Move")
ExpansionPolicy = Literal["smaller_frontier", "alternating", "estimated_work"]


@dataclass(frozen=True)
class BidirectionalBfsResult(Generic[State, Move]):
    """Path and accounting from an exact bidirectional traversal."""

    found: bool
    distance: int | None
    moves: tuple[Move, ...]
    meeting_state: State | None
    forward_visited: int
    reverse_visited: int
    forward_expanded: int
    reverse_expanded: int
    generated_transitions: int
    expansion_trace: tuple[str, ...]


def bidirectional_breadth_first_search(
    forward_transitions: Callable[[State], Iterable[tuple[Move, State]]],
    reverse_transitions: Callable[[State], Iterable[tuple[Move, State]]],
    *,
    start: State,
    target: State,
    expansion_policy: ExpansionPolicy = "smaller_frontier",
    forward_work_estimate: Callable[[State], int] | None = None,
    reverse_work_estimate: Callable[[State], int] | None = None,
) -> BidirectionalBfsResult[State, Move]:
    """Find an exact shortest labeled path under a complete-level policy.

    A reverse transition from ``state`` must return ``(move, predecessor)`` such
    that applying the forward ``move`` to ``predecessor`` produces ``state``.
    ``estimated_work`` sums user-supplied nonnegative per-state estimates for
    each current frontier and expands the lower estimated outgoing work side.
    """

    valid_policies = {"smaller_frontier", "alternating", "estimated_work"}
    if expansion_policy not in valid_policies:
        raise ValueError(f"unknown expansion policy: {expansion_policy!r}")
    if expansion_policy == "estimated_work" and (
        forward_work_estimate is None or reverse_work_estimate is None
    ):
        raise ValueError("estimated_work requires forward and reverse estimators")

    if start == target:
        return BidirectionalBfsResult(
            found=True,
            distance=0,
            moves=(),
            meeting_state=start,
            forward_visited=1,
            reverse_visited=1,
            forward_expanded=0,
            reverse_expanded=0,
            generated_transitions=0,
            expansion_trace=(),
        )

    forward_distance = {start: 0}
    forward_parent: dict[State, State | None] = {start: None}
    forward_move: dict[State, Move | None] = {start: None}
    reverse_distance = {target: 0}
    reverse_next: dict[State, State | None] = {target: None}
    reverse_move: dict[State, Move | None] = {target: None}
    forward_frontier = [start]
    reverse_frontier = [target]
    forward_frontier_depth = 0
    reverse_frontier_depth = 0
    forward_expanded = 0
    reverse_expanded = 0
    generated_transitions = 0
    best_distance: int | None = None
    meeting: State | None = None
    expansion_trace: list[str] = []

    while forward_frontier and reverse_frontier:
        if (
            best_distance is not None
            and forward_frontier_depth + reverse_frontier_depth >= best_distance
        ):
            break

        if expansion_policy == "smaller_frontier":
            expand_forward = len(forward_frontier) <= len(reverse_frontier)
        elif expansion_policy == "alternating":
            expand_forward = len(expansion_trace) % 2 == 0
        else:
            assert forward_work_estimate is not None
            assert reverse_work_estimate is not None
            forward_work = sum(forward_work_estimate(s) for s in forward_frontier)
            reverse_work = sum(reverse_work_estimate(s) for s in reverse_frontier)
            if forward_work < 0 or reverse_work < 0:
                raise ValueError("work estimates must be nonnegative")
            expand_forward = forward_work <= reverse_work

        if expand_forward:
            expansion_trace.append("forward")
            next_frontier: list[State] = []
            for state in forward_frontier:
                forward_expanded += 1
                for move, child in forward_transitions(state):
                    generated_transitions += 1
                    if child in forward_distance:
                        continue
                    child_depth = forward_frontier_depth + 1
                    forward_distance[child] = child_depth
                    forward_parent[child] = state
                    forward_move[child] = move
                    next_frontier.append(child)
                    if child in reverse_distance:
                        candidate = child_depth + reverse_distance[child]
                        if best_distance is None or candidate < best_distance:
                            best_distance = candidate
                            meeting = child
            forward_frontier = next_frontier
            forward_frontier_depth += 1
        else:
            expansion_trace.append("reverse")
            next_frontier = []
            for state in reverse_frontier:
                reverse_expanded += 1
                for move, predecessor in reverse_transitions(state):
                    generated_transitions += 1
                    if predecessor in reverse_distance:
                        continue
                    predecessor_depth = reverse_frontier_depth + 1
                    reverse_distance[predecessor] = predecessor_depth
                    reverse_next[predecessor] = state
                    reverse_move[predecessor] = move
                    next_frontier.append(predecessor)
                    if predecessor in forward_distance:
                        candidate = predecessor_depth + forward_distance[predecessor]
                        if best_distance is None or candidate < best_distance:
                            best_distance = candidate
                            meeting = predecessor
            reverse_frontier = next_frontier
            reverse_frontier_depth += 1

    if meeting is None or best_distance is None:
        return BidirectionalBfsResult(
            found=False,
            distance=None,
            moves=(),
            meeting_state=None,
            forward_visited=len(forward_distance),
            reverse_visited=len(reverse_distance),
            forward_expanded=forward_expanded,
            reverse_expanded=reverse_expanded,
            generated_transitions=generated_transitions,
            expansion_trace=tuple(expansion_trace),
        )

    prefix_reversed: list[Move] = []
    cursor = meeting
    parent = forward_parent[cursor]
    while parent is not None:
        move = forward_move[cursor]
        if move is None:
            raise ValueError(f"forward state {cursor!r} has no parent move")
        prefix_reversed.append(move)
        cursor = parent
        parent = forward_parent[cursor]
    prefix_reversed.reverse()

    suffix: list[Move] = []
    cursor = meeting
    next_state = reverse_next[cursor]
    while next_state is not None:
        move = reverse_move[cursor]
        if move is None:
            raise ValueError(f"reverse state {cursor!r} has no move to target")
        suffix.append(move)
        cursor = next_state
        next_state = reverse_next[cursor]

    moves = tuple(prefix_reversed + suffix)
    if len(moves) != best_distance:
        raise AssertionError("reconstructed path length disagrees with search distance")

    return BidirectionalBfsResult(
        found=True,
        distance=best_distance,
        moves=moves,
        meeting_state=meeting,
        forward_visited=len(forward_distance),
        reverse_visited=len(reverse_distance),
        forward_expanded=forward_expanded,
        reverse_expanded=reverse_expanded,
        generated_transitions=generated_transitions,
        expansion_trace=tuple(expansion_trace),
    )
