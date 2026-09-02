"""Exact unidirectional target BFS with explicit stop granularity."""

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, Literal, TypeVar


State = TypeVar("State", bound=Hashable)
Move = TypeVar("Move")
StopGranularity = Literal["candidate", "parent_batch", "level"]


@dataclass(frozen=True)
class TargetBfsResult(Generic[State, Move]):
    """Shortest path and work accounting at the declared stop boundary."""

    found: bool
    distance: int | None
    moves: tuple[Move, ...]
    generated_transitions: int
    expanded_states: int
    discovered_states: int
    completed_levels: int


def _reconstruct_moves(
    parent: dict[State, State | None],
    parent_move: dict[State, Move | None],
    target: State,
) -> tuple[Move, ...]:
    reversed_moves: list[Move] = []
    cursor = target
    previous = parent[cursor]
    while previous is not None:
        move = parent_move[cursor]
        if move is None:
            raise ValueError(f"non-source state {cursor!r} has no parent move")
        reversed_moves.append(move)
        cursor = previous
        previous = parent[cursor]
    reversed_moves.reverse()
    return tuple(reversed_moves)


def target_breadth_first_search(
    transitions: Callable[[State], Iterable[tuple[Move, State]]],
    *,
    start: State,
    target: State,
    stop_granularity: StopGranularity = "candidate",
    parent_batch_size: int = 1,
) -> TargetBfsResult[State, Move]:
    """Find an exact shortest path and stop at a declared processing boundary."""

    if stop_granularity not in ("candidate", "parent_batch", "level"):
        raise ValueError(f"unknown stop granularity: {stop_granularity!r}")
    if parent_batch_size <= 0:
        raise ValueError("parent_batch_size must be positive")
    if start == target:
        return TargetBfsResult(
            found=True,
            distance=0,
            moves=(),
            generated_transitions=0,
            expanded_states=0,
            discovered_states=1,
            completed_levels=0,
        )

    distance = {start: 0}
    parent: dict[State, State | None] = {start: None}
    parent_move: dict[State, Move | None] = {start: None}
    frontier = [start]
    depth = 0
    generated_transitions = 0
    expanded_states = 0

    while frontier:
        next_frontier: list[State] = []
        target_found = False
        batch_size = (
            parent_batch_size
            if stop_granularity == "parent_batch"
            else len(frontier)
        )
        for batch_start in range(0, len(frontier), batch_size):
            batch = frontier[batch_start : batch_start + batch_size]
            for state in batch:
                expanded_states += 1
                for move, child in transitions(state):
                    generated_transitions += 1
                    if child in distance:
                        continue
                    distance[child] = depth + 1
                    parent[child] = state
                    parent_move[child] = move
                    next_frontier.append(child)
                    if child == target:
                        target_found = True
                        if stop_granularity == "candidate":
                            moves = _reconstruct_moves(parent, parent_move, target)
                            return TargetBfsResult(
                                found=True,
                                distance=depth + 1,
                                moves=moves,
                                generated_transitions=generated_transitions,
                                expanded_states=expanded_states,
                                discovered_states=len(distance),
                                completed_levels=depth,
                            )
            if target_found and stop_granularity == "parent_batch":
                moves = _reconstruct_moves(parent, parent_move, target)
                return TargetBfsResult(
                    found=True,
                    distance=depth + 1,
                    moves=moves,
                    generated_transitions=generated_transitions,
                    expanded_states=expanded_states,
                    discovered_states=len(distance),
                    completed_levels=depth,
                )
        if target_found:
            moves = _reconstruct_moves(parent, parent_move, target)
            return TargetBfsResult(
                found=True,
                distance=depth + 1,
                moves=moves,
                generated_transitions=generated_transitions,
                expanded_states=expanded_states,
                discovered_states=len(distance),
                completed_levels=depth + 1,
            )
        frontier = next_frontier
        depth += 1

    return TargetBfsResult(
        found=False,
        distance=None,
        moves=(),
        generated_transitions=generated_transitions,
        expanded_states=expanded_states,
        discovered_states=len(distance),
        completed_levels=depth,
    )
