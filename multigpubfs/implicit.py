"""Exact CPU reference traversal for implicitly generated labeled graphs."""

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, TypeVar

from multigpubfs.reference import BfsResult, validate_bfs_result


State = TypeVar("State", bound=Hashable)
Move = TypeVar("Move")


@dataclass(frozen=True)
class LabeledBfsResult(Generic[State, Move]):
    """Complete traversal with enough metadata to reconstruct generator paths."""

    distance: dict[State, int]
    parent: dict[State, State | None]
    parent_move: dict[State, Move | None]
    frontiers: tuple[tuple[State, ...], ...]
    generated_transitions: int


def labeled_breadth_first_search(
    transitions: Callable[[State], Iterable[tuple[Move, State]]],
    *,
    sources: Iterable[State],
) -> LabeledBfsResult[State, Move]:
    """Traverse a finite implicit graph in deterministic level order."""

    distance: dict[State, int] = {}
    parent: dict[State, State | None] = {}
    parent_move: dict[State, Move | None] = {}
    frontier: list[State] = []

    for source in sources:
        if source in distance:
            continue
        distance[source] = 0
        parent[source] = None
        parent_move[source] = None
        frontier.append(source)

    frontiers: list[tuple[State, ...]] = []
    generated_transitions = 0
    depth = 0
    while frontier:
        frontiers.append(tuple(frontier))
        next_frontier: list[State] = []
        for state in frontier:
            for move, child in transitions(state):
                generated_transitions += 1
                if child in distance:
                    continue
                distance[child] = depth + 1
                parent[child] = state
                parent_move[child] = move
                next_frontier.append(child)
        frontier = next_frontier
        depth += 1

    return LabeledBfsResult(
        distance=distance,
        parent=parent,
        parent_move=parent_move,
        frontiers=tuple(frontiers),
        generated_transitions=generated_transitions,
    )


def reconstruct_moves(
    result: LabeledBfsResult[State, Move],
    target: State,
) -> tuple[Move, ...]:
    """Reconstruct the selected shortest sequence of move labels to target."""

    reversed_moves: list[Move] = []
    cursor = target
    parent = result.parent[cursor]
    while parent is not None:
        move = result.parent_move[cursor]
        if move is None:
            raise ValueError(f"non-source state {cursor!r} has no parent move")
        reversed_moves.append(move)
        cursor = parent
        parent = result.parent[cursor]
    reversed_moves.reverse()
    return tuple(reversed_moves)


def replay_moves(
    source: State,
    moves: Iterable[Move],
    apply_move: Callable[[State, Move], State],
) -> State:
    """Apply a move sequence to a source state."""

    state = source
    for move in moves:
        state = apply_move(state, move)
    return state


def validate_labeled_bfs_result(
    transitions: Callable[[State], Iterable[tuple[Move, State]]],
    result: LabeledBfsResult[State, Move],
) -> tuple[str, ...]:
    """Return graph and move-replay errors in a complete labeled BFS result."""

    transition_cache: dict[State, tuple[tuple[Move, State], ...]] = {}

    def cached_transitions(state: State) -> tuple[tuple[Move, State], ...]:
        if state not in transition_cache:
            transition_cache[state] = tuple(transitions(state))
        return transition_cache[state]

    unlabeled = BfsResult(
        distance=result.distance,
        parent=result.parent,
        frontiers=result.frontiers,
    )
    errors = list(
        validate_bfs_result(
            lambda state: (child for _, child in cached_transitions(state)),
            unlabeled,
        )
    )

    for state, depth in result.distance.items():
        if depth == 0:
            continue
        parent = result.parent.get(state)
        move = result.parent_move.get(state)
        if parent is None or move is None:
            continue
        if (move, state) not in cached_transitions(parent):
            errors.append(
                f"parent move {move!r} from {parent!r} does not produce {state!r}"
            )
    return tuple(errors)
