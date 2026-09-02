"""Small, deterministic CPU reference for exact breadth-first search."""

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, TypeVar


Vertex = TypeVar("Vertex", bound=Hashable)


@dataclass(frozen=True)
class BfsResult(Generic[Vertex]):
    """Observable result of a complete level-synchronous traversal."""

    distance: dict[Vertex, int]
    parent: dict[Vertex, Vertex | None]
    frontiers: tuple[tuple[Vertex, ...], ...]


def breadth_first_search(
    neighbors: Callable[[Vertex], Iterable[Vertex]],
    *,
    sources: Iterable[Vertex],
) -> BfsResult[Vertex]:
    """Traverse every vertex reachable from one or more sources."""

    distance: dict[Vertex, int] = {}
    parent: dict[Vertex, Vertex | None] = {}
    frontier: list[Vertex] = []

    for source in sources:
        if source in distance:
            continue
        distance[source] = 0
        parent[source] = None
        frontier.append(source)

    frontiers: list[tuple[Vertex, ...]] = []
    depth = 0
    while frontier:
        frontiers.append(tuple(frontier))
        next_frontier: list[Vertex] = []
        for vertex in frontier:
            for neighbor in neighbors(vertex):
                if neighbor in distance:
                    continue
                distance[neighbor] = depth + 1
                parent[neighbor] = vertex
                next_frontier.append(neighbor)
        frontier = next_frontier
        depth += 1

    return BfsResult(
        distance=distance,
        parent=parent,
        frontiers=tuple(frontiers),
    )


def validate_bfs_result(
    neighbors: Callable[[Vertex], Iterable[Vertex]],
    result: BfsResult[Vertex],
) -> tuple[str, ...]:
    """Return semantic errors found in a complete BFS result."""

    errors: list[str] = []
    for frontier_depth, frontier in enumerate(result.frontiers):
        for vertex in frontier:
            recorded_depth = result.distance.get(vertex)
            if recorded_depth != frontier_depth:
                errors.append(
                    f"frontier {frontier_depth} contains {vertex!r} "
                    f"with recorded depth {recorded_depth!r}"
                )
    for vertex, depth in result.distance.items():
        vertex_neighbors = tuple(neighbors(vertex))
        for child in vertex_neighbors:
            if child not in result.distance:
                errors.append(f"reachable child {child!r} of {vertex!r} is missing")
                continue
            child_depth = result.distance[child]
            if child_depth > depth + 1:
                errors.append(
                    f"edge {vertex!r} -> {child!r} violates shortest distances "
                    f"{depth} -> {child_depth}"
                )
        parent = result.parent.get(vertex)
        if depth == 0:
            continue
        expected_parent_depth = depth - 1
        if parent is None or result.distance.get(parent) != expected_parent_depth:
            errors.append(
                f"parent edge/depth invalid for {vertex!r}: "
                f"parent {parent!r} is not at depth {expected_parent_depth}"
            )
            continue
        if vertex not in neighbors(parent):
            errors.append(
                f"parent edge/depth invalid for {vertex!r}: "
                f"{parent!r} is not connected to the child"
            )
    return tuple(errors)
