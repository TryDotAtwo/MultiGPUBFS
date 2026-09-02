# REF-001: deterministic CPU reference and validator

Date: 2026-08-27.

## Question

Can a small level-synchronous CPU implementation and an independently written
validator detect failure modes expected from later parallel implementations?

## Hypothesis

A result validator must inspect frontier/depth consistency, parent edges,
shortest-distance inequalities, and closure over reachable outgoing neighbors.
Checking only parent chains is insufficient.

## Semantics

- Directed or undirected unweighted graph supplied through `neighbors(vertex)`.
- One or more sources, with duplicate sources ignored.
- Complete traversal of every reachable vertex.
- Deterministic first-parent selection follows source and neighbor iteration
  order.

## Command

```powershell
py -m unittest discover -s tests -v
```

## Correctness oracle

The tests use hand-derived literal distances/frontiers and deliberately corrupted
results. They do not compute expected values with the implementation under test.

## Result

Six tests passed:

- exact shortest distances and complete frontiers;
- valid traversal with duplicate sources, duplicate edges, self-loop, and cycle;
- rejection of a parent from the wrong depth;
- rejection of a silently dropped reachable vertex;
- rejection of a connected parent tree containing a nonminimal distance;
- rejection of frontier/depth disagreement.

## Observation

A parent chain can be locally valid while the recorded distance is not minimal.
For example, a vertex may be attached through a length-three path even though a
different explored edge reaches it at depth two. The edge relaxation condition
exposes this corruption.

## Limitations

- No bounded-depth, target-stop, bidirectional, or weighted semantics.
- No performance measurement; the implementation favors clarity.
- The validator assumes finite complete traversal and enumerates outgoing
  neighbors of every discovered vertex.
- The result does not yet expose move labels required for implicit-state path
  replay.

## Next experiments

- Add path reconstruction with edge/move replay.
- Add a tiny permutation-group neighbor generator.
- Exhaustively enumerate small graph fixtures and cross-check against an
  independently structured queue BFS.
- Establish bounded-depth validator semantics before GPU capacity experiments.
