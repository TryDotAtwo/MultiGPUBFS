# REF-034: secondary cost among shortest-hop paths

## Question

Does an ordinary first-discovery BFS parent minimize an additive secondary cost
among minimum-hop paths, and what information is lost by one label per vertex?

## Method

- Rust adjacency lists with explicit secondary edge costs.
- Ordinary claim-before-enqueue BFS and one first parent.
- Dynamic programming over every edge increasing exact BFS depth by one.
- Exhaustive enumeration of all simple paths in a six-vertex fixture.
- Docker-only formatting, compilation, and execution.

## Retained failure

The first `rustfmt --check` gate reported formatting differences and stopped the
shell chain before compilation. The suggested formatting was applied; the same
gate then passed.

## Result

```text
target=3 bfs_hops=Some(2) first_parent=Some(1) first_parent_cost=200
shortest_dag_secondary_cost=2
all_simple_path_pairs=[(2, 2), (2, 200), (3, 0)] pareto_pairs=[(2, 2), (3, 0)]
```

The BFS hop distance was exact, but first discovery selected secondary cost 200
instead of 2. The Pareto set retained both `(2,2)` and `(3,0)`, proving that one
scalar label cannot encode the two objectives simultaneously.

## Status

Pass after one formatting-only failed gate. This is exhaustive for the declared
fixture and is not performance evidence.

