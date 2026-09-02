# BFS shortest-hop paths, secondary cost, and Pareto boundaries

Ordinary BFS minimizes edge count. If edges also carry a secondary cost, its
first-discovery parent is still a valid minimum-hop parent, but need not be the
cheapest parent among minimum-hop paths. The BFS layers can support that second
calculation only after the objective contract is stated explicitly.

No optimizer, production implementation, benchmark, or GPU code is added.

## 1. Three different objectives

For a path `P`, let `h(P)` be its number of edges and let
`c(P)=sum_(e in P) c(e)`. At least three natural requests exist:

```text
lexicographic hop-first:  minimize (h(P), c(P));
lexicographic cost-first: minimize (c(P), h(P));
Pareto: retain every nondominated (h(P), c(P)) trade-off.
```

They are not interchangeable. Ordinary BFS solves only the primary `h(P)`
part of the first request. Weighted shortest-path search addresses `c(P)` as
the primary objective under its own weight assumptions. Pareto search can need
several incomparable labels for one graph vertex.

## 2. First discovery does not optimize secondary cost

Suppose `s` has two two-edge paths to `t`:

```text
s -> a -> t, secondary cost 200
s -> b -> t, secondary cost   2.
```

If adjacency/queue order processes `a` first, ordinary claim-before-enqueue BFS
sets `parent(t)=a`. The depth `d(t)=2` is exact, but the chosen path is not
secondary-optimal among depth-two paths.

This is not a BFS correctness failure. A one-parent shortest-path tree promises
one arbitrary shortest-hop parent unless a stronger tie contract was declared.
Sorting one adjacency list or relying on a thread race changes selection order,
not the mathematical guarantee.

## 3. The shortest-path DAG

After exact BFS distances `d` from `s` are known, retain every directed edge

```text
(u,v) such that d(v)=d(u)+1.
```

These are exactly the edges that can occur on minimum-hop paths from `s` to
their endpoints. Depth strictly increases, so this subgraph is acyclic even if
the original graph has cycles.

The best secondary cost among shortest-hop paths satisfies

```text
best(s)=0,
best(v)=min_(u,v):d(u)+1=d(v) (best(u)+c(u,v)).
```

Processing vertices by increasing BFS depth is a dynamic program on the
shortest-path DAG. A minimizing predecessor reconstructs a path meeting the
lexicographic `(hop count, secondary cost)` contract.

Because this retained graph is acyclic, the secondary edge costs may even be
negative without creating a negative-cycle ambiguity. Arithmetic range and the
finite-path contract still need validation.

## 4. What must be finalized before expansion

A level-synchronous implementation can combine discovery with secondary-cost
reduction if it collects every candidate from `F_(d-1)` before expanding
`F_d`. The hop label becomes final on first discovery, but the secondary label
is final only after all shortest predecessors have contributed.

If `v` is expanded with secondary value 200 and later improved to 2 at the same
depth, descendants may inherit the wrong secondary value. Merely changing
`best(v)` is insufficient: the improvement must propagate, or the entire layer
must have been reduced before expansion.

This is analogous to canonical nearest-source labels: scalar BFS distance can
be correct while richer equal-distance metadata is stale.

## 5. Cost-first is a different problem

Add a three-edge path from `s` to `t` with secondary cost zero. Then:

```text
hop-first chooses (2,2),
cost-first chooses (3,0).
```

No queue-order choice reconciles these objectives. Cost-first search must use a
weighted shortest-path method appropriate for the cost domain. Minimizing hops
only among minimum-cost paths is then a separate tie problem; zero-cost cycles
make its tight-edge structure subtler than the depth-increasing BFS DAG.

Encoding an edge with cost 100 as 100 unit edges changes hop semantics unless
those subdivision vertices are explicitly part of the requested graph.

## 6. Pareto labels invalidate one visited bit

A pair `(h1,c1)` dominates `(h2,c2)` when it is no worse in either coordinate
and strictly better in at least one. In the probe, `t` has path pairs

```text
(2,2), (2,200), (3,0).
```

`(2,200)` is dominated, while `(2,2)` and `(3,0)` are incomparable. A single
distance, cost, parent, or visited bit cannot represent both nondominated
choices.

General multicriteria shortest-path algorithms therefore use multiple labels
and dominance pruning. Their label counts and complexity belong to a different
contract from BFS frontier size. This note does not propose such an algorithm.

## 7. Resource constraints and product states

If the request is “minimum cost using at most `K` edges” or “minimum hops under
a budget,” arriving at vertex `v` with different consumed resources can change
future feasibility. Merging those arrivals by vertex alone can be unsound.

One exact modeling option is a product state such as `(v,used_budget)` or
`(v,hops)`, followed by the traversal appropriate to the resulting edge costs.
Another is a nondominated-label method. This is the same state-sufficiency
principle as history-constrained BFS: visited identity must retain every field
that affects legal continuations or requested output.

Resource-constrained shortest path is not generally reduced to ordinary BFS by
calling the resource a tie-breaker. Handler and Zang study the genuinely
constrained problem; Martins treats multiple nondominated criteria.

## 8. Parent and count contracts

The secondary DP can return:

- only the best secondary scalar;
- one arbitrary minimizing predecessor;
- a deterministic predecessor under an additional total order;
- every minimizing predecessor;
- the count of secondary-optimal shortest-hop paths.

These require progressively richer metadata. The ordinary shortest-path DAG
may contain predecessors whose paths have the correct hop count but nonminimal
secondary cost. Counting every shortest-hop path is therefore not the same as
counting paths optimal under both criteria.

## 9. Cayley interpretation

In an implicit Cayley graph, hop-first secondary cost can mean: use the minimum
number of generators, then minimize energy, move penalty, or another additive
generator/state-transition cost among geodesic words.

Different geodesic words for the same group element can have different
secondary costs. A visited key must still identify the full group/action state;
the best secondary label is metadata attached to that state and depth. If cost
depends on prior moves rather than only the current state and edge, the state
must be augmented with the relevant history.

Cayley translation symmetry transfers the unweighted word metric only when the
secondary edge-cost rule is also translation invariant. State-dependent costs
can destroy that reduction.

## 10. GPU and multi-GPU boundary

First-arrival atomic visited claims are sufficient for arbitrary shortest-hop
parents, not for minimum-secondary parents. Exact hop-first secondary output
requires an equal-depth reduction over all shortest-predecessor candidates and
finalization before dependent expansion, or explicit repropagation.

On multiple GPUs, candidate predecessors may reside on different owners. The
authoritative owner must combine secondary values consistently; message timing
cannot be the semantic tie-break unless arbitrary output was requested.

Measurements should separate:

- generated candidates and shortest-DAG candidates;
- visited claims and equal-depth secondary relaxations;
- reduction/routing bytes and parent metadata;
- number of labels per state for Pareto variants;
- traversal time, secondary-DP time, and end-to-end time.

This is a correctness and cost decomposition, not a proposed optimized
pipeline.

## 11. Docker/Rust probe

`experiments/shortest_hop_secondary_cost_probe.rs` uses six vertices and
exhaustively enumerates all simple source-target paths. In Docker it produced:

```text
target=3 bfs_hops=Some(2) first_parent=Some(1) first_parent_cost=200
shortest_dag_secondary_cost=2
all_simple_path_pairs=[(2,2), (2,200), (3,0)]
pareto_pairs=[(2,2), (3,0)]
```

The first `rustfmt --check` gate failed and prevented execution. After applying
the formatter's exact changes, formatting, compilation, and execution passed.
The fixture is a semantic witness, not a performance result or solver.

## Sources

- E. Q. V. Martins, [*On a Multicriteria Shortest Path
  Problem*](https://doi.org/10.1016/0377-2217(84)90077-8), *European Journal of
  Operational Research* 16(2), 1984, for nondominated multicriteria path labels.
- G. Y. Handler and I. Zang, [*A Dual Algorithm for the Constrained Shortest
  Path Problem*](https://doi.org/10.1002/net.3230100403), *Networks* 10(4),
  1980, for shortest paths under an additional resource constraint.
- Notes 11, 13, 19, 20, 50, 57, 64, and 74 provide this repository's
  shortest-DAG, equal-distance metadata, shortlex, product-state, secondary
  objective, output-finalization, word-multiplicity, and first-claim boundaries.

## Takeaway

BFS determines the minimum number of edges. To minimize an additive secondary
cost among those paths, retain all depth-increasing shortest edges and reduce
over that DAG; a first-arrival parent is insufficient. Cost-first, constrained,
and Pareto objectives are different problems and may require weighted search,
augmented states, or several labels per vertex.

