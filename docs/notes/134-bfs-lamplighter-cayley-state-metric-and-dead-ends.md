# BFS on lamplighter Cayley graphs: state, metric, and dead ends

The lamplighter group makes an implicit BFS state visibly larger than the
agent's current position. A vertex records both a lamp configuration and the
lamplighter position. A shortest word must pay for every required lamp change
and for a route that physically visits those lamps.

This produces a Cayley graph with fixed local degree, exponential state growth,
and dead ends caused by global configuration geometry rather than missing
successors. This note adds no optimizer, production implementation, benchmark,
or GPU code. A small exhaustive Rust probe checks two finite cyclic instances.

## 1. Exact state contract

The classical infinite binary lamplighter group is

```text
C_2 wr Z = (direct_sum over Z of C_2) semidirect_product Z.
```

Represent an element as `(F,p)`, where

- `F subset Z` is the finite set of lit lamps;
- `p in Z` is the lamplighter position.

The identity is `(empty,0)`. The finite-support condition matters: arbitrary
infinite binary strings are not group elements in this restricted wreath
product.

For a finite cyclic calibration, replace the street `Z` by `C_m`. Then there
are exactly

```text
m * 2^m
```

states.

## 2. Generator convention

This note uses the symmetric generating set

```text
a      : toggle the lamp at the current position,
t      : move one step right,
t^-1   : move one step left.
```

Each operation is a bijection of the complete state, so these are genuine
Cayley moves. Every vertex has three labeled outgoing moves, although on very
small cyclic bases two movement labels may share an endpoint.

Other common conventions combine operations, such as switch-then-walk or
walk-then-switch. They change word length, frontier layers, dead ends, and path
counts. The group alone does not determine the BFS metric; the generating set
does.

## 3. Position alone is not a vertex

The states

```text
(empty,0), ({0},0), ({1},0), ({0,1},0)
```

have the same lamplighter position but are distinct group elements. Conversely,
the same lamp set with different positions is also distinct.

A visited key containing only `p` loses reachable states and distances. A key
containing only `F` merges states with different legal next toggles and movement
costs. Exact identity is the pair `(F,p)` under the declared finite encoding.

## 4. Distance decomposes into changes plus routing

For two states `(F,x)` and `(G,y)`, let

```text
D = F symmetric_difference G.
```

Every lamp in `D` must be toggled an odd number of times; lamps outside `D`
must be toggled an even number of times. With unit toggle cost, an optimal word
toggles each required lamp exactly once. Movement must start at `x`, finish at
`y`, and visit every vertex of `D`.

Therefore

```text
d((F,x),(G,y)) = |D| + T_base(x,y;D),
```

where `T_base` is the shortest base-graph walk from `x` to `y` visiting all of
`D`. This is an open traveling-salesperson-path term, not ordinary base distance
unless the required set happens to lie on a shortest `x`-to-`y` route.

Lower bound: required toggles and movement edges are disjoint generator costs.
Upper bound: follow an optimal visiting walk and toggle each required lamp on
its first visit.

## 5. Closed form on the infinite line

From the identity to `(F,p)` on `Z`, set

```text
L = min(F union {0,p}),
R = max(F union {0,p}).
```

Any visiting walk must cover the interval extremes. It can visit `L` first or
`R` first, giving movement cost

```text
min(
  |0-L| + (R-L) + |R-p|,
  |0-R| + (R-L) + |L-p|
).
```

Word length is this route cost plus `|F|`. The simple formula relies on the base
being a line. For a general base Cayley graph, the visiting-route term can be a
substantially harder problem and must not be assumed to have the same form.

## 6. The graph is not a Cartesian product

One might try to view the state space as

```text
lamp hypercube x base position.
```

The vertex set has that product shape, but adjacency is coupled: the toggle
coordinate depends on the current position, and moving changes which lamp the
same label `a` will affect next. Algebraically this is a semidirect/wreath
product, not an independent Cartesian product.

Consequently note 69's additive Cartesian-product frontier convolution does not
apply. The metric is additive only after solving the coupled visiting-route
problem.

## 7. Why exponential growth appears over a line

The base street has linear balls, but a lamplighter word of length `r` can visit
an interval and leave many subsets of its visited lamps lit. Position choices
and configuration choices multiply.

Thus a base graph with slow volume growth can produce a wreath-product Cayley
graph with exponential growth. This is compatible with note 94: exponential
growth does not by itself imply nonamenability.

The BFS frontier records complete configurations at exact word length, not just
the positions reachable after `r` walking steps.

## 8. Dead ends despite regular degree

A dead end relative to the identity is a state whose every generator neighbor
has distance no greater than its own. It still has all three moves. The problem
is radial: each immediate move either spends effort undoing part of an optimal
route or changes a lamp in a way that does not yet escape the current ball.

Cleary and Taback show that infinite lamplighter groups with natural generating
sets contain dead ends of arbitrarily large depth. This expands note 72's brief
example: the configuration can force a long retreat before any state farther
from the identity becomes reachable.

A dead end does not terminate global BFS. Other states in the same frontier can
still have outward neighbors.

## 9. Finite cyclic calibration

`experiments/lamplighter_bfs_probe.rs` exhaustively traverses `C_2 wr C_4` and
`C_2 wr C_5` with the generator convention above. Independently for every
state, it computes the shortest cycle walk that visits the lit-lamp set and
checks

```text
BFS distance = popcount(lamps) + visiting-route length.
```

Observed in Docker with Rust 1.85.1:

```text
C2wrC4 states=64 diameter=8 decomposition_mismatches=0
layers=[1,3,5,8,11,13,13,8,2]
interior_dead_ends=5

C2wrC5 states=160 diameter=10 decomposition_mismatches=0
layers=[1,3,6,10,16,24,31,32,23,11,3]
interior_dead_ends=18
```

The dead-end count excludes diameter-layer states, so these are genuine
interior radial dead ends in the finite calibration. This is exhaustive evidence
only for `m=4,5`, not a general counting theorem.

## 10. Frontier meaning in this Cayley graph

For the identity root,

```text
F_k = {(F,p) : |F| + T_base(0,p;F) = k}.
```

Unlike a plain walk on the base cycle or line, one frontier mixes states with
different numbers of lit lamps and different route lengths. The same `k` can be
split many ways between switching and walking.

Useful diagnostics therefore include a joint histogram by

```text
(BFS depth, lamp popcount, route cost, cursor position),
```

not just frontier size. This is a semantic observation, not a request to build
an optimized histogram kernel.

## 11. Path multiplicity and commuting actions

Several words can reach the same `(F,p)`:

- independent lamp toggles can be visited in different route orders;
- a lamp can be toggled redundantly twice;
- movement can backtrack;
- group relations identify different generator words.

Exact visited deduplicates the final group state, while shortest-path DAG output
may retain multiple optimal parents. Counting generator histories, visiting
routes, shortest words, and distinct states gives different numbers.

## 12. Bidirectional and reverse search

Because the declared generator set is symmetric, backward BFS uses the same
three labeled move types. But a meeting key must include both configuration and
position. Two waves at the same cursor coordinate with different lamp sets have
not met.

The distance between arbitrary endpoints depends on `F symmetric_difference G`,
so translating one endpoint to the identity is valid group symmetry. Merely
XORing lamp masks without transforming cursor coordinates under the same group
convention is not a complete translation proof.

## 13. Finite encoding versus infinite completeness

On `C_m`, a bit mask plus cursor is a complete finite rank and exhaustive BFS
terminates. On `Z`, every individual state still has finite support and finite
encoding, but the Cayley graph is infinite. A completed finite-radius frontier
is not an exhaustion certificate.

Bounding lamp positions to a window silently changes the graph unless boundary
escape is represented as unknown. Note 42's three-valued bounded-lookup
semantics applies.

## 14. GPU and multi-GPU boundary

The finite cyclic model has regular, cheap-looking successors:

- toggle one bit;
- increment or decrement a cursor modulo `m`.

That does not prove a useful high-throughput implementation. The state count is
`m2^m`, so authoritative visited storage can dominate long before successor
generation. Duplicate patterns depend on group relations and frontier
composition, while owner balance depends on the exact rank/hash partition.

A multi-GPU study should separate:

- complete state bytes and key equality;
- three generated labeled candidates per frontier state;
- local duplicates and cross-owner duplicate convergence;
- frontier composition by popcount/route cost/cursor;
- visited capacity and load factor;
- owner routing and global level completion;
- logical generator edges and physical interconnect paths;
- end-to-end time and isolated primitive time.

The traveling-route formula is a semantic oracle for validation on simple base
graphs, not evidence that BFS should be replaced by a general TSP solver.

## Sources

- J. Taback,
  [*Lamplighter Groups*](https://academic.oup.com/princeton-scholarship-online/book/13467/chapter-abstract/166977230),
  in *Office Hours with a Geometric Group Theorist*, Princeton University
  Press, 2017, for configuration-position state, generators, word length, and
  Cayley geometry.
- E. Silva,
  [*Dead ends on wreath products and lamplighter groups*](https://arxiv.org/abs/2206.08775),
  2022, for the standard word-length decomposition into lamp-change cost and a
  traveling-salesperson path on the base Cayley graph.
- S. Cleary and J. Taback,
  [*Dead end words in lamplighter groups and other wreath products*](https://arxiv.org/abs/math/0309344),
  2005, for arbitrary-depth dead ends and related Cayley geometry.
- Notes 16, 20, 23, 28, 35, 42, 61, 64, 69, 72, 93, and 94 supply this
  repository's Cayley action, product state, word-tree, identity, growth,
  bounded lookup, relations, multiplicity, Cartesian-product, dead-end,
  generator-metric, and amenability boundaries.

## Takeaway

A lamplighter BFS vertex is the complete pair `(lamp configuration, cursor)`.
With separate toggle and movement generators, distance is required lamp-change
cost plus the shortest base walk visiting those lamps. This coupled metric is
not a Cartesian product. Regular degree does not prevent interior or arbitrarily
deep dead ends, and cheap implicit successors do not remove exponential visited
state growth.
