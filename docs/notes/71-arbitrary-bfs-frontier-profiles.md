# How arbitrary can a BFS frontier profile be?

The familiar picture of a frontier growing to one peak and then shrinking is
not a theorem for general graphs. This note makes the failure precise. It also
separates unrestricted graph geometry from the additional constraints imposed
by bounded degree, regularity, and Cayley symmetry.

## Realization theorem

Let

```text
a_0,a_1,...,a_D
```

be any finite sequence of positive integers with `a_0=1`. There is a finite
connected simple undirected tree with a root `s` whose exact BFS sphere sizes
are precisely those numbers.

### Construction

Create disjoint vertex sets `F_i` with `|F_i|=a_i`; let `F_0={s}`. For every
`i<D`, choose one anchor vertex `p_i` in `F_i` and join every vertex of
`F_(i+1)` to `p_i`. Add no other edges.

Every non-root vertex has exactly one parent in the preceding layer. The graph
is connected and has one fewer edge than vertices, so it is a tree. Every path
from the root to `F_i` follows the layer chain and has length `i`; there are no
edges that skip a layer. Therefore exact BFS returns `|F_i|=a_i`.

The sequence can rise, fall, rise again, or oscillate repeatedly. For example,
`1,100,1,100,1` is a valid tree profile. Neither cycles nor duplicate endpoint
convergence are required to produce multiple frontier peaks.

## The only unrestricted termination constraint

For a source-rooted finite or infinite graph, once an exact frontier is empty,
every later frontier is empty. A profile such as

```text
1,5,0,7
```

is impossible: the empty `F_2` proves that the reached ball has no outgoing
edge to an unreached vertex. Apart from `a_0=1`, positivity before termination,
and total cardinality, unrestricted finite connected graphs impose no shape
condition on the sphere sizes.

For multiple sources, replace `a_0=1` by the number of distinct sources. The
same layered construction works after assigning every first-layer vertex to
one source and ensuring that all sources belong to the intended component.

## Bounded-degree constraints

Suppose the graph is simple, undirected, and has maximum degree `Delta`.
The root gives

```text
a_1 <= Delta.
```

Every vertex in `F_i`, for `i>=1`, needs at least one incident edge toward
`F_(i-1)`. It therefore has at most `Delta-1` remaining edge slots toward the
next layer. Since every vertex in `F_(i+1)` needs at least one predecessor,

```text
a_(i+1) <= (Delta-1) a_i.
```

These inequalities are also sufficient for realization by a rooted tree:
distribute the `a_(i+1)` children among the `a_i` parents, never assigning more
than `Delta-1` children to a non-root parent or more than `Delta` to the root.

Thus a degree bound limits each upward jump, but still does not impose
unimodality. With `Delta>=3`, a layer may shrink enough to leave a few vertices
with spare child capacity and then expand again within the ratio bound.

For a directed graph with maximum out-degree `Delta`, the corresponding crude
bound is `a_(i+1)<=Delta a_i`; an incoming predecessor does not consume an
outgoing slot. Direction conventions therefore change even this elementary
profile constraint.

## Regular and Cayley graphs are stricter families

The construction deliberately concentrates a whole next layer under one
high-degree anchor. It does not preserve regularity or vertex transitivity.
Consequently it proves what arbitrary graphs can do, not what a fixed-degree
Cayley family can do.

In an inverse-closed `q`-generator Cayley graph, every vertex has the same
labeled out-degree. Relations and visited history decide how those occurrences
split among:

- the preceding ball;
- the current layer;
- repeated endpoints outside the ball;
- unique states in the next frontier.

The degree bound gives only the tree upper envelope. It does not determine
which sphere profiles are realizable by groups, which are unimodal, or where a
finite Cayley graph saturates.

## What frontier cardinalities fail to reveal

The same layer profile can be realized by graphs with very different internal
work. Starting from the layered tree, one may add edges within a layer without
changing any root distance. One may also add several edges from `F_i` to the
same vertex of `F_(i+1)`, creating candidate convergence while leaving all
sphere sizes unchanged, provided no edge skips layers.

Therefore a frontier sequence alone does not identify:

- edge occurrence counts;
- number of same-level edges;
- number or multiplicity of shortest parents;
- duplicate pressure;
- a unique graph, relation family, or memory-access pattern.

This strengthens the earlier cardinality warning: not only can two state sets
with equal sizes differ, but the entire exact sequence of sizes can agree while
the traversal work and shortest-path DAG differ substantially.

## Practical interpretation

1. A few initial growth ratios provide no universal upper bound on a later
   frontier in an unrestricted graph.
2. A frontier that shrinks is not necessarily near exhaustion; a narrow bridge
   may precede another large region.
3. Multiple memory peaks are mathematically ordinary, even in trees.
4. Degree supplies a local growth ceiling, not a predicted profile.
5. Stronger forecasts require declared structure: regularity, expansion,
   product form, growth series, isoperimetry, or application-specific evidence.
6. Hardware capacity planning from a frontier prefix is an empirical gamble
   unless one of those structural bounds is actually proved.

No executable experiment is needed for the realization theorem: the
construction itself is a witness for every admissible finite sequence.

