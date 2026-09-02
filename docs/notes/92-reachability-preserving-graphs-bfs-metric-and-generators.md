# Reachability-preserving graphs, BFS metric, and generators

Transitive closure and transitive reduction preserve which vertices can reach
which others. BFS usually asks more: how many unit transitions are needed,
which layer discovers a state, which shortest witnesses exist, and how much
frontier work appears at each depth. Reachability equivalence preserves none of
those quantities in general.

This note studies that separation and its Cayley-generator analogue. It does
not implement closure, reduction, or generator optimization.

## 1. Three graph objects

For a directed graph `G=(V,E)`:

- its transitive closure `G+` contains `u->v` whenever `v` is reachable from
  `u` by a nonempty path in `G`;
- a transitive reduction preserves the same reachability relation with a
  minimal arc representation;
- the original graph records which transitions count as one step.

For a finite DAG, the transitive reduction is unique. Directed graphs with
cycles require more care and need not share this simple uniqueness statement.
In every case, the defining preserved object is reachability, not distance.

## 2. The chain-to-closure counterexample

Take the directed chain

```text
v_0 -> v_1 -> ... -> v_(n-1).
```

BFS from `v_0` has one vertex in every layer and reaches `v_(n-1)` at distance
`n-1`. Its transitive closure adds `v_i->v_j` for every `i<j`. BFS from `v_0`
then discovers all other `n-1` vertices at distance one.

Both graphs have exactly the same reachable ordered pairs, but:

```text
diameter:          n-1  versus  1
root frontier:     1    versus  n-1
number of levels:  n    versus  2 including the root
```

Thus even the coarsest BFS geometry can change arbitrarily under a
reachability-preserving transformation.

## 3. Monotonicity that does survive

Adding unit arcs cannot increase shortest-path distance:

```text
d_(G+E')(u,v) <= d_G(u,v)
```

whenever the original distance is finite. Deleting arcs while preserving
reachability cannot decrease distance. Neither inequality is generally an
equality.

The final reachable set from each source is preserved by transitive
closure/reduction. Consequently, a complete traversal may end with the same
visited vertex set while every intermediate frontier, parent, distance, path
count, and work profile differs.

## 4. Dominators and disjoint routes can change

In the chain

```text
s -> a -> b -> t,
```

both `a` and `b` dominate `t`. Adding only the reachability-redundant shortcut
`s->t` preserves the fact that `t` is reachable but destroys both nontrivial
dominators.

The shortcut also creates a path internally disjoint from the old chain.
Therefore reachability-equivalent graphs need not have the same:

- dominator or postdominator trees;
- minimum vertex/arc separators;
- internally disjoint path counts;
- shortest-path DAGs or shortest gateways.

Reachability says that at least one route exists. These objects describe the
shape and redundancy of routes.

## 5. SCCs survive, condensation distance does not

Any two graphs with exactly the same directed reachability relation have the
same strongly connected component partition: mutual reachability is unchanged.
They also induce the same reachability partial order between SCCs.

But the condensation DAG's arc set can be a reduction, an original
representation, or a closure of that order. BFS distance in the condensation
therefore changes just like it does in a chain. The SCC quotient preserves
reachability classes; it does not select a canonical intercomponent unit
metric.

## 6. Graph powers and closure

Note 26's `k`-hop graph power inserts a macro-edge for an original path of up to
or exactly `k` steps, according to its contract. Transitive closure is the
unbounded reachability limit: every finite-length reachable pair becomes one
arc.

Calling every inserted macro-edge unit cost deliberately changes the metric.
If a shortcut `u->v` instead receives weight `d_G(u,v)`, it cannot improve any
original shortest distance: replacing it by an original shortest path has the
same cost. This weighted preservation requires the exact old distance and a
replayable witness; an arbitrary macro cost does not suffice.

## 7. Cayley generator redundancy is metric relevance

Let finite symmetric generating sets `S` and `T` generate the same group. Their
Cayley graphs have the same vertex set and are connected to the same states,
but their word metrics can differ.

If `S subseteq T` and every `t in T` has an `S`-word of length at most `L`, then

```text
d_T(g,h) <= d_S(g,h) <= L d_T(g,h).
```

The first inequality follows because every `S` move remains available. For the
second, replace each `T` move in a shortest `T` word by its at-most-`L` `S`
word. A generator that is algebraically redundant can therefore still shorten
distances by a factor approaching its old word length.

At the extreme, adding every nonidentity group element as a generator turns a
finite Cayley graph into a diameter-one complete directed graph while preserving
the generated group. "Same group" is analogous to "same reachability," not
"same BFS problem."

For positive-only alphabets in infinite groups or monoids, even reachability
needs a semigroup-level check; equality of generated groups after allowing
formal inverses is not enough.

## 8. Frontier and duplicate consequences

Adding a macro generator can simultaneously:

- reduce depth and diameter;
- increase branching and candidate count;
- move a large volume of states into earlier frontiers;
- create new short relations and duplicate convergence;
- change parent words and tie multiplicity;
- alter peak memory and communication in either direction.

No single direction follows for peak frontier. The chain closure reduces
depth but makes the first frontier huge.

Total generated transitions need a traversal contract. In an exhaustive BFS
that expands each reached state once and scans every outgoing occurrence,
including those of the last frontier, the count is
`sum_{v in R} outdegree(v)`. Adding edge occurrences while preserving the
reached set `R` cannot decrease that count. With `q` total generator actions
available at every state, it is exactly `q*|R|`; adding generator labels
increases this count even when their endpoints duplicate existing moves.

A target-stopped search can instead expand fewer states after a shortcut is
added, so its total candidate count can decrease. State the stopping rule and
expansion schedule before comparing such runs. Peak
memory, communication, and elapsed time still require measurement under the
declared generator set, not just the abstract generated group.

## 9. Distributed and GPU evidence boundary

- Equal final visited cardinality validates neither distances nor level traces.
- A reduced edge store may preserve reachability while increasing BFS depth and
  synchronization rounds.
- A closure-like store may reduce rounds while exploding adjacency and memory.
- A macro move's generation cost, wire representation, and replay expansion
  belong to its semantic edge contract.
- Comparing two runs with different generator sets is comparing different unit
  graphs, even if both enumerate the same Cayley states.
- Multi-GPU speedup cannot be attributed solely to an implementation when the
  graph representation changed its work and span.

These are reporting constraints, not a recommendation to build either
representation.

## 10. Evidence checklist

1. Preserved reachability, distance, weighted distance, paths, or only endpoint
   states.
2. Original, transitively reduced, powered, or transitively closed graph.
3. Unit shortcut or weighted/replayable macro transition.
4. Full visited set versus per-level frontier and parent evidence.
5. SCC partition versus condensation edge metric.
6. Generated group, positive semigroup, and exact generator alphabet.
7. Generator labels and macro words included in output identity.
8. Work, span, memory, and communication compared on the same graph contract.

## Sources

- A. V. Aho, M. R. Garey, and J. D. Ullman,
  [*The Transitive Reduction of a Directed Graph*](https://doi.org/10.1137/0201008),
  SIAM Journal on Computing 1(2), 1972, 131-137. Reachability-preserving
  definition, DAG uniqueness, and cyclic-graph qualification.
- Notes 06, 10, 21, 26, 35, 68, 84, 89, 90, and 91 provide Cayley-generator,
  frontier-growth, diameter, graph-power, growth-series, generator-change, SCC,
  dominator, separator, and postdominator context.

## Takeaway

Reachability equivalence is far weaker than BFS equivalence. Closure, reduction,
shortcuts, and redundant Cayley generators may leave the reachable state set
unchanged while rewriting distances, layers, path witnesses, separators,
dominators, work, and parallel span. The unit transition relation is part of the
problem, not merely a storage choice.
