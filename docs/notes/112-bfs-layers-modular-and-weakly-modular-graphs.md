# BFS layers, modular graphs, and weak modularity

Median graphs require one common interval vertex for every triple. Modular
graphs relax uniqueness, while weakly modular graphs replace the triple-median
condition by two local-looking rules on BFS layers. The rules are useful only
when their quantifiers and distance evidence are kept explicit.

No graph-class recognizer, optimizer, or GPU implementation is added here.

## 1. Triangle condition in BFS layers

Fix a root `u` and let `L_k(u)` be its BFS layer at distance `k`. The triangle
condition `TC(u)` says:

- if adjacent vertices `v,w` both lie in `L_k(u)`, with `k>0`,
- then they have a common neighbor `x` in `L_(k-1)(u)`.

Thus an edge lying horizontally inside a BFS layer can be completed downward to
a triangle whose third vertex is one step nearer the root.

The condition is not the statement that every same-layer pair has a common
predecessor. Adjacency of `v,w` is part of the premise. It is also stronger than
observing one convenient parent choice: `x` must exist in the graph, regardless
of which parents a BFS tree retained.

## 2. Quadrangle condition in BFS layers

The quadrangle condition `QC(u)` says:

- `v,w` are nonadjacent vertices at distance two;
- both lie in `L_k(u)`;
- they share a neighbor `z` in `L_(k+1)(u)`;
- then they also share a neighbor `x` in `L_(k-1)(u)`.

The upward two-edge wedge through `z` must therefore have a downward two-edge
wedge through `x`. This is a metric diamond condition, not merely the existence
of some four-cycle anywhere in the graph.

Exact layer numbers and complete adjacency are essential. A truncated successor
oracle can manufacture a false violation by omitting `x`; an approximate
distance can put a vertex in the wrong layer.

## 3. Weakly modular and modular graphs

A connected graph is weakly modular when `TC(u)` and `QC(u)` hold for every
root `u`. Although each premise describes a small diagram spanning nearby BFS
layers, the definition quantifies over every basepoint and uses global shortest
path distances.

A modular graph is equivalently a bipartite weakly modular graph. It can also be
defined by the interval statement

```text
I(a,b) intersect I(b,c) intersect I(c,a) is nonempty
```

for every triple `a,b,c`.

A median graph strengthens "nonempty" to "exactly one." Hence

```text
median graphs subset modular graphs subset weakly modular graphs.
```

The inclusions are strict.

## 4. Calibrating the strict inclusions

The complete bipartite graph `K_(2,3)` is modular but not median. Choose the
three vertices in the part of size three. Each pair is joined by a length-two
path through either of the two vertices in the other part. Both of those
vertices belong to all three intervals, so the triple has two medians.

The triangle `K3` is weakly modular but not modular. Relative to any root, its
two other vertices are adjacent in layer one and the root itself satisfies TC;
QC has no applicable distance-two pair. But the graph is not bipartite, and the
three pair intervals of its vertex triple have empty intersection.

These examples separate three statements:

- local BFS-layer triangles and diamonds exist;
- every triple has at least one median;
- every triple has exactly one median.

They must not be used interchangeably.

## 5. Why one BFS root is insufficient

A graph can satisfy `TC(u)` and `QC(u)` for one chosen root without satisfying
them for every root. Root-relative verification proves only weak modularity with
respect to that root.

Even for one root, a BFS parent array is insufficient evidence. TC and QC ask
whether particular pairs have any qualifying common predecessor. The selected
tree stores at most one predecessor per vertex and discards the other edges that
may close the required triangle or quadrangle.

A violation can be a compact certificate:

- TC: root, same-layer adjacent pair, and proof that no common neighbor lies one
  layer lower;
- QC: root, same-layer distance-two pair, their common upper neighbor, and proof
  that no common lower neighbor exists.

The positive "no neighbor exists" part requires exhaustive neighbor evidence
under the declared vertex identity and successor relation.

## 6. Local-to-global must retain its hypotheses

Weakly modular graph theory contains genuine local-to-global theorems using
local combinatorial conditions together with topological hypotheses on associated
triangle-square complexes. That does not mean an arbitrary bounded-radius BFS
sample proves the global class property.

The basic TC/QC definition already illustrates the issue: its diagrams are
small, but their layer labels depend on exact distances from every possible
root. Stronger local recognition claims must state their additional forbidden
configurations, connectedness, or simple-connectedness hypotheses explicitly.

## 7. Relations to Helly and hyperbolic geometry

Weakly modular graphs form a broad umbrella containing several important metric
classes, including median and Helly graphs, under their standard definitions.
Containment does not identify the properties:

- `K3` is weakly modular but not modular;
- `C4` is median but not ball-Helly;
- large Cartesian grids are median and weakly modular but have growing
  hyperbolicity;
- complete graphs are Helly and weakly modular but have large first frontiers.

Weak modularity therefore gives neither ball-Helly intersection, unique medians,
uniform thin triangles, nor a frontier-width bound by itself.

## 8. Cayley and generator dependence

Translation in a Cayley graph maps a rooted BFS layering to the corresponding
layering at another vertex. Therefore, for a fixed symmetric generating set, a
complete TC/QC verification relative to the identity can be transported to all
roots. This reduces root redundancy but not the need to check every applicable
configuration and exact word distance relative to the identity.

The class still depends on generators. For `Z2^2`:

- two coordinate generators give `C4`, which is median and modular;
- all three nonidentity generators give `K4`, which is weakly modular but not
  modular or median.

For `Z3`, the symmetric nonzero generator set gives `K3`, again weakly modular
but not modular. Thus group identity and vertex transitivity do not fix the
metric class.

A Schreier action may lack free Cayley translation and can change interval
structure through stabilizers. Directed positive alphabets use asymmetric
reachability distance, so the undirected TC/QC theory does not transfer without
a separately defined directed version.

## 9. GPU and multi-GPU evidence boundary

TC/QC checking can be expressed using BFS levels, adjacency intersections, and
existence reductions. That makes it parallelizable in principle, but it is not
ordinary BFS throughput:

- distance labels must first be exact;
- all relevant edges or successor occurrences must be complete;
- common-neighbor existence uses set intersections beyond parent arrays;
- a distributed absence claim needs a globally complete reduction;
- checking sampled roots/configurations can find counterexamples but cannot
  certify the universal property.

For Cayley graphs, translation may reduce the number of roots, but generator
expansion, exact deduplication, and configuration coverage remain separate costs.
Report distance construction, candidate diagram count, neighbor-intersection
work, and global reductions separately from frontier/visited BFS throughput.

## Sources

- J. Chalopin, V. Chepoi, H. Hirai, and D. Osajda,
  [*Weakly Modular Graphs and Nonpositive Curvature*](https://arxiv.org/abs/1409.3892),
  Memoirs of the AMS 268, 2020. TC/QC definitions, subclasses, and explicit
  local-to-global hypotheses.
- V. Chepoi,
  [*Graphs of Some CAT(0) Complexes*](https://doi.org/10.1006/aama.1999.0677),
  Advances in Applied Mathematics 24, 2000. Modular and median graph geometry
  in nonpositively curved complexes.
- H.-J. Bandelt and V. Chepoi,
  [*Metric Graph Theory and Geometry: A Survey*](https://doi.org/10.1090/conm/453/08795),
  Contemporary Mathematics 453, 2008. Relations among modular, median, Helly,
  and weakly modular graph classes.

