# Resolving sets, metric dimension, and BFS coordinates

Note 78 treated BFS distance arrays as metric coordinates. This raises a
different question from pair-distance bounds:

> When does a collection of exact BFS coordinates uniquely identify every
> vertex?

The answer is expressed by resolving sets and metric dimension. This is an
information question about BFS outputs, not an implementation strategy.

## 1. Distance vectors and resolving sets

Let `G` be a finite connected undirected graph and let the ordered landmark set
be `W=(w_1,...,w_q)`. Define the metric representation

```text
r_W(v) = (d(v,w_1), ..., d(v,w_q)).
```

`W` is a **resolving set** when `r_W` is injective: every distinct pair `u,v`
differs in at least one coordinate. A minimum resolving set is a metric basis;
its cardinality is the **metric dimension** `beta(G)`.

Each coordinate can be constructed by one complete ordinary BFS rooted at its
landmark. Exactness of every coordinate does not imply injectivity of the
combined vector: that is a separate global property to prove or check.

## 2. Joint multi-source BFS is not the coordinate vector

Seeding all landmarks together computes

```text
d(v,W) = min_i d(v,w_i),
```

possibly with a nearest-source label or tie metadata. It does not compute
`r_W(v)`. Taking a minimum is a many-to-one projection of the coordinate
vector.

On path `0--1--2--3` with landmarks `W=(0,3)`:

```text
v             0      1      2      3
r_W(v)       (0,3)  (1,2)  (2,1)  (3,0)
d(v,W)        0      1      1      0
```

The two landmarks resolve every vertex, but their joint multi-source distance
merges `0` with `3` and `1` with `2`. Source labels can distinguish some of
these cases, but one selected label is still not the full distance vector and
tie policy changes its semantics.

## 3. Why one landmark characterizes a path

Suppose one vertex `w` resolves a connected graph with `n` vertices. At most
one vertex may occur at each distance from `w`. Connectivity implies that all
integer layers from zero through the eccentricity of `w` are nonempty, hence
each contains exactly one vertex and the eccentricity is `n-1`.

Every consecutive pair of layers must be adjacent to make the farther vertex
reachable. An edge cannot skip a layer because adjacent vertices' distances
from `w` differ by at most one. There cannot be an additional same-layer edge
because each layer has one vertex. Therefore the graph is exactly a path and
`w` is an endpoint.

Conversely, distances from an endpoint of a path are `0,1,...,n-1`, so they
resolve the path. Thus, for nontrivial connected graphs,

```text
beta(G)=1  iff  G is a path.
```

A finite connected undirected Cayley graph is vertex-transitive. A path is
vertex-transitive only for one or two vertices. Hence every such Cayley graph
with more than two vertices has metric dimension at least two.

## 4. Three calibration families

### Path `P_n`

One endpoint resolves all vertices, so `beta(P_n)=1` for `n>=2`.

### Cycle `C_n`

One landmark cannot resolve a cycle with `n>=3`: vertices on opposite sides
can share its distance. Two suitably chosen landmarks resolve the cycle, so
`beta(C_n)=2`. For an even cycle, an antipodal pair is not suitable because
reflection across that axis leaves equal vectors; landmark placement matters,
not only count.

### Complete graph `K_n`

Any vertex outside `W` has coordinate vector `(1,...,1)`. Therefore at most one
vertex may be omitted and `|W|>=n-1`. Taking any `n-1` vertices works, giving
`beta(K_n)=n-1`.

These examples separate size, diameter, and metric dimension. A long path has
dimension one; a diameter-one complete graph needs almost every vertex.

## 5. A counting lower bound

For a connected graph with `n>1` vertices and diameter `D>=1`, each undirected
coordinate lies in `{0,...,D}`. With `q` landmarks there are at most `(D+1)^q`
possible vectors. Injectivity on `n` vertices therefore requires

```text
n <= (D+1)^q,
q >= ceil(log(n) / log(D+1)).
```

This bound is necessary but often weak because not every formal vector is
geometrically realizable. For `K_n`, `D=1`, so it gives
`q>=ceil(log_2(n))`, whereas the exact answer is `n-1`. For example, on `K_4`
the counting bound is two landmarks but the exact minimum is three.

The same counting idea must be reformulated for directed graphs with a declared
finite-distance/unreachable alphabet. Treating infinity as an ordinary finite
coordinate without specifying reachability semantics hides information.

## 6. Symmetry obstruction

Let an automorphism `phi` fix every landmark in `W` pointwise. Since
automorphisms preserve distance,

```text
r_W(phi(v)) = r_W(v).
```

If `W` resolves the graph, this forces `phi(v)=v` for every vertex. Therefore
the pointwise stabilizer of a resolving set is trivial.

The converse need not hold: destroying all global automorphisms does not prove
that no unrelated pair has the same distance vector. A distinguishing set and
a resolving set answer different questions.

For Cayley graphs, translating both the landmark tuple and queried vertex by
the same group element preserves the entire coordinate tuple. Thus every left
translate of a metric basis is another metric basis. This homogeneity does not
make one landmark sufficient; it only moves equivalent bases around the graph.

## 7. Reusing an identity table in a Cayley graph

In a genuine right-action Cayley graph,

```text
d(w_i,v) = d(e,w_i^-1 v).
```

Therefore a complete identity-rooted table keyed by exact group elements can
synthesize every coordinate of `r_W(v)` without running a new BFS per landmark.
This statement assumes the candidate vertex `v` and group operations are
already known. It does not say the scalar identity distance alone identifies
an unknown group element: many elements can lie on the same sphere.

For a bounded identity table, a missing relative element makes that coordinate
unknown beyond the stored radius. For a Schreier action, note 78's stabilizer
warning applies and the relative-element lookup may not represent the intended
state distance.

## 8. Validation contract

A claim that BFS landmarks identify states should record:

1. graph snapshot, vertex equality, direction, and reachability convention;
2. ordered landmark set and whether coordinates are `d(w,v)` or `d(v,w)`;
3. whether each BFS is complete or depth bounded;
4. independent coordinate arrays versus one multi-source minimum;
5. evidence that all coordinate vectors are unique;
6. handling of unreachable/unknown entries;
7. any quotient, stabilizer, or symmetry semantics;
8. whether the goal is identification, pair-distance bounds, or reconstruction.

## Sources

- R. A. Melter and F. Harary, *On the Metric Dimension of a Graph*, Ars
  Combinatoria 2 (1976), 191-195. One of the original formulations of metric
  dimension.
- R. A. Slater, *Leaves of Trees*, Proceedings of the Sixth Southeastern
  Conference on Combinatorics, Graph Theory, and Computing (1975), 549-559.
  Introduced the closely related locating-set formulation.
- M. Fehr, S. Gosselin, and O. R. Oellermann,
  [*The metric dimension of Cayley digraphs*](https://doi.org/10.1016/j.disc.2005.09.015),
  Discrete Mathematics 306 (2006), 31-41. Defines resolving sets for directed
  Cayley graphs and studies their metric dimension.
- Notes 13 and 78 provide the multi-source-minimum and BFS-landmark contracts
  used here.

## Takeaway

Several exact BFS arrays form a coordinate map, but exact coordinates are not
automatically identifying coordinates. A resolving set makes that map
injective; metric dimension measures the fewest roots needed. Joint
multi-source BFS computes only the minimum coordinate and generally destroys
this information. Cayley homogeneity permits coordinate reuse through relative
elements, while still leaving sphere collisions and Schreier semantics intact.
