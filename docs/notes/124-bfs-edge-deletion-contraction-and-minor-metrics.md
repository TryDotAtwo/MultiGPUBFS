# BFS under deletion, contraction, and graph minors

BFS distances are monotone under some graph edits, but the direction depends on
the edit. Deleting transitions removes candidate paths; adding or contracting
transitions creates shortcuts in the represented metric. A graph minor can do
both, so minor containment alone is not a distance-preservation theorem. This
note studies those contracts and adds no implementation, optimizer, benchmark,
or GPU code.

## 1. Deletion and subgraph monotonicity

Let `H` be a subgraph of `G`, and let `u,v` survive in the same component of
`H`. Every `H` path is also a `G` path, hence

```text
d_G(u,v) <= d_H(u,v).
```

If deletion disconnects them, the right side is infinity. Therefore, for a
fixed source and shared vertex set,

```text
B_r^H(s) subset B_r^G(s).
```

Exact BFS in a subgraph produces valid original-graph paths and upper bounds on
original distance. It gives exact original distance only when an isometry,
spanner equality, or independent lower bound closes the gap, as notes 115 and
116 explain.

## 2. Addition and supergraph monotonicity

If `G` is a subgraph of `H`, every old path remains available and new edges may
create shortcuts:

```text
d_H(u,v) <= d_G(u,v),
B_r^G(s) subset B_r^H(s).
```

A supergraph BFS distance is therefore a lower bound on the original graph's
distance, but its path may use added edges and need not replay in the original
graph.

Deletion and addition give opposite inequalities. Saying only that one graph
is a "simplified" version leaves the bound direction unknown.

## 3. Contracting one edge

Contract edge `a-b` into one supervertex `z`, deleting loops and optionally
merging parallel simple edges. Let `q` map original vertices to contracted
vertices. Projection of every original path gives

```text
d_(G/ab)(q(u),q(v)) <= d_G(u,v).
```

For a single unit edge contraction, the decrease for fixed original endpoints
is at most one:

```text
d_G(u,v)-1 <= d_(G/ab)(q(u),q(v)) <= d_G(u,v).
```

To lift a shortest contracted path, replace its one visit to `z`, when
necessary, by traversal across `a-b`. A simple shortest path visits `z` at most
once, so at most one unit is restored.

The bound includes `u=a,v=b`: original distance one becomes zero because their
images coincide.

## 4. Contracting connected clusters

Partition vertices into connected clusters and form a quotient `Q` with one
supervertex per cluster and an edge for every surviving cross-cluster adjacency.
For `q:V(G)->V(Q)`,

```text
d_Q(q(u),q(v)) <= d_G(u,v).
```

Suppose every cluster has original diameter at most `Delta`. A quotient path of
`k` edges can be lifted between fixed original endpoints by crossing the `k`
intercluster edges and moving inside at most `k+1` clusters:

```text
d_G(u,v) <= k + (k+1)*Delta.
```

With cluster-specific diameters, replace the last term by the sum of diameters
of the visited clusters. This is a coarse lifting bound, not a claim that every
cluster actually incurs its worst case.

Without a cluster-diameter or stored intra-cluster path guarantee, a short
quotient path can hide arbitrarily long original motion.

## 5. Coarse BFS balls overreach

For a contraction quotient,

```text
q(B_r^G(s)) subset B_r^Q(q(s)).
```

The reverse inclusion need not hold for fixed radius: a quotient vertex can be
`r` intercluster hops away while every desired concrete representative requires
additional intra-cluster movement.

This contrasts with note 123's graph covers, where lifted balls project exactly
onto base balls. A cover is locally bijective and duplicates fibers; a
contraction deliberately collapses local distance inside fibers to zero.

## 6. What a graph minor means

A graph `H` is a minor of `G` when it can be obtained through vertex deletion,
edge deletion, and edge contraction. Equivalently, vertices of `H` can be
represented by disjoint connected branch sets in a subgraph of `G`, with each
minor edge witnessed by an original edge between the corresponding branch
sets.

The operations pull distance in opposite directions:

- deletion can increase distance or disconnect;
- contraction can decrease distance or identify endpoints.

Therefore no universal inequality compares `d_H` with `d_G` merely from
`H minor G`. The minor relation preserves structural containment, not the
original unit shortest-path metric.

## 7. Two counterexamples to minor-distance monotonicity

### Increase by deletion

Delete one edge of `C_4`. Its endpoints had distance one in the cycle and
distance three in the remaining path. The resulting path is a minor whose
corresponding distance increased.

### Decrease by contraction

Contract the middle edge of the four-vertex path `0-1-2-3`. Endpoint distance
drops from three to two. Contracting all three edges drops it to zero.

Thus a minor can be either farther or nearer depending on its model. Even two
minor models of the same abstract `H` can encode different original branch
sets and lifting costs.

## 8. Minor paths need branch-set witnesses

A minor path

```text
C_0, C_1, ..., C_k
```

only certifies `k` cross-branch adjacencies. To replay it in `G`, retain:

1. the original edge witnessing each `C_i-C_(i+1)` adjacency;
2. a path inside every intermediate connected branch set joining its incoming
   and outgoing witness endpoints;
3. paths from the requested concrete endpoints to the first and last witnesses.

Without these witnesses, the minor path is a lower-resolution route, not an
original move sequence. Its hop count omits within-branch travel.

This resembles emulator/hopset unpacking from note 116, but minor edges are
original cross edges while the zero-cost contraction of branch interiors is
the source of distortion.

## 9. Distances are not the only changed output

Contraction can also change:

- shortest-path multiplicity by merging endpoints or parallel routes;
- girth by collapsing cycles or creating loops before simplification;
- degree and frontier width;
- bipartiteness if loops are removed or odd cycles shorten;
- parent and predecessor-DAG structure;
- state and edge identity.

Suppressing loops and parallel edges preserves simple-graph vertex distances
after the contraction but can destroy labeled-edge and path-count semantics.
The simplification convention must be part of the graph epoch.

## 10. Cayley quotient groups

Let `N` be a normal subgroup of a generated group `G`. Mapping elements to
cosets gives a quotient Cayley graph for the image generators, subject to
declared loop and duplicate-generator conventions. Its word distance satisfies

```text
d_(G/N)(Ng,Nh) <= d_G(g,h).
```

More precisely, quotient distance is distance to a coset fiber:

```text
d_(G/N)(Ng,Nh) = min_(n in N) d_G(g,nh)
```

under compatible left/right conventions. A quotient word lifts to some member
of the target coset, not automatically the fixed target `h`. Reaching `h` may
require an additional kernel word.

If a generator lies in `N`, it becomes a loop; distinct generators may acquire
the same quotient image. Collapsing those transitions preserves the scalar
simple quotient metric but changes label and multiplicity semantics.

## 11. Schreier and arbitrary state aggregation

For a Schreier action, aggregating states into connected blocks again produces
a contraction-like coarse graph only if every coarse adjacency is witnessed
and all relevant states are assigned consistently. An arbitrary canonicalizer
may instead be an orbit quotient, homomorphism, or invalid transition merge;
note 17's audit remains necessary.

Local contraction of a few states generally breaks Cayley/Schreier symmetry.
A global quotient by invariant blocks may retain an algebraic action but changes
the target from a concrete state to a block unless lifting is supplied.

## 12. Bounds and exact-search use

A contraction quotient can provide a lower bound on original distance. A
replayable lifted path provides an upper bound. Exactness follows only when the
bounds meet:

```text
coarse lower bound = replayable original upper bound.
```

A minor name alone supplies neither the correct lower bound direction nor a
replay witness, because deletions may also have occurred. The precise sequence
or branch-set model must identify which quotient and subgraph inequalities are
valid.

## 13. GPU and multi-GPU boundary

Coarsening may reduce stored vertices and BFS rounds, but it changes the metric
problem. Report separately:

- cluster/branch-set construction and validation;
- coarse vertices, edges, loops, and parallel-edge policy;
- quotient BFS distance and frontier profile;
- intra-cluster lifting and replay cost;
- original upper-bound validation;
- exactness gap between lower and upper bounds;
- owner assignment and cross-cluster communication before and after coarsening;
- original versus coarse traversal throughput.

In multi-GPU execution, contracting across owner boundaries changes ownership
and routing. Treating every supervertex as one unit-cost original state is a
semantic change, not merely a storage optimization.

## Sources

- R. Diestel,
  [*Graph Theory*, Chapter 1.7: Contraction and Minors](https://diestel-graph-theory.com/),
  Graduate Texts in Mathematics 173. Gives the standard contraction, branch-set,
  and minor definitions.
- A. Bernstein, K. Daubel, Y. Disser, M. Klimm, T. Mutze, and F. Smolny,
  [*Distance-Preserving Graph Contractions*](https://arxiv.org/abs/1705.04544),
  studies explicit distortion guarantees for contracted graph metrics.
- Notes 17, 22, 83, 92, 115, 116, 118, and 123 supply this repository's
  quotient, dynamic, bridge, reachability, isometry, spanner, failure, and cover
  distinctions.

## Takeaway

Deletion makes surviving shortest paths no shorter; addition and contraction
make represented distances no longer. A graph minor mixes these directions and
therefore carries no general BFS-distance inequality. A contraction quotient
is useful as a lower-resolution lower bound only when its clusters, internal
diameters, cross-edge witnesses, target fibers, and replay semantics are all
declared.
