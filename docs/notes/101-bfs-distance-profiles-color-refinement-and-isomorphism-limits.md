# BFS distance profiles, color refinement, and isomorphism limits

A complete BFS from a root produces a distance label for every reachable
vertex. Compressing those labels to layer sizes gives a useful radial profile,
but discards vertex identities. Neither representation specifies the full
adjacency relation.

This note asks exactly what such profiles can certify, how they differ from
color refinement, and why neither is a replacement for exact state identity.
It adds no graph-isomorphism implementation.

## 1. Equivalent distance representations and a lossy profile

For a connected simple undirected graph `G` rooted at `s`, distinguish:

1. the full distance map `v -> d(s,v)` with vertex identities;
2. the indexed distance partition `(F_0(s),F_1(s),...,F_D(s))` with exact vertex identities;
3. the layer-size profile `(w_0,w_1,...,w_D)`, where `w_i=|F_i(s)|`.

The full map and indexed exact partition contain the same information:
`F_i(s)={v:d(s,v)=i}`, and conversely `d(s,v)` is the unique index of the layer
containing `v`. The size profile remembers only how many vertices have each
distance, not which ones.

Neither the map nor the partition specifies edges within one layer or between
adjacent layers. The profile additionally discards the vertex identities.

## 2. A sound one-way isomorphism test

A rooted graph isomorphism sends the root to the root and preserves adjacency.
It therefore preserves every path length and induces a bijection on every BFS
layer. Hence rooted-isomorphic graphs have identical layer-size profiles.

The contrapositive is useful:

```text
different rooted BFS profiles -> not rooted-isomorphic.
```

The converse is false. Equal profiles are a consistency check, not an
isomorphism certificate.

## 3. A complete-profile counterexample

Compare the triangular prism and the complete bipartite graph `K_(3,3)`.
Both are connected 3-regular vertex-transitive graphs on six vertices with
diameter two. From every root, both have the complete BFS profile

```text
[1, 3, 2].
```

They are not isomorphic: the prism contains triangles, while `K_(3,3)` is
bipartite and contains no odd cycle.

The missing information is already visible inside `F_1`. In `K_(3,3)` no two
root neighbors are adjacent. In the prism, two of the three root neighbors are
joined by an edge belonging to a triangular face. Equal sphere cardinalities
hide different same-layer adjacency.

This also separates profile equality from the richer intersection data of
note 32.

## 4. A bounded profile knows even less

For any chosen radius `r`, sufficiently long cycles of different total lengths
have identical rooted induced balls and identical profiles through depth `r`:

```text
1, 2, 2, ..., 2.
```

Yet the cycles have different orders and diameters. Therefore no fixed-radius
profile can certify global size, exhaustion, diameter, or isomorphism without
additional family assumptions.

This is stronger than saying that the histogram is lossy: even the entire
identity-preserving induced neighborhood through that radius can be shared by
globally different graphs.

## 5. Full distance matrices are different

For a connected simple unweighted graph, the complete all-pairs distance
matrix retains the graph up to simultaneous permutation of rows and columns,
because

```text
{u,v} is an edge  <->  d(u,v)=1.
```

Thus the loss is not caused by distance as a concept. It comes from observing
one root, truncating radius, or aggregating identities into histograms.

One complete BFS gives one row of the distance matrix. Repeating from all roots
but retaining only a multiset of row histograms still does not recover the
matrix: the prism and `K_(3,3)` have the same histogram from every root.

## 6. Color refinement is not BFS

One-dimensional Weisfeiler-Leman, also called color refinement, repeatedly
recolors each vertex from its current color and the multiset of neighbor
colors. At stabilization, vertices in one color class have the same number of
neighbors in every stable color class.

BFS instead assigns minimum root distance. It does not repeatedly split one
distance layer according to its internal and cross-layer adjacency patterns.
Color refinement can therefore reveal heterogeneity hidden inside a BFS layer.

If the root is first given a unique color, refinement can propagate that
distinction outward and may split vertices at the same root distance. In the
prism, its three root neighbors do not remain equivalent: two see one another,
while the cross-edge neighbor has a different neighbor-color multiset. In
`K_(3,3)`, all three root neighbors remain symmetric.

This rooted refinement distinguishes the concrete counterexample even though
the BFS profiles agree. It is still a heuristic invariant rather than a
complete graph-isomorphism procedure on all graph families.

## 7. Uniform color refinement has its own blind spot

Start color refinement with every vertex the same color. In any `k`-regular
graph, every vertex sees the same multiset of `k` identical neighbor colors, so
all vertices retain one common color forever.

Consequently two `k`-regular graphs with the same number of vertices are not
distinguished by this uniform 1-WL color histogram. In particular, the same
prism/`K_(3,3)` pair defeats both:

- complete rooted BFS layer sizes;
- unrooted 1-WL from a uniform initial coloring.

Combining two incomplete summaries does not automatically make them complete.
Individualization changes the test and must be declared explicitly.

## 8. Cayley symmetry makes profiles poor state fingerprints

In a Cayley graph, translation sends any root to any other root while
preserving the graph; under the standard right-edge convention, left
translation also preserves generator labels. Every root therefore has the
same complete layer-size profile and the same rooted labeled geometry up to
translation.

That homogeneity is useful for growth studies but fatal to state
identification:

```text
same BFS profile in a Cayley graph is expected, not evidence of equal states.
```

Similar symmetry can occur in puzzle or Schreier graphs when the fixed graph
has the required vertex-transitive automorphisms; a transitive group action
alone does not establish that property for a fixed generator alphabet. A
profile may classify a local orbit type in a less symmetric graph, but it is
not an exact state key unless injectivity is independently proved.

## 9. Validation versus identity

Layer histograms are valuable retained evidence:

- a mismatch against an exact reference exposes an error;
- expected total volume checks `sum_i w_i`;
- parity or known growth formulas can check a family-specific invariant;
- repeated roots can expose or support claims of radial regularity;
- richer per-layer edge/intersection histograms can localize discrepancies.

But matching histograms can hide one missing state and one spurious state in
the same layer. They cannot validate parent edges, replay paths, exact frontier
sets, or successor completeness.

For `visited`, a BFS profile is at most an advisory fingerprint. Treating a
profile collision as state equality can silently merge distinct vertices. The
exact-key/fingerprint boundary from note 28 still applies.

## 10. GPU and multi-GPU interpretation

Layer counts and small radial statistics are attractive because they are easy
to reduce across devices. Their low communication cost does not increase their
semantic strength.

Useful separation is:

```text
global count reduction       -> aggregate consistency evidence;
exact distributed set digest -> stronger probabilistic set evidence;
exact canonical state set    -> deterministic identity evidence.
```

Even a collision-resistant digest needs a declared probabilistic contract; a
few integers of layer sizes are vastly more collision-prone structurally. No
partition, hashing, or GPU policy follows from the profile alone.

## 11. Evidence checklist

1. Rooted or unrooted comparison.
2. Full traversal or completed prefix radius.
3. Directed, labeled, multigraph, or simple undirected contract.
4. Exact distance map/indexed identity-preserving layers, or only a size histogram.
5. One root, all roots, or sampled roots.
6. Color-refinement initialization and any individualized vertices.
7. Necessary invariant test versus claimed complete identification.
8. Exact-state fallback after any fingerprint match.

## Sources

- M. Grohe, [*Colour Refinement: A Simple Partitioning Algorithm with
  Applications From Graph Isomorphism Testing to Machine
  Learning*](https://doi.org/10.4230/LIPIcs.FSTTCS.2014.31), FSTTCS 2014.
  Defines color refinement as iterated degree-sequence partitioning and places
  it in graph-isomorphism testing.
- M. Grohe, K. Kersting, M. Mladenov, and E. Selman, [*Dimension Reduction via
  Colour Refinement*](https://arxiv.org/abs/1307.5697), 2013. Stable color
  classes and their neighbor-count characterization.
- Notes 10, 21, 28, 31, 32, 37, 41, 61, 78, 79, and 93 provide frontier
  profiles, graph certificates, exact identity, bipartiteness, intersection
  data, validation, local labels, landmarks, resolving sets, and Cayley growth
  context.

## Takeaway

A BFS profile is a radial volume signature. Different profiles soundly refute
rooted isomorphism, but equal profiles do not prove it, identify a state, or
recover adjacency. Color refinement can split structure hidden within a layer,
yet has its own regular-graph blind spots. In Cayley graphs, identical rooted
profiles are forced by symmetry, so they should be used for growth diagnostics
and validation -- never as exact state identity.
