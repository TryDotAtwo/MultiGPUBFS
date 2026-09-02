# BFS-tree root exactness and pairwise stretch

A BFS parent tree is a shortest-path tree **from its root**. It is not generally
an all-pairs distance-preserving tree. This distinction matters whenever parent
pointers are reused as a compact model of the explored graph.

This note concerns semantics and counterexamples, not construction of a
low-stretch or production tree.

## 1. What a BFS tree preserves exactly

Let `G` be a finite connected undirected unweighted graph, let `s` be the BFS
root, and let every non-root vertex choose one predecessor from the preceding
layer. The resulting spanning tree `T` satisfies

```text
d_T(s,v) = d_G(s,v)
```

for every vertex `v`. Each parent edge decreases the BFS label by one, so the
tree path has exactly `d_G(s,v)` edges.

Because `T` is a subgraph of `G`, it can never shorten any pair:

```text
d_G(u,v) <= d_T(u,v).
```

Only pairs involving `s` are guaranteed equality by the BFS-tree contract.

## 2. A universal root-detour bound

The tree contains both exact root paths, hence

```text
d_T(u,v)
  <= d_T(u,s)+d_T(s,v)
   = d_G(u,s)+d_G(s,v)
  <= 2 ecc_G(s).
```

This is a finite upper bound, not a constant stretch guarantee. If `u,v` are
adjacent in `G`, their original distance is one while their tree distance can
grow linearly with the root eccentricity.

More precisely, if `a` is the lowest common ancestor of `u,v` in `T`, then

```text
d_T(u,v) = d_G(s,u)+d_G(s,v)-2 d_G(s,a).
```

The BFS labels of the endpoints do not determine the LCA depth. Parent choices
therefore preserve every root distance while changing arbitrary-pair distance.

## 3. Odd-cycle witness: one omitted edge stretches to `2r`

Take cycle `C_(2r+1)` and root BFS at any vertex `s`. Its two farthest vertices
`u,v` lie at depth `r` and are adjacent to each other. Their edge is lateral
between equal BFS layers and cannot be a parent edge.

The BFS tree contains the two length-`r` arms from `s` and omits `{u,v}`. Thus

```text
d_G(u,v)=1,
d_T(u,v)=2r.
```

So a BFS tree can stretch one original edge by the full `2 ecc(s)` bound. The
phenomenon does not require a poor queue order; every BFS tree of the odd cycle
rooted at `s` has this farthest lateral edge.

This also links two earlier observations:

- the lateral edge is exactly the same-level collision used in odd-cycle
  certificates;
- deleting it is harmless for root distances but destructive for its own pair
  distance.

## 4. Parent ties change tree geometry

Use vertices `s,a,b,u,v` with edges

```text
s-a, s-b,
a-u, b-u,
a-v, b-v,
u-v.
```

The BFS depths from `s` are zero, one for `a,b`, and two for `u,v`. Both `a`
and `b` are valid parents of each depth-two vertex.

- If `u` and `v` choose the same parent, `d_T(u,v)=2`.
- If they choose different parents, `d_T(u,v)=4`.
- In the original graph, `d_G(u,v)=1`.

All choices produce equally valid shortest-path trees from `s`; deterministic
parent rules make the tree reproducible but do not make its pair geometry
canonical in a graph-theoretic sense.

## 5. Tree, shortest-path DAG, and original graph

The BFS predecessor DAG retains every edge from layer `i-1` to layer `i` that
can precede a shortest root path. It contains more root-path information than
one parent tree. It still need not preserve arbitrary-pair shortest paths:

- same-layer edges are absent from the root predecessor relation;
- an edge between adjacent layers may point the wrong way for a pair query;
- shortest pair paths may move toward and then away from the root in ways not
  represented by one directed predecessor orientation.

Keeping the original explored adjacency, a spanner with a proved guarantee, or
another query-specific structure is a different output contract from retaining
parents for root-path replay.

## 6. Cayley interpretation: normal forms are not the word metric tree

In a Cayley BFS rooted at the identity, choosing one parent assigns each group
element `g` one geodesic generator word `p(g)`. The parent tree is therefore a
prefix tree of selected normal forms after shared prefixes are identified.

For two elements `g,h`, tree distance removes only their common selected tree
prefix:

```text
d_T(g,h) = |p(g)| + |p(h)| - 2 |common_tree_prefix|.
```

The Cayley distance instead is

```text
d_G(g,h) = |g^-1 h|_S,
```

minimized over all generator relations. Algebraic cancellation or a short
relation between `g` and `h` need not appear as a common prefix of their chosen
identity normal forms.

The odd cycle is already a Cayley example: `C_(2r+1)` is a Cayley graph of the
cyclic group with generators `+/-1`. The two depth-`r` elements are generator
neighbors, yet their selected root branches meet only at the identity and have
tree distance `2r`.

Thus a parent table proves one shortest word from the identity to every stored
element. It does not turn the chosen normal forms into an all-pairs word-metric
embedding.

## 7. Parent storage and distributed meaning

In a sharded BFS, a parent record may be enough to reconstruct a root path by
following owners across ranks. Even perfect replay validates only:

- the parent edge exists;
- labels decrease by one;
- the reconstructed root path has the recorded length.

It does not validate distances between arbitrary stored states through the
parent tree. Different race winners or deterministic tie policies can change
LCAs and pair stretch without changing distance labels, visited membership,
frontier counts, or root-path correctness.

Consequently, a checkpoint that promises root paths may permit a different
parent forest after replay, while a checkpoint promising canonical parents or
stable tree geometry needs a stronger ordering contract.

## 8. Relation to graph spanners

A spanning subgraph `H` is a multiplicative `t`-spanner when

```text
d_H(u,v) <= t d_G(u,v)
```

for all pairs. A BFS tree is a spanning subgraph, but its defining guarantee is
root exactness, not bounded all-pairs `t`. The odd-cycle family makes its
maximum stretch grow with graph size.

Several BFS trees, selected edges, or specialized tree constructions can be
used in spanner research. That does not make an arbitrary single BFS parent
tree a spanner with a small universal factor.

## 9. Evidence checklist

Before using a BFS tree beyond root replay, record:

1. root and exact BFS labels;
2. parent admissibility and tie rule;
3. desired query class: root pairs, arbitrary pairs, or original edges;
4. whether lateral and non-parent predecessor edges are retained;
5. claimed additive or multiplicative stretch and its proof scope;
6. Cayley action, generator set, and normal-form convention;
7. whether distributed replay requires arbitrary or canonical parents;
8. whether the original adjacency remains available for validation.

## Sources

- D. Peleg and A. A. Schaffer,
  [*Graph Spanners*](https://doi.org/10.1002/jgt.3190130114), Journal of Graph
  Theory 13 (1989), 99-116. Gives the standard all-pairs multiplicative-spanner
  definition and foundational results.
- Notes 11, 19, 21, 27, 30, 31, and 63 supply the existing parent-DAG,
  deterministic-parent, eccentricity, relation, replay, odd-cycle, and
  same-layer-edge contracts used here.

## Takeaway

A BFS tree is exact radially and potentially very inaccurate tangentially. It
preserves every distance from its root, yet an original edge can become a tree
path of length `2 ecc(s)`. Parent ties alter this pair geometry without altering
any BFS distance. In a Cayley graph, selected geodesic normal forms preserve
identity word lengths but not the all-pairs word metric.
