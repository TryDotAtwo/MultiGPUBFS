# BFS landmarks, triangle bounds, and Cayley homogeneity

A complete BFS does not merely answer one source-to-target query. It assigns
an exact metric coordinate `d(l,x)` relative to its root `l` for every reached
vertex `x`. Several such coordinate fields can certify bounds on distances
between pairs that were never used as BFS roots.

This note studies the information in those fields. It is not a proposal for an
optimized shortest-path implementation.

## 1. One landmark in an undirected graph

Let `D_l(x)=d(l,x)` be the result of a complete BFS from landmark `l`. For any
`u,v` in its connected component, the triangle inequality gives

```text
|D_l(u)-D_l(v)| <= d(u,v) <= D_l(u)+D_l(v).
```

The lower bound follows by writing the triangle inequality in both directions.
The upper bound is witnessed by the concrete walk `u -> l -> v`; it may repeat
vertices and need not be shortest.

For a landmark set `L`, the bounds combine as

```text
max_l |D_l(u)-D_l(v)| <= d(u,v)
                            <= min_l (D_l(u)+D_l(v)).
```

Adding a landmark can only raise this certified lower bound and lower this
certified upper bound. It need not make either exact.

### Equal coordinates do not imply nearby vertices

On the four-cycle `0--1--2--3--0`, choose `l=0`, `u=1`, and `v=3`. Both
landmark coordinates equal one, so the lower bound is zero, while
`d(1,3)=2`. One BFS layer is a sphere, not a set of mutually close vertices.

Only distance from the landmark is exact. A bound for a new pair and the exact
distance of that pair are different outputs.

## 2. Directed graphs need two coordinate directions

For a directed graph, distance is asymmetric and may be infinite. Where all
terms used below are finite, triangle inequalities give

```text
d(l,t) - d(l,v) <= d(v,t)
d(v,l) - d(t,l) <= d(v,t).
```

After clamping negative values at zero, a landmark lower bound is

```text
max(0, d(l,t)-d(l,v), d(v,l)-d(t,l)).
```

The first coordinate field comes from a forward search **from** `l`. The
second comes from a search from `l` in the transposed graph, equivalently
distances **to** `l` in the original graph. One forward BFS does not supply
both fields.

A landmark detour gives `d(v,t) <= d(v,l)+d(l,t)` only when both directed legs
exist. Subtracting infinities or silently replacing unreachable values by a
finite depth is not a valid certificate.

These are the landmark/triangle inequalities underlying ALT. Their metric
meaning is independent of whether a later query uses A*, BFS, or no search.

## 3. Bounded BFS produces bounded evidence

If BFS from `l` stops at depth `k`, it proves exact `D_l(x)` only for vertices
discovered in completed layers through `k`. An absent vertex means only
`d(l,x)>k` or unreachable under the declared graph, not an exact coordinate.

- A triangle expression using two exact stored coordinates remains valid.
- An expression needing an unknown coordinate cannot insert `k+1`, infinity,
  or a default table value.
- The three-valued lookup discipline from note 42 still applies.

The landmark table inherits the graph version, source, direction,
depth-completion, and identity contracts of the BFS that created it.

## 4. Cayley graphs collapse pair queries to the identity

Let a directed Cayley graph use right-action edges `g -> g s` for `s in S`.
Left multiplication by any group element is a graph automorphism, even when
`S` is not inverse closed. Therefore

```text
d(g,h) = d(e, g^-1 h).
```

For an inverse-closed word metric this is the usual left-invariance formula.
For a directed alphabet it remains true with directed distance and the same
alphabet.

This is stronger than an ordinary landmark bound. A complete identity-rooted
distance table plus exact group composition and inversion is an exact all-pairs
distance oracle: form `r=g^-1 h`, then look up `d(e,r)`.

It does not make a bounded table complete. If it contains only `B_k(e)`, it
answers a pair exactly only when the relative element is present. Nor is the
identity geometrically superior to another root; it is convenient because
group operations express every pair in its coordinate frame.

## 5. Why this is unsafe in a Schreier graph

For a non-free action, vertices are orbit states/cosets rather than uniquely
identified group elements. Two group elements may represent the same state,
and representatives need not define one relative target by `g^-1 h`. As note
16 derives, the valid target may be a stabilizer coset such as `a^-1 H b`.

Vertex transitivity can make rooted distance distributions alike without
licensing the concrete Cayley formula. Before reusing one table for all pairs,
establish:

- whether a vertex is a group element or a coset/orbit state;
- action side and multiplication order;
- generator alphabet and direction;
- exact inversion, composition, and state identity;
- concrete-target versus orbit-target semantics.

## 6. Semantic cost accounting

With `q` landmarks and `n` reachable vertices, an uncompressed table contains
`q*n` distance values. In an explicit unweighted graph, ordinary construction
traverses reachable adjacency once per landmark; in an implicit graph it
generates successors once per expanded state per landmark.

These are accounting identities, not a recommendation for `q`, a layout, or a
GPU strategy. Landmark usefulness depends on how strongly the coordinate
vectors separate later query pairs. The four-cycle shows an exact field can
still give a vacuous lower bound.

## 7. Evidence checklist

1. Directed or undirected graph and exact vertex identity.
2. Whether each field stores distances from or to its landmark.
3. Complete traversal, complete depth, or partial-frontier status.
4. Treatment of unreachable and unknown entries.
5. Exact bound formula and finite-term preconditions.
6. For Cayley reuse, the action convention behind `g^-1 h`.
7. For quotient actions, stabilizer/orbit semantics instead.
8. Whether the output is a bound, exact distance, or replayable path.

## Sources

- A. V. Goldberg and C. Harrelson,
  [*Computing the Shortest Path: A* Search Meets Graph Theory*](https://www.microsoft.com/en-us/research/publication/computing-the-shortest-path-a-search-meets-graph-theory-2/),
  SODA 2005. Introduces landmark preprocessing and directed triangle lower
  bounds in the ALT setting.
- [*Word metric*](https://encyclopediaofmath.org/wiki/Word_metric),
  Encyclopedia of Mathematics. States the Cayley-graph interpretation and
  left invariance of the word metric.
- Notes 16, 21, 40, 42, and 50 supply the existing Schreier/stabilizer,
  eccentricity, reverse-BFS, bounded-lookup, and bound-certified-search
  contracts used here.

## Takeaway

A BFS distance field is an exact coordinate relative to its root. In a general
undirected graph, coordinate differences and landmark detours bound a new pair
but need not determine it. Directed graphs require distinct forward and reverse
fields. A genuine Cayley graph is exceptional: group homogeneity turns one
complete identity table into exact all-pairs distances through
`d(g,h)=d(e,g^-1 h)`. That shortcut must not be transferred silently to
Schreier or quotient state spaces.
