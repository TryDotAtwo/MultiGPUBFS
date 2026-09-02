# BFS sweeps, eccentricity bounds, and diameter certificates

A complete BFS from `s` computes one exact eccentricity `ecc(s)`. Choosing a
farthest vertex and sweeping again often finds a large distance, but a large
lower bound is not yet a diameter certificate. Exactness appears only when an
independent upper bound meets it.

This note deepens note 21's double-sweep boundary. It adds no implementation,
optimizer, benchmark, or GPU code.

## 1. What one complete sweep certifies

For a finite connected undirected graph,

```text
ecc(s) = max_x d(s,x)
ecc(s) <= D <= 2*ecc(s),
```

where `D=diam(G)`. The BFS tree height equals `ecc(s)`, but its own tree
diameter is an upper bound on graph diameter because tree paths can be longer
than graph shortest paths.

For every vertex `x`, one pivot sweep also gives

```text
ecc(x) <= d(s,x)+ecc(s),
ecc(x) >= max(d(s,x), ecc(s)-d(s,x)).
```

The upper bound follows by routing every `x-y` comparison through `s`. For the
second lower term, choose a vertex farthest from `s` and apply the reverse
triangle inequality. These are certificates from complete exact distances,
not predictions about graph shape.

## 2. Why a farthest sweep improves monotonically

Let `v` be any vertex farthest from `u`. Then

```text
d(u,v)=ecc(u)
ecc(v) >= d(v,u)=ecc(u).
```

Therefore a chain that repeatedly selects a farthest vertex has nondecreasing
eccentricities. Every observed eccentricity is a valid diameter lower bound,
so keeping their maximum is safe.

This monotonicity does not imply strict progress. Equal consecutive
eccentricities can occur below the true diameter. A sweep chain may enter a
plateau whose vertices all miss the actual diametral pair.

## 3. Double sweep is lower-bound production

Starting at `s`, double sweep chooses a farthest `u`, then computes `ecc(u)` and
a farthest `v`. Its universal output contract is

```text
d(u,v)=ecc(u) <= D.
```

On a tree, a farthest vertex from any start can serve as a diameter endpoint,
so the second sweep is exact. In a general graph, alternate shortest routes
break the unique-path argument.

Note 21's seven-vertex counterexample has a unique first farthest vertex and
returns `3` although `D=4`. The failure is therefore not merely an unfortunate
tie among the first farthest layer.

## 4. Tie policy is part of a sweep heuristic

If a sweep has several farthest vertices, selecting one by minimum ID, maximum
degree, random choice, generator order, or parent-tree order can lead to
different later roots and bounds. All choices preserve the lower-bound
contract; none is universally entitled to produce the best next sweep.

Recording only the numeric result hides this dependency. A reproducible sweep
record needs the start, farthest-set completeness, selection rule, graph epoch,
and resulting eccentricity.

## 5. Stabilization is not exactness

The following stopping rules are not generally sound diameter certificates:

- two consecutive sweeps return the same eccentricity;
- a farthest vertex points back to the previous root;
- several restarts return the same value;
- the current lower bound equals a known experiment from another graph epoch;
- the chosen root looks central or peripheral by degree.

They may be useful heuristic stopping policies only when the output is labeled
as a lower bound. Exact stopping requires a theorem for the graph family or a
separate upper bound no greater than the lower bound.

## 6. Multiple pivot bounds

Suppose complete BFS sweeps have been run from pivots `P`. Their eccentricities
give a global lower bound

```text
L = max_(p in P) ecc(p) <= D.
```

For every vertex `x`, triangle inequality gives

```text
ecc(x) <= U(x) = min_(p in P) (d(p,x)+ecc(p)).
```

Hence

```text
D <= U = max_x U(x).
```

Adding a pivot cannot decrease `L` and cannot increase `U`. When `L=U`, the
diameter is certified. If only a subset of vertices is represented or a pivot
BFS is depth-bounded, the missing coordinates cannot be silently inserted into
this proof.

## 7. A fringe upper-bound certificate

Fix a root `c` and its complete BFS layers `F_i(c)`. Suppose complete BFS
sweeps have been run from every vertex in all outer layers with depth at least
`i`. Let

```text
M = maximum eccentricity seen among those outer vertices.
```

Any pair with at least one outer endpoint has distance at most `M`. Both
endpoints of every remaining pair lie within distance `i-1` of `c`, so

```text
d(x,y) <= d(x,c)+d(c,y) <= 2(i-1).
```

Therefore

```text
D <= max(M, 2(i-1)).
```

At the same time `M<=D`. If `M>2(i-1)`, then `D=M`; more generally, combining
this upper bound with any lower bound yields exactness when they meet. This is
the semantic core behind fringe upper-bound methods such as iFUB.

The proof depends on processing every vertex in the declared outer layers.
Sampling that fringe does not preserve the upper bound.

## 8. Root choice changes work, not validity

A root near a graph center can make the unprocessed inner ball small in radius,
and a small outer fringe can reduce the number of additional sweeps. Four-sweep
procedures use repeated farthest searches and midpoints of returned BFS-tree
paths to seek such a root and a strong initial lower bound.

The midpoint is a midpoint of one selected tree path, not automatically a graph
center. The cited iFUB study gives infinite bad families for four-sweep quality
and `Theta(nm)` worst-case iFUB work. Root selection therefore affects observed
cost, while the outer-layer upper-bound proof supplies correctness.

## 9. BFS-tree diameter is a different upper bound

For a spanning BFS tree `T` of `G`,

```text
d_G(x,y) <= d_T(x,y),
D_G <= D_T.
```

Computing `D_T` gives a valid upper bound, but it need not meet the sweep lower
bound. Parent tie-breaking can change `T` and `D_T` without changing any graph
distance. The cited iFUB paper also notes families where no BFS tree has tree
diameter equal to the graph diameter.

Thus tree-diameter tightness and graph-diameter exactness are separate claims.

## 10. Directed graphs need another theorem

In a directed graph, out-distance is asymmetric and may be infinite. If `v` is
farthest from `u` by finite out-distance, `ecc_out(v)` need not be at least
`ecc_out(u)` because the return distance `d(v,u)` may be smaller, larger, or
infinite. The undirected monotone-sweep proof uses symmetry explicitly.

Directed diameter also depends on strong connectivity and infinity conventions.
Undirected double-sweep or fringe certificates cannot be transferred by merely
replacing BFS with forward directed BFS.

## 11. Cayley and Schreier boundary

In a finite connected undirected Cayley graph, left translations make every
vertex eccentricity equal to `D`. One complete identity BFS already certifies
diameter, so double sweep adds no metric information.

This explains rather than contradicts sweep success: the graph's symmetry
collapses the global maximum to one rooted maximum. An arbitrary Schreier or
quotient state graph needs its own automorphism argument; transitivity of an
underlying group action does not by itself prove that the fixed-generator
unlabeled graph is vertex-transitive under the required conventions.

For a finite strongly connected directed Cayley graph, the analogous statement
holds for directed out-eccentricity because left translation preserves the
directed generator graph. Strong connectivity remains essential.

## 12. GPU and multi-GPU interpretation

Repeated sweeps are repeated complete traversal workloads. Distinguish:

- parallelism inside one BFS across devices;
- independent BFS roots run concurrently;
- the sequential dependency that chooses later roots or lowers `i`;
- replicated versus shared graph storage;
- per-root frontier, visited, and completion evidence;
- reduction of eccentricities and global lower/upper bounds;
- heuristic root-selection time and exact certificate time.

All BFS roots in one fringe layer may be logically independent once the layer
is fixed, but their completed eccentricities must all be included before the
fringe upper bound is asserted. Faster partial results do not certify the
unprocessed roots.

For Cayley graphs, distributing many redundant roots would measure a different
workload from distributing the one sufficient identity BFS.

## Sources

- P. Crescenzi, R. Grossi, M. Habib, L. Lanzi, and A. Marino,
  [*On Computing the Diameter of Real-World Undirected Graphs*](https://doi.org/10.1016/j.tcs.2012.09.018),
  Theoretical Computer Science 514, 2013. Defines four-sweep and iFUB, proves
  the fringe upper-bound procedure, and gives worst-case negative families.
- D. G. Corneil, F. F. Dragan, M. Habib, and C. Paul,
  [*Diameter Determination on Restricted Graph Families*](https://www.cs.kent.edu/~dragan/DiamRestrGr.pdf),
  Discrete Applied Mathematics 113, 2001. Shows that sweep guarantees can
  strengthen under explicit chordal and AT-free graph-family hypotheses.
- Notes 21, 30, 42, 57, 72, 78, 109, and 122 supply this repository's diameter,
  replay, bounded-evidence, output, dead-end, landmark, hyperbolicity, and
  pseudo-peripheral boundaries.

## Takeaway

Farthest-point sweeps monotonically accumulate valid diameter lower bounds in
connected undirected graphs, but plateaus and repeated values do not certify
exactness. Exact diameter follows when a separately proved upper bound meets
the lower bound. Pivot inequalities and fully processed outer BFS layers supply
such upper bounds; root and tie heuristics affect how soon they tighten, not
whether the final certificate is valid.
