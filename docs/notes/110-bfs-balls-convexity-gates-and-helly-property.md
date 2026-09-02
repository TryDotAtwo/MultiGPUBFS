# BFS balls, convexity, gates, and the Helly property

BFS constructs metric balls, but a metric ball need not behave like a convex
Euclidean region. This note separates four notions that are easy to conflate:
balls, convex sets, gated sets, and Helly families. The distinction matters
whenever several distance constraints are combined.

No recognition algorithm, search optimizer, or GPU implementation is added.

## 1. BFS balls and intervals

In a connected undirected unweighted graph, define

```text
B(c,r) = {x : d(c,x) <= r}
I(u,v) = {x : d(u,x) + d(x,v) = d(u,v)}.
```

An exact BFS from `c` through depth `r` constructs `B(c,r)`. The interval
`I(u,v)` contains every vertex lying on at least one shortest `u-v` path.

A set `S` is geodesically convex when `I(u,v)` is contained in `S` for every
`u,v` in `S`. This is the strong, all-shortest-path definition. A weaker notion
only asks for at least one shortest path inside `S`; the two must be named
separately.

## 2. A BFS ball need not be convex

Consider the cycle `C6` with vertices numbered cyclically. The radius-two ball
around vertex zero contains

```text
{0,1,2,4,5}.
```

Vertices two and four are both in the ball. Their unique shortest path is
`2-3-4`, which leaves the ball. Therefore this BFS ball is not even weakly
geodesically convex, and hence not convex under the all-geodesics definition.

The failure is structural: the triangle inequality only bounds each endpoint's
distance from the center. It does not require a shortest route between the
endpoints to remain near that center.

## 3. Gates are stronger than convexity

A subset `S` is gated if every vertex `x` outside `S` has a vertex `g` in `S`
such that, for every `y` in `S`,

```text
d(x,y) = d(x,g) + d(g,y).
```

The vertex `g` is the gate of `x`: every destination in `S` can be reached by a
shortest path routed through the same metric projection. The gate is necessarily
unique, and every gated set is convex. The converse does not hold in arbitrary
graphs.

In a tree, connected subtrees and metric balls are gated: the unique path from
an outside vertex first enters the subtree at one well-defined vertex. This is
one reason tree balls behave more like familiar convex regions than arbitrary
graph balls.

Finite families of pairwise intersecting gated sets have a common vertex. This
Helly behavior of gated sets should not be silently transferred to arbitrary
BFS balls, which need not be gated.

## 4. Helly graphs

A graph is a Helly graph when every finite family of pairwise intersecting
metric balls has a nonempty total intersection. In symbols, for balls `B_i`,

```text
B_i intersect B_j is nonempty for every i,j
```

implies

```text
intersection over all i of B_i is nonempty.
```

For two graph balls, pairwise intersection is equivalent to

```text
d(c_i,c_j) <= r_i + r_j.
```

Necessity is the triangle inequality. For sufficiency, choose an appropriate
integer vertex on a shortest path between the centers. Thus, in a graph already
known to be Helly, all pairwise center-distance inequalities certify joint
feasibility of the entire family.

Without the Helly premise, pairwise feasibility is insufficient. In `C4`, take
the four radius-one balls, one about each vertex. Each ball omits only the
opposite vertex. Every pair intersects, but the intersection of all four is
empty. Hence `C4` is not a Helly graph.

## 5. Radius and diameter from ball intersection

Let `D` be the diameter of a finite Helly graph and put

```text
r = ceil(D/2).
```

The balls `B(v,r)` over all vertices are pairwise intersecting because
`d(u,v) <= D <= 2r`. The Helly property gives a common vertex `c`. Therefore
`d(c,v) <= r` for every `v`, so the graph radius is at most `r`.

Every graph also satisfies `radius >= ceil(diameter/2)` by the triangle
inequality applied to a diametral pair. Consequently,

```text
radius = ceil(diameter/2)
```

for finite Helly graphs.

This proof is a semantic consequence of ball intersection, not a claim that a
particular two-sweep BFS finds a diameter or a center. The counterexample in
REF-021 remains valid for general graphs.

## 6. Multi-source distance constraints

An intersection

```text
intersection_i B(c_i,r_i)
```

is exactly the set of vertices satisfying all upper-bound constraints
`d(c_i,x) <= r_i`. Separate BFS runs can provide the required distance rows,
but their union is not the answer: union represents satisfying at least one
constraint, while intersection represents satisfying all constraints.

In a Helly graph, checking every pair of constraints suffices for existential
feasibility. It does not enumerate the feasible set, select a canonical point,
or minimize another objective. Outside Helly graphs, even exhaustive pairwise
checks cannot replace the total intersection test.

## 7. Helly is not hyperbolicity

Trees are both Helly and zero-hyperbolic, but the notions measure different
things. Helly asks whether pairwise-compatible balls have a common point;
hyperbolicity bounds the thickness of geodesic triangles.

The distinction can grow with scale. The strong grid
`P_(2r+1) strong-product P_(2r+1)` is a Helly graph: strong products of paths
are standard Helly hosts. Its metric is the `L_infinity` grid metric. For the
four axis points `(plus-or-minus r,0)` and `(0,plus-or-minus r)`, the largest
four-point sum is `4r` and the next largest is `2r`, giving hyperbolicity at
least `r` under the half-gap convention. Thus Helly graphs need not have a
uniform hyperbolicity bound.

Conversely, a finite graph having a small hyperbolicity constant does not by
itself establish the exact Helly property. The properties need separate
evidence.

## 8. Cayley and Schreier consequences

Vertex transitivity does not imply the Helly property. The cycle `C4` is a
Cayley graph of the cyclic group of order four with the usual symmetric
generator, yet its four unit balls give the counterexample above.

Changing generators changes the word-metric balls and can therefore change
their intersection structure. Passing to a Schreier action changes the vertex
identity and metric again. Neither Cayley symmetry nor a result about the
ambient group automatically certifies the exact finite action graph as Helly.

For directed positive alphabets, standard undirected balls and symmetric
center-distance inequalities no longer describe reachability. Forward and
reverse balls are different objects, so a directed analogue must be declared
rather than inferred by adding inverse moves.

## 9. GPU and multi-GPU evidence boundary

Ball constraints suggest parallel distance work, but the semantic certificate
comes first. A GPU computation of all pairwise inequalities proves total
feasibility only after the graph class's Helly property has independently been
established.

Distribution introduces a second distinction:

- each center's BFS row may be partitioned or replicated;
- pairwise constraint checks may be locally parallel;
- the actual common-intersection witness still requires globally consistent
  vertex identity and a reduction over all constraints.

Throughput for distance rows, pair tests, and witness intersection are three
different measurements. None alone measures ordinary frontier/visited BFS
throughput, and no performance conclusion follows merely from the word
"Helly".

## Sources

- H.-J. Bandelt and V. Chepoi,
  [*Metric Graph Theory and Geometry: A Survey*](https://doi.org/10.1090/conm/453/08795),
  Contemporary Mathematics 453, 2008. Definitions and structural relations
  among intervals, convexity, gates, and Helly graphs.
- H.-J. Bandelt, V. Chepoi, and A. Karzanov,
  [*A Helly theorem in weakly modular space*](https://doi.org/10.1016/0012-365X(95)00217-K),
  Discrete Mathematics 160, 1996. Helly behavior for graph-metric convexity
  classes.
- G. Ducoffe,
  [*Distance problems within Helly graphs and k-Helly graphs*](https://arxiv.org/abs/2011.00001),
  2020. Modern ball-hypergraph definition and distance-problem context.

