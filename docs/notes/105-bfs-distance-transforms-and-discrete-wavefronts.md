# BFS distance transforms and discrete wavefronts

Multi-source BFS computes an arrival-time field for a front that crosses every
allowed edge in one unit of time. On image grids this is a distance transform;
on Cayley graphs it is a word metric. The result is exact for the declared
graph metric, which need not be Euclidean or physically isotropic.

This note develops the metric boundary. It adds no image or PDE solver.

## 1. Distance to a source set

Let `A` be a nonempty set of sources in an unweighted graph. Multi-source BFS
initializes every `a in A` at distance zero and computes

```text
D(v) = min_(a in A) d(a,v).
```

For every reachable non-source vertex,

```text
D(v) = 1 + min_(u in N^-(v)) D(u)
```

in a directed graph, where `N^-(v)` contains predecessors with arcs into `v`.
For an undirected graph this is the ordinary neighbor set.

The BFS frontiers are exactly the level sets

```text
F_r = {v : D(v)=r}.
```

Thus BFS is both a shortest-path algorithm and a solver of a discrete Bellman
arrival-time equation with unit transition cost.

## 2. Iterated dilation is the same set recurrence

For a set `X`, let one graph dilation include `X` and all its neighbors:

```text
dilate(X) = X union N(X).
```

Then

```text
B_0=A,
B_(r+1)=dilate(B_r),
F_r=B_r minus B_(r-1).
```

Repeated binary dilation and multi-source BFS balls are therefore the same set
process when they use the same adjacency/structuring element. The numerical
distance transform records the first dilation round at which each vertex
enters.

Changing the structuring element changes the graph and hence the metric.

## 3. Four-neighbor grids compute Manhattan distance

On an obstacle-free integer grid with moves

```text
(+1,0), (-1,0), (0,+1), (0,-1),
```

every path from `(0,0)` to `(x,y)` needs at least `|x|` horizontal and `|y|`
vertical moves, and that bound is attained. Therefore

```text
D_4(x,y)=|x|+|y|.
```

The radius-`r` wavefront is a diamond in Euclidean coordinates. BFS is exact;
it is the graph model, not the traversal, that selected the `L1` geometry.

## 4. Unit-cost eight-neighbor grids compute Chebyshev distance

Add the four diagonal moves and charge every move one. A diagonal step can
reduce both coordinate differences simultaneously until the smaller one is
zero, after which axial steps finish the larger difference. Hence

```text
D_8(x,y)=max(|x|,|y|).
```

The wavefront is an axis-aligned square. Along `(k,k)`, BFS reports `k`, while
Euclidean distance is `k sqrt(2)`. Unit-cost diagonal adjacency therefore
underprices physical diagonal motion if edges are intended to represent equal
grid spacing.

## 5. Weighted diagonals still do not give exact Euclidean distance

Charging axial moves `1` and diagonal moves `sqrt(2)` requires a nonnegative
weighted shortest-path algorithm such as Dijkstra, not ordinary BFS. The
resulting eight-direction path metric is closer to Euclidean distance but is
still constrained to the finite stencil.

For example, displacement `(2,1)` costs `sqrt(2)+1`, whereas its Euclidean
length is `sqrt(5)`. More directions or chamfer weights change the
approximation, but no finite move stencil should be called exact Euclidean
distance without a proof.

Grid refinement alone preserves the normalized `L1` metric for a four-neighbor
stencil. Smaller pixels do not turn the wrong local norm into the Euclidean
norm automatically.

## 6. Obstacles change straight-line distance into a geodesic

Removing blocked pixels or states forms an induced/subgraph of legal motion.
BFS then returns the shortest legal graph path around obstacles. Two points can
be geometrically close and graph-far, or disconnected, when a barrier lies
between them.

This distinction is often desirable: an occupancy-grid robot needs feasible
travel distance, not distance through walls. But the output must be named
accordingly:

```text
grid graph geodesic != unobstructed Euclidean distance.
```

Unknown cells, corner cutting, diagonal contact, and boundary conditions are
part of the graph contract.

## 7. Voronoi labels and ties

If each source carries an identity, multi-source BFS can assign every reached
vertex to a nearest source and retain ties. The scalar distance transform alone
does not identify which source won, and one arbitrarily selected owner does not
prove uniqueness.

Equidistant sets depend on the graph metric. Four-neighbor, eight-neighbor,
weighted, obstacle-aware, directed, and Cayley metrics can produce different
Voronoi boundaries from the same embedded coordinates.

## 8. Relation to the Eikonal equation and fast marching

The continuous Eikonal arrival-time equation is commonly written

```text
|grad T(x)| F(x) = 1,
```

for positive local propagation speed `F`. Fast marching methods exploit
monotone causal acceptance to solve a numerical discretization of this
continuous problem.

BFS is the special discrete graph case where every edge takes one unit and
arrival times are integer hop counts. Dijkstra is the corresponding graph
generalization for nonnegative edge travel times. Fast marching uses a
geometric/PDE discretization whose local update can combine several accepted
neighbors; it is not merely BFS with a different queue.

The shared metaphor is a monotonically advancing accepted front. The equations,
local update, metric, and error analysis remain different.

## 9. Direction and time dependence

In a directed graph, arrival from `A` uses incoming-predecessor Bellman
recurrence and need not be symmetric. An edge allowed eastward but not westward
defines a directed travel-time relation, not a metric in the symmetric sense.

If edge availability or speed changes with time, a static distance transform
is no longer sufficient. Temporal state, waiting, FIFO assumptions, or
time-dependent shortest-path semantics must be declared as in note 22.

## 10. Cayley interpretation

For a symmetric unit generator set, identity-rooted BFS computes the word
metric and its spheres are discrete wavefronts. For a positive nonsymmetric
alphabet it computes directed minimum word length. Relations make different
wave branches meet; they are the discrete analogue of fronts colliding after
travel through multiple routes.

Assigning different generator costs changes the metric and generally leaves
ordinary BFS's guarantee boundary. Geometric angles or physical move lengths
do not determine those costs unless the puzzle model explicitly says so.

Multi-source Cayley BFS computes distance to a set of group/orbit states. It
does not automatically compute distance to a group subgroup, coset, or symmetry
class unless those exact states or an equivalent proven construction seed the
front.

## 11. GPU and multi-GPU interpretation

Regular grids have fixed local stencils and compact coordinates, while general
implicit Cayley states may require wide transformations and exact hashing. A
fast grid wavefront is therefore useful evidence for the traversal mechanism
but not a direct Cayley throughput prediction.

Dense dilation and sparse frontier BFS can perform different physical work for
the same balls:

- dense iteration touches a broad image/domain each round;
- sparse BFS touches active frontier states and their transitions;
- near saturation, dense regular access may differ from sparse random visited;
- multi-source seeding changes early frontier geometry;
- device partition boundaries may not align with geometric wavefronts.

Measurements must retain the metric, stencil/generators, obstacles, source set,
state representation, and exact output. No kernel policy follows here.

## 12. Evidence checklist

1. Source set and tie/label output.
2. Directed or undirected adjacency.
3. Stencil/generator set and every transition cost.
4. Graph, `L1`, `L-infinity`, chamfer, or Euclidean distance claim.
5. Obstacles, corner rules, and boundary conditions.
6. BFS, 0-1 BFS, Dijkstra, or PDE discretization guarantee.
7. Static versus temporal travel model.
8. Dense-domain work versus active-frontier work.

## Sources

- A. Rosenfeld and J. L. Pfaltz, [*Distance Functions on Digital
  Pictures*](https://doi.org/10.1016/0031-3203(68)90013-7), Pattern Recognition
  1(1) (1968), 33-61. Digital distance functions and repeated parallel local
  operations on picture elements.
- J. A. Sethian, [*A Fast Marching Level Set Method for Monotonically Advancing
  Fronts*](https://doi.org/10.1073/pnas.93.4.1591), PNAS 93(4) (1996),
  1591-1595. Monotone front propagation and numerical Eikonal solution.
- Notes 04, 05, 10, 12, 13, 22, 25, 33, 40, 75, 93, 96, and 102 provide
  frontier, algorithm-boundary, growth, weighted, Voronoi, temporal,
  fixed-point, walk, reverse, orientation, Cayley-metric, flooding, and local
  message-passing context.

## Takeaway

BFS is an exact unit-time wave solver on the graph it is given. On grids, the
neighbor stencil chooses `L1`, `L-infinity`, obstacle geodesic, or another
discrete metric; it does not silently yield Euclidean distance. Fast marching
shares causal-front intuition but solves a different discretized Eikonal
problem. The metric contract comes before the frontier implementation.
