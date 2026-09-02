# BFS, LexBFS, chordal graphs, and elimination orders

Ordinary BFS computes shortest-path layers. LexBFS computes a structural vertex
ordering using the full history of already selected neighbors. Chordal-graph
recognition is a central example where the second contract matters and the first
is insufficient.

This note extends note 19 from ordering semantics to chordal geometry. It adds
no recognizer, triangulation routine, or optimized implementation.

## 1. Chordal graphs

A finite simple undirected graph is chordal when it has no induced cycle of
length at least four. A long cycle may exist, but if it does, the subgraph on its
vertices must contain a chord joining two nonconsecutive cycle vertices.

Chordality is therefore not acyclicity. Complete graphs are chordal and contain
many cycles. Chordality is also not unique-geodesic structure: in the diamond
`K4` minus one edge, the two nonadjacent vertices have two distinct shortest
paths of length two through the two shared neighbors.

## 2. Simplicial vertices and PEOs

A vertex is simplicial when its open neighborhood is a clique. An ordering

```text
v_1, v_2, ..., v_n
```

is a perfect elimination ordering (PEO) when each `v_i` is simplicial in the
subgraph induced by `v_i,...,v_n`. Equivalently, the later neighbors of every
vertex form a clique.

The classical characterization is exact:

```text
G is chordal if and only if G has a PEO.
```

Removing a simplicial vertex creates no need to connect its remaining neighbors:
they are already mutually adjacent. This is an elimination/fill property, not a
shortest-distance invariant.

For a chordal graph, maximal cliques can form the bags of a clique tree, and
treewidth equals maximum-clique size minus one. This links chordality to note
113, but does not identify a clique-tree bag with a BFS frontier.

## 3. What LexBFS remembers

LexBFS repeatedly selects an unselected vertex with lexicographically largest
label, where the label records which previously selected vertices were its
neighbors. Exact label conventions vary, but the essential information is the
ordered selection history, not merely current distance or adjacency-list order.

With the standard selection-order convention, reversing a LexBFS ordering of a
chordal graph yields a PEO. Conversely, running LexBFS and checking the resulting
candidate PEO gives a chordality recognition method.

LexBFS always returns an ordering, including on nonchordal graphs. The ordering
alone does not assert that the graph is chordal; the PEO condition must be
verified.

## 4. Ordinary BFS cannot replace the LexBFS tie rule

Use the chordal diamond with vertices `r,a,b,v` and every edge except `a-b`.
Starting ordinary BFS at `r`, all other vertices lie in layer one. The FIFO order

```text
r, a, b, v
```

is a valid BFS ordering under one adjacency tie order.

Reverse it to obtain

```text
v, b, a, r.
```

When `v` is eliminated first, its later neighbors are `b,a,r`. Vertices `a` and
`b` are not adjacent, so the later-neighbor set is not a clique. The reversed
ordinary BFS order is not a PEO even though the graph is chordal.

After LexBFS selects `a`, the label of adjacent `v` changes while the label of
nonadjacent `b` does not. The history-sensitive priority prevents precisely this
ordinary-BFS tie behavior. Sorting adjacency lists once is not equivalent to
maintaining those evolving labels.

## 5. Distances and elimination are different outputs

LexBFS is consistent with a possible BFS layer order, as note 19 records, but
its theorem concerns the ordering inside and across admissible ties. Ordinary
BFS distance correctness permits any order within a layer.

Consequently:

- a correct distance array need not carry a PEO;
- a PEO does not by itself give distances from a source;
- a BFS parent tree discards the nontree adjacencies needed to check whether
  later neighbors form a clique;
- identical BFS layer sizes do not determine chordality.

For example, chordal and nonchordal graphs can share a root distance profile
while differing only in edges within or across already occupied layers.

## 6. Chordal completion changes the BFS problem

A triangulation or chordal completion adds edges until the graph becomes chordal.
Those fill edges are useful for elimination-based algorithms, but they change the
graph metric. A newly added chord can shorten distances, alter BFS layers, add
paths, and change frontier/duplicate behavior.

Therefore BFS on a chordal completion is not automatically an exact BFS on the
original graph. Fill edges may be used as auxiliary structure only when the
output reconstruction or bounds are separately proved for the original edge
relation.

Minimum fill, minimum treewidth triangulation, and shortest-path search are
different optimization problems.

## 7. Chordality does not bound BFS work

The complete graph `K_n` is chordal, has diameter one, and has first frontier
`n-1`. A star is chordal, has treewidth one, and also has first frontier `n-1`
from its center. Chordality therefore does not bound frontier width, degree,
visited size, or candidate count.

Chords may reduce some distances while increasing edge density and successor
work. The absence of induced long cycles is structural information, not a
monotone prediction of BFS runtime.

## 8. A finite Cayley consequence

Every finite chordal graph has a simplicial vertex. In a vertex-transitive graph,
an automorphism transports that property to every vertex. If every vertex of a
connected graph is simplicial, the graph is complete: otherwise a shortest path
of length two has a middle vertex with two nonadjacent neighbors.

Hence every finite connected simple vertex-transitive chordal graph is complete.
In particular, a finite connected undirected simple Cayley graph is chordal only
in the complete case. For an ordinary Cayley graph this requires the generating
set to connect the identity directly to every other group element.

This conclusion is finite. Infinite Cayley graphs can be chordal without being
complete; regular trees from free groups are the basic counterexample because
the finite simplicial-vertex argument does not transfer.

Schreier graphs require the same care about loops, multiple edges, and the
underlying simple graph. Directed positive alphabets fall outside standard
undirected chordality.

## 9. GPU and multi-GPU evidence boundary

Ordinary level-synchronous BFS can process a frontier largely as an unordered
set. LexBFS requires dynamically updated relative priorities from selected-neighbor
histories. A parallel traversal that preserves distances but changes these ties
is still valid BFS and may be invalid LexBFS.

Chordality evidence therefore separates into:

- construction of the exact graph or successor relation;
- LexBFS label/order maintenance;
- verification that every later-neighbor set is a clique;
- optional reconstruction of an induced-cycle witness after failure;
- ordinary BFS distance/frontier/visited measurements.

Distributing labels or clique checks introduces ordering and completeness
requirements different from frontier ownership. Report those costs separately;
do not describe a high-throughput BFS frontier kernel as a chordal recognizer.

## Sources

- D. J. Rose, R. E. Tarjan, and G. S. Lueker,
  [*Algorithmic Aspects of Vertex Elimination on Graphs*](https://doi.org/10.1137/0205021),
  SIAM Journal on Computing 5, 1976. LexBFS and linear-time elimination-order
  machinery.
- D. R. Fulkerson and O. A. Gross,
  [*Incidence Matrices and Interval Graphs*](https://doi.org/10.2140/pjm.1965.15.835),
  Pacific Journal of Mathematics 15, 1965. Classical perfect-elimination and
  clique structure underlying chordal/interval graph theory.
- N. Robertson and P. D. Seymour,
  [*Graph Minors X: Obstructions to Tree-Decomposition*](https://doi.org/10.1016/0095-8956(91)90061-N),
  Journal of Combinatorial Theory B 52, 1991. Treewidth context for clique-tree
  decompositions.

