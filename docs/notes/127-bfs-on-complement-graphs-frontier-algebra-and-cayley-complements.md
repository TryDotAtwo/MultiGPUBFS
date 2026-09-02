# BFS on complement graphs, frontier algebra, and Cayley complements

The complement of a simple graph keeps the vertex set and flips adjacency
between every distinct pair. BFS on that complement is ordinary exact BFS for
a different graph, but its frontier operation is most naturally expressed by
non-adjacency in the original graph. This makes it a useful test of whether
frontier and visited semantics have been separated from one concrete adjacency
layout.

This note adds no implementation, optimizer, benchmark, or GPU code.

## 1. Complement contract

For a finite simple undirected graph `G=(V,E)`, its complement `bar(G)` has the
same vertices and

```text
uv in E(bar(G))  iff  u != v and uv not in E(G).
```

Thus

```text
A_bar = J-I-A
deg_bar(v) = |V|-1-deg_G(v).
```

Loops, parallel edges, labels, and weights do not have one automatic
complement convention. Everything below uses the simple undirected contract
unless a section explicitly says otherwise.

## 2. What complement distance means

For distinct vertices `u,v`:

- if `uv` is absent from `G`, then `d_bar(u,v)=1`;
- if `uv` is present in `G`, then `d_bar(u,v)=2` exactly when some third vertex
  is adjacent to neither `u` nor `v` in `G`;
- otherwise their complement distance is at least three or they lie in
  different complement components.

There is no reciprocal formula such as `d_bar=n-d_G` or a fixed monotone
relation. `P_5` has diameter four while its complement has diameter two;
`K_n` has diameter one while its complement is disconnected for `n>1`.

Complementing twice restores the graph, but it does not make the two shortest
path metrics numerical inverses.

## 3. Disconnection forces a shallow complement

If `G` is disconnected and `|V|>=2`, vertices in different components are
adjacent in `bar(G)`. Two vertices in the same original component can route in
the complement through any vertex of another component. Therefore

```text
diam(bar(G)) <= 2.
```

It equals one when `G` is edgeless and equals two when `G` has at least one
edge. By symmetry, if `bar(G)` is disconnected, then `G` is connected with
diameter at most two. In particular, a graph and its complement cannot both be
disconnected when there are at least two vertices.

Complement components are often called co-components. If `bar(G)` has several
components, `G` contains every cross-edge between them: `G` is the join of the
induced subgraphs on those co-components.

## 4. Large diameter collapses to two

Suppose connected `G` has diameter at least four. Consider any original edge
`uv`. If every other vertex were adjacent in `G` to `u` or `v`, any pair could
route through `u-v` in at most three steps, contradicting the diameter.
Therefore some vertex is adjacent to neither endpoint, giving a length-two
complement path between `u` and `v`.

Original nonedges are already complement edges, so every distinct pair is at
complement distance at most two. Since `G` has edges, the complement is not
complete. Hence

```text
diam(G)>=4  implies  diam(bar(G))=2.
```

This is a metric theorem, not a claim that complement traversal has little
work: a diameter-two graph can have an enormous first or second frontier.

## 5. Complement frontier algebra

Let `B_i` be the vertices visited through complement depth `i`, `F_i` the
current complement frontier, and `U_i=V\B_i` the unvisited candidates. Then

```text
F_(i+1)
  = {x in U_i : exists v in F_i with xv not in E(G)}
  = U_i \ intersection_(v in F_i) N_G(v).
```

A candidate stays undiscovered only if it is adjacent in the original graph to
every current frontier vertex.

The intersection is crucial. Replacing it by the union of original
neighborhoods is wrong: if `x` is adjacent in `G` to frontier vertex `a` but
not to frontier vertex `b`, then `b-x` is a valid complement edge and `x` must
enter the next frontier.

## 6. Visited still owns the shortest-depth invariant

Complement adjacency changes successor generation, not BFS's proof:

- `B_i` is the exact complement ball through depth `i` after the layer is
  complete;
- `F_i` is its exact complement sphere;
- first authoritative discovery fixes minimum complement depth;
- duplicates arise when several frontier vertices are original nonneighbors of
  the same candidate.

Original-graph visited state cannot be reused as complement visited state: the
source and vertex IDs may match, but the graph epoch and metric differ.

## 7. Explicit versus implicit complement

A sparse `G` can have `Theta(n^2)` complement edges. Materializing all of them
changes the representation size before BFS begins. An implicit complement
instead answers whether a distinct candidate pair is absent from `E(G)` and
maintains the unvisited set without listing every complement edge.

Ito and Yokoyama proved that BFS and DFS trees of a complement graph can be
constructed in time linear in the size of a suitable representation of the
given graph. This existence result shows that explicit complement size is not
an unavoidable semantic cost. It does not make every naive nonedge scan linear,
and this note does not select or implement a data structure.

The work counters must say whether they count original adjacency inspections,
nonedge tests, unvisited-set operations, logical complement edges, or physically
materialized edges.

## 8. Component and reachability evidence

A BFS forest in `bar(G)` partitions vertices into co-components. Contracting
those co-components does not describe sparse original adjacency: every pair of
distinct co-components is completely joined in `G`.

Conversely, knowing original connected components immediately proves that the
complement is connected with diameter at most two, but it does not identify
the exact complement BFS parents or multiplicities. Structural reachability,
distance, and witness outputs remain distinct.

## 9. Cayley complements

Let a finite group `Gamma` have a simple right Cayley graph with generator set
`S subset Gamma\{e}`. Under the directed simple-pair complement convention,

```text
bar(Cay(Gamma,S)) = Cay(Gamma, Gamma\({e} union S)).
```

For an undirected Cayley graph, inverse closure of `S` makes the complementary
generator set inverse closed as well. Thus the complement remains a Cayley
graph and one complete identity BFS gives its component metric and, when
connected, its diameter.

This does not preserve the original puzzle problem. A complement generator is
an arbitrary group displacement that was not an original legal move. A
length-one complement solution usually says precisely that the target is not
one original generator away; it is not a short original move sequence.

If the original Cayley graph is disconnected and nontrivial, the complement is
connected with diameter at most two. If the original connected Cayley graph has
diameter at least four, its complement Cayley graph has diameter two. These
facts follow from general complement geometry plus Cayley symmetry, not from a
new search shortcut for the original word metric.

## 10. Schreier, labels, and directed boundaries

For a Schreier or action graph, complementing the underlying simple graph
forgets which generator supplied an edge and may turn many absent state pairs
into unlabeled adjacencies without any group action that realizes them as one
move. The result is not automatically a Schreier graph for a meaningful small
alphabet.

For directed graphs one must declare whether the complement flips ordered arcs,
whether reverse arcs are independent, and how loops are treated. Strong
connectivity and forward/reverse distance then require separate analysis. The
simple undirected diameter claims above cannot be copied unchanged.

## 11. GPU and multi-GPU boundary

Complement BFS exposes a difference between logical degree and stored input:
the logical graph may be dense while the stored original is sparse. Report
separately:

- original and complement vertex/edge counts;
- explicit materialization or implicit nonedge predicate;
- frontier candidates, original adjacency tests, and logical complement edges;
- unvisited-set representation and authoritative discovery;
- duplicate candidates and parent policy;
- owner-local and cross-owner nonedge decisions;
- communication caused by candidate ownership versus frontier ownership;
- graph construction, traversal, and validation time.

In multi-GPU execution, absence of a local edge record is not by itself proof
of a complement edge: another shard may own the original adjacency evidence.
The partition must provide authoritative nonedge knowledge or a globally valid
membership query. This is a correctness boundary before it is a performance
question.

## Sources

- F. Harary, [*Graph Theory*](https://archive.org/details/graphtheory0000hara),
  Addison-Wesley, 1969, for classical graph-complement terminology and
  connectivity relations.
- H. Ito and M. Yokoyama,
  [*Linear Time Algorithms for Graph Search and Connectivity Determination on Complement Graphs*](https://doi.org/10.1016/S0020-0190(98)00071-4),
  Information Processing Letters 66(4), 1998. Establishes implicit complement
  BFS/DFS without explicit quadratic materialization.
- Notes 16, 21, 29, 32, 37, 51, 52, 57, 62, 75, 92, and 124 supply this
  repository's action, diameter, complexity, layer, contract, ownership,
  visited, output, CayleyPy identity, direction, reachability, and graph-epoch
  boundaries.

## Takeaway

Complement BFS is ordinary BFS on nonedges, but its next frontier is the
unvisited set minus the intersection of original neighborhoods, not minus their
union. Complementation can collapse disconnection or large diameter to at most
two while still creating a dense logical frontier. In a finite Cayley graph the
complement can again be Cayley under exact simple-graph conventions, yet it
computes a different word metric whose edges are not original legal moves.
