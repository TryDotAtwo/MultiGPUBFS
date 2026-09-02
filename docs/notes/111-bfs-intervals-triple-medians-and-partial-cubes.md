# BFS intervals, triple medians, and partial cubes

BFS distances define intervals: the vertices that can lie on shortest paths
between endpoints. Median graphs impose a strong global rule on those intervals:
every triple has exactly one common median vertex. This is different from the
facility-location use of the word "median."

This note develops the distinction and its Cayley/GPU boundaries. It adds no
recognizer, embedding algorithm, or optimized BFS implementation.

## 1. Intervals from distance rows

For vertices `u,v`, the metric interval is

```text
I(u,v) = {x : d(u,x) + d(x,v) = d(u,v)}.
```

It is the union of the vertex sets of all shortest `u-v` paths. Two BFS distance
rows, one from each endpoint, can test membership exactly. One BFS tree cannot:
it selects one parent path and may omit vertices lying on alternative shortest
paths.

For a triple `a,b,c`, a metric median vertex lies in

```text
I(a,b) intersect I(b,c) intersect I(c,a).
```

The intersection can be empty, contain one vertex, or contain several vertices
in a general graph.

## 2. Median graphs

A connected graph is a median graph when the triple-interval intersection above
contains exactly one vertex for every triple. Denote it by `m(a,b,c)`.

Trees are median graphs: the three unique pairwise paths meet at one branch
point. Hypercubes are median graphs: encode vertices as bit vectors and take the
coordinatewise majority of the three vectors. Cartesian products of paths,
including rectangular grids, are further examples.

Triangles are not median graphs. For the triple of their three vertices, every
pair interval is just the corresponding edge endpoints and the three intervals
have empty common intersection. Complete graphs with at least three vertices
fail for the same reason.

Median graphs are bipartite and are isometric subgraphs of hypercubes. The
converse is false: `C6` is a partial cube but not a median graph. For the
alternating triple `0,2,4`, the three pair intervals have no common vertex.

## 3. Partial-cube coordinates

An isometric hypercube embedding assigns a binary coordinate vector `phi(v)` to
each graph vertex such that

```text
d_G(u,v) = Hamming(phi(u), phi(v)).
```

Each coordinate represents a convex cut: crossing that cut flips the coordinate.
In partial-cube language, edges are grouped into cut classes rather than treated
as unrelated transitions.

Relative to a BFS root `s`, layer depth equals the number of coordinates on
which `phi(v)` differs from `phi(s)`. This does not make frontier sizes binomial:
an arbitrary isometric subgraph contains only some cube vertices. The full
hypercube has binomial layers because every bit vector occurs, not merely because
Hamming coordinates exist.

In a median graph the median is coordinatewise majority in any isometric cube
representation, and the result is guaranteed to be a graph vertex. In a general
partial cube, the majority vector can be absent from the embedded vertex set;
that is precisely why partial-cube structure alone does not imply median
structure.

## 4. Triple median versus graph-median objective

For nonnegative vertex weights `w(v)`, a graph median set in the facility-location
sense minimizes

```text
f(x) = sum_v w(v) * d(x,v).
```

This is an optimization over an entire demand distribution. A median graph is a
class defined by unique medians of every three vertices. The names are related
historically and geometrically, but the contracts differ:

- a triple median takes exactly three terminals and returns one interval
  intersection in a median graph;
- a weighted graph median can use any number of demands and may have multiple
  minimizers;
- knowing one BFS profile generally cannot evaluate the sum objective for all
  candidate locations;
- unique triple medians do not mean every weighted facility-location instance
  has a unique solution.

For example, on a single edge with equal positive weight at both endpoints,
both endpoints minimize total distance. The graph is a tree and hence median,
but the weighted graph-median set is not a singleton.

## 5. Convexity, gates, and Helly are adjacent but distinct

In a median graph, convex sets are gated and finite pairwise-intersecting
families of convex sets have a common vertex. Nevertheless, median graphs are
not the same class as ball-Helly graphs.

The cycle `C4` is the two-dimensional hypercube, hence a median graph. Note 110
showed that its four radius-one balls pairwise intersect but have empty total
intersection. The reason is consistent: those balls are not geodesically convex.
Thus the Helly theorem for convex/gated sets cannot be applied to arbitrary balls
even inside a median graph.

## 6. Median does not mean tree-like or small

Median graphs generalize trees but can contain high-dimensional cubes and large
grids. Therefore:

- median does not imply zero or uniformly bounded Gromov hyperbolicity;
- median does not imply unique shortest paths;
- median does not imply small BFS frontiers;
- median does not imply a compact implicit state representation.

The `n`-cube has a unique median for every triple yet has peak BFS frontier
`binomial(n,floor(n/2))`. Large Cartesian grids are median while their
four-point hyperbolicity grows with scale. The median axiom organizes interval
intersections; it is not a memory or throughput bound.

## 7. Cayley generator dependence

Some median graphs are Cayley graphs. The hypercube is the Cayley graph of
`Z2^n` with the standard coordinate-flip generators. Translation preserves
intervals and therefore transports triple medians:

```text
g * m(a,b,c) = m(g*a, g*b, g*c).
```

But Cayley symmetry does not imply the median property, and the property depends
on the generating set. For `Z2^2`:

- the two coordinate generators produce `C4`, a median graph;
- adding the third nonidentity element as a generator produces `K4`, which is
  not median.

The group is unchanged while word distance, intervals, and triple medians
change. A Schreier quotient changes vertex identity again and requires direct
verification. Directed positive alphabets fall outside the symmetric interval
theory used here unless a directed analogue is explicitly defined.

## 8. BFS, GPU, and multi-GPU evidence

Three exact BFS rows are sufficient to materialize the three pair intervals of
one triple and test their intersection. That is a semantic oracle, not an
efficient general median-graph recognizer: recognition quantifies over all
triples unless stronger structure is exploited.

An already validated partial-cube coordinate representation could make
distance and majority calculations coordinatewise. Producing and validating
that representation is separate work; treating arbitrary puzzle coordinates or
hash bits as hypercube coordinates would be invalid because they need not
preserve graph distance.

For GPU or multi-GPU studies, keep separate:

- BFS distance-row generation;
- interval-membership filtering;
- triple-intersection or majority computation;
- validation of an isometric cube embedding;
- ordinary frontier/visited traversal throughput.

Median structure can expose parallel bit operations, but it supplies no
universal bound on frontier size, visited memory, duplicate pressure, routing,
or load balance.

## Sources

- H. M. Mulder,
  [*The Structure of Median Graphs*](https://doi.org/10.1016/0012-365X(78)90199-1),
  Discrete Mathematics 24, 1978. Original structural treatment and the unique
  triple-median definition.
- H.-J. Bandelt and J. Hedlikova,
  [*Median Algebras*](https://doi.org/10.1016/0012-365X(83)90173-5),
  Discrete Mathematics 45, 1983. Algebraic formulation of the ternary median
  operation.
- H.-J. Bandelt and V. Chepoi,
  [*Metric Graph Theory and Geometry: A Survey*](https://doi.org/10.1090/conm/453/08795),
  Contemporary Mathematics 453, 2008. Median graphs, gates, hypercube
  embeddings, and related graph-metric classes.
- H.-J. Bandelt,
  [*Centroids and Medians of Finite Metric Spaces*](https://doi.org/10.1002/jgt.3190160404),
  Journal of Graph Theory 16, 1992. Weighted total-distance median terminology.

