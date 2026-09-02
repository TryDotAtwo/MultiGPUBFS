# BFS separators, dominators, and Menger paths

Dominance says that one vertex intercepts every root-to-target path. Menger's
theorem places this statement inside a wider min-max picture: the minimum
number of vertices needed to intercept all paths equals the maximum number of
routes that can avoid sharing internal vertices.

This note connects that picture to BFS layers and Cayley graphs. It records
semantic consequences and counterexamples, not an algorithm implementation.

## 1. Local vertex separation

Fix distinct vertices `s,t` in a finite directed graph. An internal
`s`-to-`t` vertex separator is a set

```text
X subseteq V \ {s,t}
```

whose deletion destroys every directed path from `s` to `t`. Let
`kappa(s,t)` be the minimum size of such a set.

Two directed `s`-to-`t` paths are internally vertex-disjoint when they share no
vertices except `s` and `t`. Menger's directed vertex theorem states, when
there is no direct arc `s->t`, that

```text
minimum internal separator size
  = maximum number of pairwise internally vertex-disjoint s-to-t paths.
```

The no-direct-arc qualification matters: a direct path has no internal vertex
to delete. One can instead subdivide that arc or use an extended convention,
but the convention must be stated rather than silently treating an empty
internal path as interceptable.

## 2. Dominators are singleton separators

For `d` distinct from `s,t`, note 89's deletion characterization says

```text
d dominates t  <=>  {d} is an internal s-to-t separator.
```

Therefore, in the nonadjacent reachable case:

- a nontrivial dominator exists exactly when `kappa(s,t)=1`;
- no nontrivial dominator exists exactly when there are at least two internally
  vertex-disjoint directed paths from `s` to `t`.

This is stronger than finding two different paths. The paths must avoid sharing
every internal vertex.

Several dominators can occur in series. In

```text
s -> a -> b -> t,
```

both `{a}` and `{b}` are minimum separators of size one. The minimum cut value
does not list all unavoidable vertices; the dominator chain does.

## 3. Path multiplicity is not path independence

Use

```text
s -> a -> x -> t
s -> b -> x -> t.
```

There are two distinct shortest paths and `sigma_s(t)=2`, but both use `x`.
Thus `x` dominates `t`, and the maximum number of internally vertex-disjoint
paths is only one.

Consequently:

```text
number of paths != number of disjoint paths.
```

Shortest-path counts can certify whether one named vertex lies on every
shortest path, as in note 89. They do not by themselves measure vertex
connectivity, and two different BFS parents at one join do not prove two
end-to-end disjoint routes.

## 4. BFS spheres are separators, rarely minimum ones

Let `dist(s,t)=D`, and let

```text
S_i = {v : dist(s,v)=i},  0 < i < D.
```

Every directed `s`-to-`t` path meets `S_i`: along an arc, distance from `s` can
increase by at most one, so a route cannot first reach distance `D` without
passing through every intermediate distance. Hence `S_i` is an internal
separator, as studied in note 48.

Menger immediately gives

```text
kappa(s,t) <= |S_i|
```

for every intermediate BFS layer. Equivalently, `k` internally disjoint paths
must cross a layer at `k` distinct vertices.

The bound can be arbitrarily loose. Build a huge first layer
`a_1,...,a_m`, send every branch into one vertex `x`, then use `x->t`:

```text
s -> a_j -> x -> t,  for j=1,...,m.
```

Here `|S_1|=m`, but `kappa(s,t)=1` because `x` dominates `t`. Frontier width is
neither the minimum cut nor the number of independent routes.

## 5. The shortest-path DAG has its own Menger question

Apply Menger's theorem to the shortest-path DAG rather than to the full graph.
Then the minimum separator and disjoint paths are restricted to shortest
routes. A singleton separator is exactly a shortest gateway from note 89.

The value can differ from the full-graph value. In

```text
s -> a -> t
s -> b -> c -> t,
```

the shortest-path DAG has singleton gateway `a`, while the original graph has
two internally vertex-disjoint paths and no nontrivial dominator. Deleting
non-shortest arcs can only remove alternatives; it may create separators that
do not exist in the original graph.

## 6. Vertex and edge resilience differ

The edge version of Menger relates minimum deleted arcs to the maximum number
of edge-disjoint paths. This is a different failure model.

In the two-branch graph that rejoins at `x`, the two paths can be edge-disjoint
up to and including separate parallel exits if those exits exist, while still
sharing vertex `x`. Conversely, parallel labeled arcs in a Cayley multigraph
may raise arc-disjoint multiplicity without creating another state-level route.

Therefore every resilience claim must specify:

- removed vertices, semantic edges, or labeled arc occurrences;
- simple graph or multigraph identity;
- directed or undirected paths;
- all paths or shortest paths only.

## 7. Cayley examples

A one-way directed cycle generated by `+1 mod n` has one directed route segment
from `s` to a nonadjacent `t` before completing another lap. Its local internal
vertex connectivity is one, and every intermediate state on that segment is a
dominator. Strong connectivity and regular degree do not provide route
redundancy.

In an undirected cycle, the clockwise and counterclockwise `s`-to-`t` paths are
internally vertex-disjoint. Thus a nonadjacent pair has local connectivity two
and no nontrivial root-to-target dominator. The same vertex set and cyclic
symmetry support different conclusions under different generator-direction
contracts.

For labeled parallel generators, arc-disjointness may count separate generator
occurrences while vertex-disjointness still sees the same intermediate states.
For Schreier graphs or product states, separation must again be evaluated in
the declared state graph.

## 8. Distributed interpretation

Multiple owners, replicas, messages, or parent records are physical
redundancy—not proof of internally disjoint semantic paths. Conversely, two
semantic paths may be stored largely by the same owner.

To claim separator size or route multiplicity from a distributed traversal:

- all relevant arcs must belong to one consistent graph epoch;
- vertex identity must be globally exact across owners;
- duplicated storage must not be counted as distinct semantic vertices;
- path witnesses must be checked for internal disjointness, not only distinct
  parents or endpoints;
- a shortest-DAG certificate must not be promoted to an all-path claim.

No communication or GPU layout follows uniquely from these conditions.

## 9. Evidence checklist

1. Source/target adjacency and separator convention.
2. Vertex-disjoint or edge/arc-disjoint routes.
3. Directed, undirected, simple, labeled, or multigraph semantics.
4. Full graph or shortest-path DAG.
5. Path count versus pairwise internal disjointness.
6. BFS sphere size versus proved minimum separator size.
7. Cayley generator direction and parallel-label identity.
8. Logical path redundancy versus physical owner/replica redundancy.

## Sources

- K. Menger,
  [*Zur allgemeinen Kurventheorie*](https://doi.org/10.4064/fm-10-1-96-115),
  Fundamenta Mathematicae 10 (1927), 96-115. Original source of the
  path-versus-separator theorem.
- J. Bang-Jensen and G. Gutin, [*Digraphs: Theory, Algorithms and
  Applications*](https://www.cs.rhul.ac.uk/books/dbook/main.pdf), Section 7.3.
  Directed vertex and arc versions, including the nonadjacent-endpoint
  qualification and vertex-splitting relation.
- Notes 28, 30, 37, 48, 84, and 89 supply shortest-DAG, path-count,
  multigraph-identity, BFS-separator, strong-connectivity, and dominance
  context.

## Takeaway

A dominator is a separator of size one. Two distinct paths do not refute it;
two internally vertex-disjoint paths do. BFS layers intercept all deeper
routes, but their width only upper-bounds the minimum separator and can be
arbitrarily larger. The graph used for the claim—full, shortest-only, directed,
undirected, simple, or labeled—changes the answer.
