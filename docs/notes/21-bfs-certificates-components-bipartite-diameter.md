# What one BFS certifies: reachability, bipartiteness, and diameter

A complete BFS result contains more than a shortest path tree, but less than
many global graph claims attributed to it. The exact certificate depends on
directedness, connectivity, stopping condition, and symmetry.

## Reachability is the primary certificate

After exhaustive BFS from source `s`, visited is exactly

```text
R+(s) = {v | a directed path s -> v exists}.
```

In an undirected graph this is the connected component containing `s`. Repeating
BFS from one unvisited vertex per component produces a BFS forest and counts
connected components.

In a directed graph, one forward BFS gives only the outward reachable set. It
does not certify:

- weak connectivity unless edge directions are explicitly ignored;
- strong connectivity;
- the strongly connected component containing `s`.

If every vertex is reached from `s`, `s` can reach all vertices, but they may
not reach `s`. A reverse-graph BFS from `s` supplies the complementary condition
for strong reachability to/from that root; full SCC decomposition has its own
algorithmic contract.

An early target stop does not certify the whole reachable set. Exhaustion—or a
declared complete depth bound—is part of the certificate.

## BFS tree as positive evidence

For every reached non-source vertex, a parent edge with depth one less provides
a replayable path witness. This certifies membership in `R+(s)` and the recorded
distance when layer correctness is established.

Unreached vertices need a different argument: only complete expansion of every
reachable frontier proves no path exists. Absence from a partially explored
visited table is not an unreachability certificate.

## Bipartiteness from layer parity

For a connected undirected graph, color every vertex by

```text
color(v) = distance(s,v) mod 2.
```

Every edge joins vertices whose BFS depths differ by at most one. Therefore:

- if no edge has equal-depth endpoints, every edge crosses parity classes and
  the graph is bipartite;
- if an edge `x--y` joins one layer, the two BFS-tree paths from `s` to `x,y`
  plus that edge contain an odd cycle.

To extract a simple odd-cycle witness, remove the shared tree prefix up to the
lowest common ancestor `z`. If `x,y` are at depth `d` and `z` at depth `h`, the
cycle length is

```text
(d-h) + 1 + (d-h) = 2(d-h)+1.
```

For a disconnected graph the test must cover every component. For a directed
graph, "bipartite" usually refers to the underlying undirected endpoints; that
convention must be stated.

Same-level edge detection is a property of exact BFS depths, not merely of an
arbitrary spanning tree level assignment.

## Eccentricity is exactly one complete BFS maximum

For a finite connected undirected graph,

```text
ecc(s) = max_v dist(s,v).
```

The index of the last nonempty complete frontier is exactly `ecc(s)`. This is a
source-specific quantity.

Graph diameter and radius are

```text
diam(G) = max_s ecc(s)
rad(G)  = min_s ecc(s).
```

Thus one BFS gives

```text
ecc(s) <= diam(G)
rad(G) <= ecc(s),
```

but ordinarily neither global extremum exactly.

For directed graphs, out-eccentricity uses directed distances from `s`. If some
vertices are unreachable, conventions may assign infinity or restrict the
maximum to the reachable set. A reported "directed diameter" must state its
connectivity and infinity convention.

## Why a farthest vertex need not be peripheral

A vertex `u` farthest from chosen root `s` satisfies

```text
dist(s,u) = ecc(s).
```

It need not satisfy

```text
ecc(u) = diam(G).
```

The second property defines a peripheral vertex. Confusing them motivates the
incorrect universal two-sweep diameter claim.

## Double sweep: exact on trees, heuristic in general

Two sweep performs:

1. BFS from arbitrary `s`, choose a farthest `u`;
2. BFS from `u`, return `ecc(u)` and a farthest `v`.

On a tree, a farthest vertex can be used as a diameter endpoint and the second
sweep returns the exact diameter. Cycles and alternate routes break the tree
argument.

REF-021 exhaustively found this connected seven-vertex graph:

```text
edges = 0-2, 0-4, 0-6, 1-2, 1-4, 1-5, 2-3
start = 4
unique farthest from 4 = 3 at distance 3
ecc(3) = 3
diameter = 4, witnessed by 5-1-2-0-6.
```

The first farthest is unique, so tie-breaking cannot save the run. Two-sweep
returns a valid lower bound `3`, not the true diameter `4`.

## Diameter bounds from any BFS

For any source `s` in a connected undirected graph:

```text
ecc(s) <= diam(G) <= 2*ecc(s).
```

The lower bound is immediate. For any `u,v`, triangle inequality gives

```text
dist(u,v) <= dist(u,s)+dist(s,v) <= 2*ecc(s).
```

If the exact diameter is required on a generic graph, repeated/all-source BFS or
a separately proved bounding algorithm is needed. A two-sweep result should be
labeled lower bound unless its graph-family theorem applies.

## Why one Cayley BFS gives diameter

For right Cayley edges `g -> g*s`, left multiplication `L_a(g)=a*g` is a graph
automorphism. It maps any vertex to any other and preserves directed labels and
distances. Therefore every group element has the same eccentricity.

For a finite connected inverse-closed Cayley graph,

```text
diam(Cay(G,S)) = ecc(identity) = max_g length_S(g).
```

One exhaustive BFS from identity is enough. This is not a BFS miracle; it is a
consequence of vertex transitivity.

The statement also extends to a finite strongly connected directed Cayley graph
using directed out-distances: left translation makes every out-eccentricity
equal. If the positive generator monoid does not reach all represented group
elements, one source BFS only describes its reachable directed component and an
ambient finite-diameter claim needs different conventions.

## Why the same claim need not hold for Schreier/puzzle graphs

A Schreier graph under a fixed generator set is not automatically vertex-
transitive as a graph merely because the underlying group action on states is
transitive. Moving all states by a group element can conjugate/rename the fixed
generators rather than preserve the same edge relation.

Therefore a complete BFS from one puzzle state yields that state's eccentricity,
not automatically the puzzle-graph diameter. One-BFS diameter is valid only if
an actual graph-automorphism group is proved transitive under the precise move
set, direction, labels/costs relevant to distance.

### Three-point Schreier witness

Let `S_3` act transitively on points `{1,2,3}` and use the inverse-closed fixed
generator set

```text
S={(12),(23)}.
```

The labeled point successors are

```text
1 -> 2,1
2 -> 1,3
3 -> 3,2.
```

Ignoring self-loops for scalar distance, the support graph is the path

```text
1 -- 2 -- 3.
```

This same fixture separates two parity contracts. At point `1`, both the
identity and the odd stabilizer element `(23)` represent the same state. Thus
permutation sign does not descend from `S_3` elements to point states. The
labeled Schreier graph exposes the failure directly: generator `(23)` creates
a loop at `1` (and `(12)` creates one at `3`), so that loop-retaining graph is
not bipartite.

After loops are suppressed, however, the simple support path `1--2--3` is
bipartite. Its coloring is a property of that simplified state graph, not a
descended group-sign character. Therefore failure of Cayley parity on Schreier
cosets does not imply that every simplified support convention is
nonbipartite; deleting labeled loops can change the predicate itself.

An exhaustive BFS from middle state `2` has last depth one, so
`ecc(2)=1`. The graph diameter is two, attained between points `1` and `3`.
The underlying `S_3` action is transitive, but applying an arbitrary point
permutation conjugates the fixed transpositions and need not preserve the set
`{(12),(23)}`. Hence action transitivity does not supply graph-automorphism
transitivity for this move metric.

For the regular Cayley action on group elements, left multiplication preserves
right edges `g -> g*s` without conjugating `s`; that is the missing property
which makes one Cayley BFS sufficient.

### Same complete occurrence total, different frontier profile

The same three-point fixture also separates total work from its level shape.
Every state has exactly two labeled generator occurrences, so any complete
state BFS expands three states and generates

```text
3 * 2 = 6 occurrences.
```

From middle root `2`:

```text
frontier sizes:       1,2
occurrences by level: 2,4
```

Both root occurrences are outward; the last level generates two inward returns
to `2` and two endpoint self-loops.

From endpoint root `1`:

```text
frontier sizes:       1,1,1
occurrences by level: 2,2,2
```

The first level mixes one outward transition with one self-loop, the middle
level one inward with one outward transition, and the last level one inward
with one self-loop.

Both runs accept exactly two nonroot states and reject four occurrences, yet
their depth, peak frontier, per-level batches, and rejection types differ. Thus
equal complete generated work does not imply equal temporal parallelism,
buffer pressure, synchronization count, or target-stopped work. This is a
workload-shape observation, not a GPU performance prediction.

### Cayley root translation preserves the whole semantic wave

For right Cayley edges `g -> g*s`, left multiplication by `r^(-1)` maps a BFS
rooted at `r` to one rooted at identity:

```text
r -> e,
g -> r^(-1)g,
g*s -> r^(-1)g*s.
```

The move label `s`, path length, and endpoint equality are unchanged. Hence

```text
F_d(r) = r F_d(e)
```

as translated state sets. Every root has the same:

- frontier cardinality at every depth;
- eccentricity/diameter profile;
- number of generated labeled occurrences per level;
- outward, same-layer, and old-ball occurrence counts;
- translated shortest-predecessor structure and labeled-word multiplicities.

This holds for directed right Cayley graphs as well, relative to their
translated reachable components, because left translation preserves the
positive generator transitions.

The theorem is semantic, not a statement about one implementation's physical
balance. A dense rank, memory layout, hash owner, or graph partition need not
commute with left translation. Two translated frontiers can therefore have
identical global BFS structure but different per-rank counts, locality, routing
traffic, or cache behavior. Root-invariant Cayley geometry does not imply
root-invariant execution placement.

Symmetry quotienting can also reduce distances to orbits, so the quotient's
maximum layer is its own eccentricity/diameter object, not automatically that of
the concrete graph.

## Cayley parity and bipartiteness certificate

In an inverse-closed Cayley graph, a homomorphism `chi:G->Z_2` mapping every
generator to `1` colors every edge across parity. BFS depth parity from identity
matches `chi`.

A same-level Cayley edge exposes an odd relation/odd cycle relative to the
generator alphabet. Identity generators create loops, immediately refuting
bipartiteness even though they do not change distances. Note 16 gives the
stabilizer condition for this parity to descend to a Schreier graph.

## Multi-source does not directly give all eccentricities

Multi-source BFS computes distance to the nearest source:

```text
min_(s in S) dist(s,v).
```

It does not retain each source's distance field and therefore cannot replace
many independent BFS runs for generic diameter/radius computation. Seeding all
vertices makes every distance zero, not the diameter.

This is another instance where batching sources changes the mathematical output
unless source identity remains a separate dimension.

### `Z_4` witness: a shallower joint wave is not a faster diameter BFS

In the Cayley cycle of `Z_4` with generators `{+1,-1}`, one source at zero
gives

```text
F_0={0},  F_1={1,3},  F_2={2}.
```

Its maximum depth is two, which equals the Cayley graph diameter by root
translation symmetry. Now initialize opposite sources `{0,2}` jointly:

```text
F_S(0)={0,2},  F_S(1)={1,3}.
```

The joint wave has maximum depth one, but the graph diameter is still two.
The value one is the covering radius of this source set,

```text
max_v min(dist(0,v),dist(2,v)),
```

not the maximum distance between graph vertices. The joint run starts wider
and ends sooner because it solves nearest-source distance. Running independent
BFS instances from several roots may share execution machinery, but preserving
diameter/all-source semantics requires retaining the source dimension instead
of merging all sources into one visited wave.

## GPU and multi-GPU certification

A claimed complete eccentricity/component/diameter result needs evidence that:

- every final frontier was expanded without overflow;
- distributed candidates and visited ownership converged exactly;
- global empty frontier was established after all in-flight work;
- maximum depth was reduced across all owners;
- graph symmetry assumptions apply to the exact represented move graph;
- the run was exhaustive rather than target-stopped or depth-bounded.

Peak depth from one rank's local frontier is not global eccentricity. Maximum
depth from a complete distributed Cayley BFS is diameter only after the
transitivity and reachability premises are explicit.

## Counterexamples and rejected shortcuts

- **One forward BFS proves strong connectivity.** It proves only that its source
  reaches every reported vertex.
- **An unvisited vertex is unreachable.** Only after complete exhaustion under
  exact expansion/visited semantics.
- **The last BFS layer is graph diameter.** It is source eccentricity in general.
- **Every farthest vertex is peripheral.** REF-021 gives a unique farthest
  vertex with eccentricity below diameter.
- **Two-sweep is exact on every unweighted graph.** It is exact on trees and
  selected families, a lower-bound heuristic generally.
- **One Schreier BFS gives diameter because the group action is transitive.**
  The fixed generator graph still needs a vertex-transitive automorphism proof.
- **Multi-source BFS computes all-source distances.** It computes their pointwise
  minimum unless source identity is retained.

## Audit checklist

1. Was BFS exhaustive, target-stopped, or depth-bounded?
2. Is the graph directed, undirected, or symmetrized for the claimed property?
3. Does visited mean one reachable set, one component, or all components?
4. Is the maximum layer called eccentricity or diameter?
5. What theorem makes the selected source peripheral/representative?
6. Is a two-sweep result exact or only a lower bound on this graph family?
7. Does an equal-depth edge yield a replayable odd-cycle witness?
8. Is Cayley/Schreier vertex transitivity proved for the exact move graph?
9. Did global termination account for every in-flight distributed candidate?
10. Are unreachable directed pairs treated as infinity or excluded?

## Sources

- MIT OpenCourseWare 6.006, *Breadth-First Search*,
  [Lecture 13](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2008/resources/lec13/),
  for BFS reachability, shortest paths, and layer structure.
- IIT Delhi COL106, *Verifying Bipartiteness*,
  [lecture notes](https://web.iitd.ac.in/~keerti/Courses/COL106-2023-Notes/Lec_25.pdf),
  for the equal-level-edge bipartiteness characterization.
- Michael DeVos, *Pretty Theorems on Vertex Transitive Graphs*,
  [notes](https://www.sfu.ca/~mdevos/notes/misc/vertex-trans.pdf), for distance
  geometry under vertex transitivity.
- Crescenzi et al., *Finding the Diameter in Real-World Graphs*,
  [record](https://arpi.unipi.it/handle/11568/142112), for two-sweep as a general
  diameter lower-bound method.
- Local evidence: REF-021 gives the exact Docker/Rust general-graph
  counterexample; notes 10 and 16 give Cayley growth, parity, and transitivity
  conventions.

## Current synthesis

One exhaustive BFS canonically certifies one source's reachable set, distances,
and eccentricity; it can also certify bipartiteness with edge checks. Diameter
is a global maximum and needs more, except when symmetry proves every source
equivalent. Finite connected Cayley graphs have that symmetry, while generic
graphs and fixed-generator Schreier graphs do not automatically. The name of
the reported quantity—eccentricity, lower bound, orbit diameter, or exact
diameter—is part of correctness.
