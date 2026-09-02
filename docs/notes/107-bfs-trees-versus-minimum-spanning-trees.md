# BFS trees versus minimum spanning trees

A BFS tree, a shortest-path tree, and a minimum spanning tree are all spanning
trees under suitable connectivity assumptions. They optimize different
objects. Similar output shape does not imply interchangeable guarantees.

This note separates root distance, total selected-edge weight, bottleneck
paths, and tree geometry. It adds no tree algorithm implementation.

## 1. Three objectives

For a connected undirected graph rooted at `s`:

- a **BFS tree** in an unweighted graph preserves every hop distance from `s`;
- a **shortest-path tree** in a nonnegative weighted graph preserves every
  minimum weighted distance from `s`;
- a **minimum spanning tree** minimizes the sum of its `|V|-1` selected edge
  weights, with no distinguished root.

Formally, an SPT `T_s` satisfies

```text
d_(T_s)(s,v)=d_G(s,v) for every v.
```

An MST `T` minimizes

```text
sum_(e in T) w(e)
```

over all spanning trees. The first constraint is pointwise and root-relative;
the second objective is global and root-free.

## 2. Unit weights make every spanning tree an MST

If every edge has weight one, every spanning tree has exactly `|V|-1` edges
and total weight `|V|-1`. Therefore every spanning tree is an MST.

Only some of them are BFS trees from a chosen root. In the complete graph
`K_n`, graph distance from root `s` to every other vertex is one. A BFS tree
must connect every non-root vertex directly to `s`, so it is a star.

A Hamiltonian path rooted at one endpoint is also an MST under unit weights,
but its farthest tree distance is `n-1`. Thus

```text
unweighted MST  does not imply  BFS tree.
```

Conversely, every unweighted BFS tree is trivially an MST because every
spanning tree ties on total weight. This implication is mathematically true but
contains no extra MST information.

## 3. A weighted MST need not be a shortest-path tree

Take a triangle with root `s` and weights

```text
w(s,a)=2,
w(a,b)=2.1,
w(s,b)=3.
```

The unique MST selects `{s,a}` and `{a,b}` with total `4.1`. Its tree path from
`s` to `b` costs `4.1`, while the graph has direct shortest path `{s,b}` of
cost `3`.

The SPT selects `{s,a}` and `{s,b}` with total `5`, which is not minimum among
spanning trees. One three-vertex graph therefore rejects both implications:

```text
MST -> SPT,
SPT -> MST.
```

## 4. The objectives can diverge by a large factor

Let root `s` have a direct weight-one edge to each of `n-1` other vertices.
Also connect those other vertices in a chain with edge weight `1-epsilon`, for
small positive `epsilon`.

Every direct root edge is the unique shortest route to its endpoint: any route
through another leaf begins with cost one and then adds positive chain cost.
The root SPT is therefore the star with total weight `n-1`.

An MST can use one root edge plus the `n-2` cheaper chain edges, with total

```text
1 + (n-2)(1-epsilon).
```

Its root-to-far-leaf tree distance can approach `n-1`, while the original graph
distance is one. By choosing a much smaller positive chain weight, the SPT's
total selected weight can also be a linear factor larger than MST weight.

The examples show two independent distortions:

- MST can have large root-distance stretch;
- SPT can have unnecessarily large total tree weight.

No constant-factor relation follows without additional graph assumptions.

## 5. MST does preserve a bottleneck notion

For vertices `x,y`, consider minimizing the largest edge weight along a path:

```text
beta_G(x,y) = min_(paths P from x to y) max_(e in P) w(e).
```

The unique `x-y` path in any MST is a minimax/bottleneck-optimal path: its
largest edge weight equals `beta_G(x,y)`. An exchange/cut argument proves this;
if another path had every edge strictly lighter than the heaviest MST-path
edge, it could reconnect the cut after removing that edge and lower the tree.

This does not minimize the sum of path weights or number of hops. MST preserves
a threshold-connectivity fact, not BFS distance.

## 6. Parent choices and uniqueness differ

BFS can have many valid parents for one vertex when several shortest paths tie.
All resulting BFS trees preserve root distances but can have different shapes.

MST can have multiple optima when edge weights tie. Distinct MSTs preserve
minimum total weight but can give different root distances. Distinct sources
do not change the MST objective, whereas changing the BFS source usually
changes every layer and parent choice.

Distinct edge weights guarantee a unique MST, but do not make it an SPT.
Unique shortest paths from `s` guarantee a unique SPT parent relation, but do
not make that tree minimum-total-weight.

## 7. Certificates are different

A BFS/SPT certificate uses distance feasibility and predecessor witnesses:

```text
d(v) <= d(u)+w(u,v),
```

with equality along a retained shortest parent and the appropriate weighted
algorithm assumptions.

An MST certificate uses cut and cycle properties: selected light edges must be
safe across cuts, and a non-tree edge cannot replace a heavier tree edge on its
fundamental cycle to reduce total weight.

Checking one certificate does not validate the other. Both outputs containing
`|V|-1` edges proves only that they are tree-sized candidates.

## 8. Dynamic updates affect them differently

Adding a light shortcut can change BFS/SPT distances and many parents while
leaving the MST unchanged if the edge is not useful for total tree weight.
Conversely, changing one edge weight can replace an MST edge while preserving
all unweighted hop distances and BFS layers.

Edge deletion may invalidate a tree edge in either structure, but the repair
goal differs: restore pointwise root distances versus restore minimum total
weight. A dynamic connectivity certificate alone supplies neither objective.

## 9. Cayley interpretation

In a finite connected unit-generator Cayley or Schreier graph, every spanning
tree is an MST under unit edge weights. The term "minimum spanning" therefore
does not select the BFS tree of geodesic normal forms.

A root BFS tree selects one shortest generator word per state. An arbitrary MST
can select long root paths even though its total unit weight is equally optimal.

With unequal generator costs, an MST minimizes the total cost of a global tree
connecting all states. It need not preserve minimum weighted word cost from the
identity. Generator symmetry of the graph also need not be preserved by one
selected finite tree.

## 10. GPU and multi-GPU interpretation

MST and BFS may share primitives such as edge scans, filtering, prefix sums,
sorting, or component operations. Their dependency structures and validation
targets remain different:

- BFS advances source-relative metric frontiers;
- Kruskal-style MST processes edges by weight and prevents cycles;
- Boruvka-style MST selects component outgoing edges;
- MST may use union-find without producing BFS distances;
- distributed MST must reconcile component/cut decisions;
- distributed BFS must reconcile exact first discovery and depth.

Primitive throughput transfers only after the surrounding work, ordering, and
output contract are measured. No implementation strategy is selected here.

## 11. Evidence checklist

1. Unit or weighted graph and direction assumptions.
2. Rooted distance objective or root-free total-tree objective.
3. BFS, Dijkstra/SPT, MST, bottleneck, or diameter claim.
4. Tie handling and uniqueness conditions.
5. Replayable graph edges and exact parent semantics.
6. Distance-label certificate versus cut/cycle certificate.
7. Static or dynamic edge/weight model.
8. Tree construction work versus downstream query quality.

## Sources

- J. B. Kruskal, [*On the Shortest Spanning Subtree of a Graph and the
  Traveling Salesman
  Problem*](https://doi.org/10.1090/S0002-9939-1956-0078686-7), Proceedings of
  the American Mathematical Society 7(1) (1956), 48-50. Minimum-total-weight
  spanning-tree construction.
- E. W. Dijkstra, [*A Note on Two Problems in Connexion with
  Graphs*](https://doi.org/10.1007/BF01386390), Numerische Mathematik 1 (1959),
  269-271. Explicitly separates minimum-length spanning tree and minimum-length
  path problems.
- Notes 01, 03, 11, 21, 28, 41, 47, 81, 82, 83, 93, 104, and 106 provide BFS
  foundations, levels, shortest-path trees, certificates, identity, distance
  validation, parallel work, tree stretch, cycles, cuts, Cayley metrics,
  critical paths, and connectivity context.

## Takeaway

A BFS tree minimizes every root-to-vertex hop count; an MST minimizes one
global sum of selected edge weights. Under unit weights every spanning tree is
an MST, so MST optimality says nothing about BFS depth. Under unequal weights,
MST and SPT can reject each other's objective even on a triangle. The tree's
certificate and intended downstream queries determine which object is useful.
