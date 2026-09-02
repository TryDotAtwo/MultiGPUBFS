# BFS frontiers, treewidth, and layered treewidth

BFS frontier width and graph treewidth both contain the word "width," but they
measure different objects. A frontier is one root-relative distance layer. A
tree decomposition is an overlapping family of vertex bags arranged by a tree.
Layered treewidth connects the two structures without identifying them.

No decomposition algorithm, dynamic program, or optimized BFS is added here.

## 1. Tree decompositions

A tree decomposition consists of a tree `T` and a bag `X_t` of graph vertices
for every node `t` of `T`, satisfying:

1. every graph vertex occurs in at least one bag;
2. every graph edge has both endpoints together in some bag;
3. the bags containing any fixed graph vertex form a connected subtree of `T`.

Its width is

```text
max_t |X_t| - 1,
```

and graph treewidth is the minimum width over all tree decompositions.

The connected-subtree condition prevents copies of one graph vertex from
disappearing and later reappearing independently along the decomposition. It is
not a BFS visited invariant and bags are neither disjoint nor ordered by
distance.

Trees have treewidth one, while `K_n` has treewidth `n-1`. Large square grids
have treewidth growing with their side length. These examples calibrate
tree-likeness in the decomposition sense, not in the Gromov-hyperbolic or
unique-path sense.

## 2. BFS layering

A layering is a partition

```text
V_0, V_1, ..., V_h
```

such that every edge has endpoints in the same or consecutive layers. Exact BFS
layers from a root form a layering because an edge changes distance from the
root by at most one.

The BFS frontier size at depth `i` is `|V_i|`. Removing a complete intermediate
layer separates strictly earlier layers from strictly later layers, but the
separator may be enormous. This is a root-relative, sequential cross-section of
the graph.

## 3. Small treewidth does not mean a small frontier

Every nontrivial tree has treewidth one. Yet a star on `n` vertices, rooted at
its center, has

```text
|V_0| = 1,
|V_1| = n-1.
```

A complete binary tree of depth `d` also has treewidth one and a last BFS
frontier of size `2^d`. Therefore bounded treewidth gives no bound on BFS peak
frontier, visited memory, or available frontier parallelism.

Changing the root can change the frontier profile without changing treewidth.
Treewidth is root-free; BFS width is not.

## 4. A BFS layer is not a decomposition bag

Taking every BFS layer as one bag generally fails the edge-coverage or
running-intersection requirements unless bags are enlarged and linked carefully.
Conversely, a valid tree-decomposition bag can contain vertices from many
different BFS depths and can overlap many other bags.

The whole frontier `V_i` is a separator between past and future BFS layers. A
tree-decomposition bag often acts as a separator for branches of the
decomposition tree. These are analogous roles with different quantifiers and
different chosen partitions; one is not automatically a certificate for the
other.

## 5. Layered treewidth

Given a layering and a tree decomposition, their layered width is

```text
max over bags X_t and layers V_i of |X_t intersect V_i|.
```

Layered treewidth minimizes this quantity over allowed layerings and tree
decompositions. It bounds how much of one layer any one bag must contain, not
the total number of vertices in that layer and not the bag's total size across
many layers.

The star again separates the quantities. Use one edge bag `{center,leaf}` per
leaf. Each bag meets each of the two BFS layers in at most one vertex, so the
layered width is one, while layer one has `n-1` vertices.

Thus a small layered-width certificate may distribute a huge frontier across
many bags. It does not place the entire frontier into bounded memory.

## 6. Planar and local structure

Every planar graph has layered treewidth at most three. This is compatible with
planar graphs having unbounded ordinary treewidth and arbitrarily large BFS
layers: the theorem controls each bag-layer intersection, not the number of
bags or layer vertices.

Local treewidth asks for a function `f(r)` bounding the treewidth of every
radius-`r` ball in a graph class. Planar graphs and several other proper
minor-closed classes have linear local-treewidth bounds. Again, bounded
treewidth of a ball does not bound its number of vertices: a radius-one star ball
can contain arbitrarily many vertices while retaining treewidth one.

These theorems can support algorithms that explicitly exploit a decomposition.
They do not change the amount of state enumerated by exact BFS merely because
the explored set is a ball.

## 7. Layering choice and operational roots

Layered treewidth is an existential minimum over a compatible layering and
decomposition. An operational BFS uses a particular source and therefore a
particular distance layering. A small layered-treewidth theorem does not by
itself prove that an arbitrary requested BFS root attains the same certificate.

Even when a theorem constructs a decomposition from a BFS layering, the
decomposition is additional structure. It must be represented, validated, and
used by an algorithm whose output contract matches the task. Ordinary exact BFS
does not automatically become dynamic programming over bags.

## 8. Cayley and Schreier graphs

Cayley translation makes BFS layer profiles root-independent for a fixed
symmetric generating set, but it does not provide a low-width tree
decomposition. Cycles, hypercubes, complete Cayley graphs, and expander Cayley
families have very different treewidth.

Generator choice changes the edge set and can change both widths. For `Z2^2`,
the two coordinate generators produce `C4`, of treewidth two; adding the diagonal
generator produces `K4`, of treewidth three. The group and vertex count remain
fixed.

In an implicit puzzle graph, even the existence of a useful decomposition is
not the same as possessing it. Constructing or validating bags may require graph
information that BFS generates only incrementally. A Schreier quotient changes
vertices and edges again, so width claims need direct evidence for the actual
action graph.

## 9. GPU and multi-GPU boundary

Tree-decomposition dynamic programs and frontier BFS have different dataflow:

- a bag algorithm communicates along the decomposition tree and carries a state
  table indexed by bag boundaries;
- BFS expands a distance layer, deduplicates successors, and updates visited;
- layered width bounds per-bag/per-layer overlap, not total frontier records;
- partitioning a frontier by decomposition bags may replicate vertices because
  bags overlap;
- an implicit graph may not have the decomposition materialized at all.

Therefore treewidth, layered treewidth, frontier width, separator size, and
owner-partition load must be reported separately. A low-width theorem is not a
GPU throughput prediction, and a fast BFS kernel is not evidence of a useful
tree decomposition.

## Sources

- N. Robertson and P. D. Seymour,
  [*Graph Minors X: Obstructions to Tree-Decomposition*](https://doi.org/10.1016/0095-8956(91)90061-N),
  Journal of Combinatorial Theory B 52, 1991. Treewidth and decomposition
  structure.
- V. Dujmovic, P. Morin, and D. R. Wood,
  [*Layered Separators in Minor-Closed Graph Classes with Applications*](https://arxiv.org/abs/1306.1595),
  Journal of Combinatorial Theory B 127, 2017. Layered treewidth and the planar
  bound.
- D. Eppstein,
  [*Diameter and Treewidth in Minor-Closed Graph Families*](https://doi.org/10.1007/s004530010020),
  Algorithmica 27, 2000. Diameter-treewidth and bounded local-treewidth
  relationships.

