# BFS trees, fundamental cuts, and bridges

Fundamental cycles from note 82 arise by adding a non-tree edge. The dual
construction removes a tree edge. This exposes a cut separating one parent
subtree from the rest of the graph.

The construction clarifies what parent trees say about bridges, but its cuts
must not be confused with BFS layer boundaries or distributed owner cuts.

## 1. One tree edge, one fundamental cut

Let `G=(V,E)` be a finite connected simple undirected graph and `T` a BFS
spanning tree rooted at `s`. For a tree edge `t={p,c}`, where `p` is the parent
of `c`, deleting `t` splits `T` into

```text
A_t     = the tree subtree rooted at c,
V\A_t   = the component containing s.
```

The **fundamental cut** of `t` is

```text
D_t = delta_G(A_t)
    = {{u,v} in E : exactly one endpoint lies in A_t}.
```

It contains `t`. Every other member is a non-tree edge reconnecting the two
sides that the tree edge alone separates.

The cut is determined by the selected parent tree, not by BFS distances alone.
Changing a tied parent can move a whole descendant subtree and alter many
members of `D_t` while every distance label stays fixed.

## 2. Exact bridge criterion

A tree edge `t` is a bridge of `G` exactly when

```text
D_t = {t}.
```

- If another edge crosses the cut, it reconnects the two tree components after
  `t` is removed, so `t` is not a bridge.
- If no other edge crosses, every path between the sides must use `t`, so its
  removal disconnects `G`.

Equivalently, `t` is a bridge exactly when it belongs to no fundamental cycle.
Every non-tree edge already lies in its own fundamental cycle and therefore
cannot be a bridge.

This criterion requires knowledge of all original edges crossing the subtree
boundary. Parent pointers alone cannot certify that a tree edge is a bridge.

## 3. Fundamental cuts form the binary cut-space basis

Represent edge sets over `F_2`, with symmetric difference as addition. The cut
space consists of all vertex-boundary edge sets `delta(X)`.

There are `n-1` tree edges. Each fundamental cut `D_t` contains its own tree
edge `t` and no other fundamental cut contains `t`: after deleting a different
tree edge, `t` remains wholly on one side. Hence the family is independent.

To see spanning, take any cut `D=delta(X)` and select the fundamental cuts
whose unique tree edges lie in `D`. Their symmetric difference agrees with `D`
on all tree edges. The remaining difference is itself a cut containing no tree
edge. Since the spanning tree connects all vertices, a nonempty proper cut must
cross it. Therefore the remainder is empty.

Thus the fundamental cuts form a basis and, for a connected graph,

```text
dim cut_space(G) = n-1.
```

Unlike the cycle-space dimension `m-n+1`, this dimension does not grow with
extra edges; extra edges change the coordinates of cuts, not the number of
independent vertex bipartitions.

## 4. Cycle-cut parity orthogonality

Every closed cycle crosses from `X` to `V\X` as many times as it crosses back.
Therefore any cycle edge set `C` and cut `D` satisfy

```text
|C intersect D| = 0 mod 2.
```

The binary cycle space and cut space are orthogonal under the edge-vector dot
product over `F_2`.

For a fundamental cycle `C_e` and fundamental cut `D_t`:

- if tree edge `t` is not on `P_T(e)`, their intersection is empty;
- if `t` is on that tree path, both `t` and closing non-tree edge `e` cross the
  cut, so the intersection has two edges.

This provides a local sanity check for recorded tree/non-tree incidence. It
does not validate generator labels or traversal order.

## 5. Three different boundaries

### Parent-subtree cut

`D_t` separates descendants of one selected parent edge from all other
vertices. It depends on parent ties and can cross several BFS levels.

### BFS ball/frontier cut

The boundary of completed ball `B_d(s)` separates distances at most `d` from
larger distances. In an undirected graph its crossing edges join `F_d` to
`F_(d+1)`. It is invariant under parent choice.

### Distributed owner cut

An owner partition separates vertices assigned to different ranks or devices.
It depends on the ownership function, not on tree ancestry or distance. A tree
subtree can be scattered across many owners, and one owner can contain pieces
of many subtrees and layers.

Counts for these three boundaries answer different questions: alternate tree
connectivity, search progress, and communication volume respectively.

## 6. Bounded BFS and false bridge conclusions

Suppose a depth-bounded exploration sees a tree edge but no alternate crossing
edge inside the current ball. An alternate path may leave the ball and return
later. Therefore the edge is only bridge-like in the observed subgraph, not
certified as a bridge of the full graph.

A global bridge claim needs either complete reachable adjacency or another
proof excluding every unseen crossing edge. The `UNKNOWN` discipline from note
42 applies: absence from a bounded table is not global absence.

In distributed BFS, the same applies to delayed or in-flight cross-owner edges.
A fundamental cut is complete only after all transition work relevant to both
sides belongs to one consistent cut of the execution.

## 7. Cayley interpretation and its limits

For a Cayley parent tree, `A_t` consists of elements whose selected normal form
has one particular tree prefix. The fundamental cut lists all generator
transitions that escape or enter that prefix subtree.

This is a property of the selected normal-form tree, not an algebraic coset in
general. A shared word prefix need not define a subgroup, and changing the
shortlex/parent convention changes the subtree.

A non-tree crossing transition supplies an alternate path and hence the
identity-word witness described in note 82. If none exists in the complete
graph, the tree edge is a genuine bridge. Such behavior is possible in
infinite Cayley graphs: a free-group Cayley graph is a tree, so all its edges
are bridges. Finite-puzzle intuition must not be promoted to a universal Cayley
claim.

For Schreier actions, the subtree contains orbit states and a closed lifted
word may be a stabilizer element. Again, a fundamental cut is combinatorial
state-graph data, not automatically a subgroup cut.

## 8. Validation checklist

For BFS-derived cut or bridge evidence, record:

1. complete graph scope, direction, multiplicity, and graph version;
2. exact parent tree and root;
3. subtree membership for each audited tree edge;
4. all original transitions crossing that subtree boundary;
5. whether the result concerns the observed subgraph or the full graph;
6. distinction from level and owner boundaries;
7. cycle-cut parity checks over the same edge identity convention;
8. Cayley labels/action or Schreier stabilizer semantics when applicable.

## Sources

- R. Diestel, [*Graph Theory*](https://diestel-graph-theory.com/), chapter on
  cycle and cut spaces. Provides the standard fundamental-cut basis and
  cycle-cut orthogonality framework.
- Notes 15, 30, 42, 48, 51, 81, and 82 provide the external-memory,
  checkpoint, bounded-unknown, frontier-separator, owner-partition,
  tree-stretch, and fundamental-cycle contracts used here.

## Takeaway

Removing a BFS-tree edge defines a fundamental cut around its selected
descendant subtree. The edge is a bridge exactly when it is the sole crossing
edge. The `n-1` fundamental cuts form the binary cut-space basis and are
orthogonal to every cycle. Parent-subtree cuts, BFS frontier cuts, and
distributed owner cuts are different partitions with different semantics; a
bounded or incomplete traversal cannot silently turn local absence of a
crossing edge into a global bridge certificate.
