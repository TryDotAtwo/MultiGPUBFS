# BFS on growing trees: birth orientation, rerooting, and exact frontiers

When every arriving vertex attaches by exactly one edge to an existing vertex,
the growth graph is a tree (assuming a one-vertex seed). This case separates:

- the parent chosen when the graph was generated;
- the parent chosen by BFS from a later query root;
- the statistical tendency of old vertices to have high degree.

No Docker experiment is needed for the identities below; they are direct tree
proofs. This note also narrows note 151: a hub-rich BFS layer in a tree creates
large unique expansion, not collision-heavy candidate convergence.

## 1. Two roots, two orientations

Let `q` be the seed root. Every later vertex `v` has a unique birth parent
`p_q(v)`, orienting every edge away from `q`.

Choose an arbitrary BFS source `r`. A tree has one simple path between every
pair, so the BFS parent of `v != r` is forced:

```text
p_r(v) = the neighbor of v on the unique v-to-r path.
```

Adjacency order cannot change this parent. The shortest-path DAG and the
one-parent BFS tree coincide because every shortest path is unique.

## 2. Exactly which parent edges reverse

Comparing orientations away from `q` and `r`, every edge on the unique path
`q ... r` reverses and every edge outside that path keeps its direction.

Proof: deleting an edge splits the tree into two components. Its orientation
changes exactly when `q` and `r` lie in different components, which is exactly
when the edge lies on their path. Hence

```text
number of reversed parent edges = dist(q,r).
```

A BFS rooted at a young vertex follows birth-parent arrows only along its path
toward the seed. In other branches it traverses birth edges outward.

## 3. Exact rerooted distances

Root the tree at `q` and let `lca_q(r,v)` be the lowest common ancestor of `r`
and `v`. Unique-path decomposition gives

```text
d_r(v) = d_q(r) + d_q(v) - 2 d_q(lca_q(r,v)).
```

This is an identity for the frozen tree, not a claim that BFS needs an LCA data
structure. It explains why birth depth is not BFS depth from an arbitrary
source.

## 4. Frontier recurrence without collisions

Let `F_d` be the exact BFS frontier from `r`. For every `d>=1`, each vertex in
`F_d` has exactly one neighbor in `F_(d-1)`, no neighbor in `F_d`, and all its
remaining `deg(v)-1` neighbors in `F_(d+1)`.

Two vertices in `F_d` cannot share a child in `F_(d+1)`, because that would
create a cycle. Therefore

```text
|F_1| = deg(r),
|F_(d+1)| = sum_(v in F_d) (deg(v)-1),  d>=1.
```

The candidate taxonomy is exact:

```text
previous-layer occurrences = |F_d|,
same-layer occurrences = 0,
repeated-next-parent occurrences = 0,
unique new states = sum_(v in F_d)(deg(v)-1).
```

There is no visited convergence beyond the inevitable edge back to each
vertex's parent. A large frontier-degree mass becomes equally large
next-frontier membership after those inward edges are removed.

## 5. Correction to the hub-core picture

Note 151 described possible entry into a hub-rich core where scanned candidate
occurrences greatly exceed unique new vertices. The tree identity proves that
this mechanism requires cycles or parallel support paths. It does not occur in
a simple preferential-attachment tree.

For `m=1`, an encountered hub fans out into distinct subtrees, duplicate
pressure is exactly characterized, and high degree affects frontier width
directly. For `m>=2`, multiple attachments introduce cycles, alternate parents,
and candidate convergence. The tree recurrence then becomes a collision-free
upper intuition, not an equality.

Thus `m=1` and `m>=2` differ in the algebra of BFS duplicates, not merely in
average degree.

## 6. Queue and visited consequences

On a known tree with a trusted parent-relative traversal, avoiding the incoming
edge is sufficient to prevent revisits. On an implicit generator presented
only as an undirected neighbor function, the traversal may not know that the
graph is a tree or which edge is incoming until it stores a parent.

Removing `visited` is exact only when the tree property and parent-exclusion
contract are guaranteed. Transferring the same code to a cyclic graph can
re-enqueue ancestors or expand a walk tree. Structural knowledge, not an
observed absence of early duplicates, justifies the omission.

## 7. Ownership interpretation

Birth-contiguous ownership can place the seed and many old high-degree vertices
on one owner. In a tree, this can cause a short, wide burst of unique remote
subtrees rather than repeated proposals to the same child.

Per-level edge work is `sum_(v in F_d) deg(v)`, while accepted next states
subtract exactly one inward edge per nonroot frontier vertex. This makes vertex
balance and incident-edge balance visibly different. Communication depends on
which tree edges cross owners, but cross-owner duplicate merging is absent in
the simple-tree case.

## 8. What this teaches about BFS

The graph may have a stochastic, history-dependent origin while frozen BFS is
purely metric. BFS does not inspect which endpoint is older. Birth time matters
through realized topology or because an implementation uses it for layout and
ownership.

Conversely, the generation tree is the BFS parent tree only when the query root
is the growth seed.

## Sources and internal dependencies

- Note 11 distinguishes a one-parent BFS tree from a shortest-path DAG; on a
  tree they coincide.
- Notes 23, 27, 39, and 74 distinguish a unique-state tree traversal from a
  path/walk tree and explain why visited semantics cannot be removed casually.
- Note 71 shows that tree frontier profiles can be arbitrarily shaped.
- Notes 145, 147, and 151 supply regular-tree, size-biased-degree, and
  preferential-attachment context.
- The process references in note 151 remain the model sources; statements
  specific to this note follow from the declared one-edge growing-tree
  condition.

## Takeaway

Birth parents describe history; BFS parents orient the same tree toward the
query root. Rerooting reverses exactly one path, and every nonparent edge from a
frontier vertex produces a distinct next-layer vertex. This exact case prevents
collision-heavy hub expansion from being treated as a universal PA property.
