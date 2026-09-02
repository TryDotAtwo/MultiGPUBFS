# BFS trees, shortest gateways, and dominators

A BFS parent tree records one chosen shortest path to each reached vertex.
Dominance asks a stronger, path-independent question: which vertices are
unavoidable on every route from the source? The two structures can agree, but
one must not be used as evidence for the other.

This note studies the semantic distinction. It does not implement or optimize
a dominator algorithm.

## 1. Flow graph and dominance

Let `G=(V,E,s)` be a directed flow graph: every vertex under consideration is
reachable from the distinguished start `s`. A vertex `d` **dominates** `v` when
every directed path from `s` to `v` contains `d`.

Under the usual reflexive convention, `s` dominates every reachable vertex and
every vertex dominates itself. A strict dominator differs from its target. For
each `v != s`, the unique strict dominator closest to `v` in the dominance
order is its **immediate dominator**, `idom(v)`. The edges

```text
idom(v) -> v
```

form the dominator tree. This is a logical tree: its edges need not be graph
edges and need not connect adjacent BFS layers.

## 2. What a BFS tree can prove

If `d` dominates `v`, then every shortest `s`-to-`v` path contains `d`.
Consequently, `d` is an ancestor of `v` in every valid BFS parent tree.

The converse is false. In the diamond

```text
s -> a -> v
 \-> b -> v
```

a BFS run may choose `parent(v)=a`. Then `a` is an ancestor of `v` in that BFS
tree, but `s->b->v` avoids `a`, so `a` does not dominate `v`. Parent tie-breaking
creates ancestry evidence only for the selected witness path.

Nor is an immediate dominator necessarily a predecessor. In

```text
s -> d -> a -> v
          \      ^
           -> b -|
```

with arcs `d->a`, `d->b`, `a->v`, and `b->v`, the immediate dominator of `v`
is `d`, even though neither `d->v` nor a one-layer dominator-tree step exists.

## 3. Shortest-path gateway is weaker

Consider

```text
s -> a -> v
s -> b -> c -> v.
```

The unique shortest path uses `a`, so `a` is mandatory in the shortest-path
DAG. The longer path through `b,c` avoids `a`; therefore `a` does not dominate
`v` in the original graph.

It is useful to name the two contracts separately:

- a **shortest gateway** lies on every shortest `s`-to-`v` path;
- a **dominator** lies on every `s`-to-`v` path, regardless of length.

Computing the complete shortest-path DAG still answers only the first question.
Same-layer, forward-skipping, and back/cyclic arcs omitted from that DAG may
provide a longer bypass.

### A partial frontier may preserve distance without being a separator

Use

```text
s -> a -> t
s -> b -> c -> t.
```

The depth-one frontier is `{a,b}` and `dist(s,t)=2`. Keeping only `{b}` still
leaves `t` reachable, but its offset continuation gives

```text
1 + dist(b,t) = 3,
```

so reachability from a retained frontier subset is too weak to preserve the
distance. Keeping only `{a}` does preserve the scalar distance because one
shortest path crosses `a`. Yet `{a}` is not an `s`-to-`t` separator or a
dominator set: the longer path through `b,c` avoids it.

For a subset `P subseteq F_d` and target `t` outside `B_d`, scalar distance is
preserved exactly when

```text
min_(p in P) dist(p,t) = dist(s,t)-d,
```

equivalently, at least one shortest `s`-to-`t` path crosses `P` at depth `d`.
This is weaker than intersecting every shortest path, which is in turn weaker
than intersecting every path. The requested output decides which condition is
actually necessary.

## 4. Dominator-set fixed point

The classical data-flow equations are

```text
Dom(s) = {s}
Dom(v) = {v} union intersection over (u,v) in E of Dom(u),  v != s.
```

For cyclic graphs these are fixed-point equations. A single forward BFS-layer
pass is insufficient because the intersection uses **all reachable graph
predecessors**, not just predecessors at distance `dist(v)-1`.

If the same recurrence is restricted to shortest-DAG predecessors, it computes
unavoidability among shortest paths. That is a valid different object, not an
approximation certificate for full dominance.

## 5. Deletion characterization

For a distinct candidate `d` and target `v`, `d` dominates `v` exactly when
deleting `d` and its incident arcs makes `v` unreachable from `s`.

- If a surviving path existed, it would witness non-dominance.
- If `d` does not dominate `v`, an avoiding path survives deletion.

In an undirected graph this is the source-target form of an articulation or
gateway question. It is still root-relative: a vertex can separate one target
from `s` without being a global articulation for every pair.

## 6. A shortest-path counting certificate

Suppose exact forward and reverse shortest distances and exact edge-path counts
are available. Write `sigma_s(y)` for the number of shortest paths from `s` to
`y`, and `sigma_x(v)` for the number from `x` to `v`. Then `x` lies on every
shortest path from `s` to `v` exactly when

```text
dist(s,x) + dist(x,v) = dist(s,v)
```

and

```text
sigma_s(x) * sigma_x(v) = sigma_s(v).
```

The product counts shortest `s`-to-`v` paths passing through `x`, by unique
prefix/suffix decomposition. The certificate requires matching path identity
semantics, exact counts without overflow, and reverse traversal of the declared
graph. It certifies a shortest gateway, not a full dominator.

## 7. Missing edges create false dominators

Dominance is anti-monotone under adding bypass edges: an edge absent from a
snapshot can make a vertex appear unavoidable, while restoring that edge can
destroy the claim. Thus a dominator result requires complete transition
coverage for one graph epoch.

This is sharper than ordinary positive reachability. Missing an edge may leave
`v` reachable and its BFS distance unchanged, yet erase the only observed
bypass and create a false dominator.

A false-positive visited decision at a true dominator can lose the target and
an entire dominated region. Dominance helps describe the possible blast
radius, but the converse is unsafe: dropping a non-dominator can still remove
shortest paths, parents, counts, or other outputs.

## 8. Cayley and Schreier boundary

Vertex transitivity does not by itself remove root-relative dominators. A
one-way directed cycle is a strongly connected directed Cayley graph, yet the
vertices encountered between `s` and `v` in its cyclic order dominate `v`.
Strong connectivity promises a route back, not two independently avoiding
forward routes.

For a Cayley graph, left translation carries a dominance question rooted at
`s` to the corresponding question rooted at the identity. It does not turn
dominance into a distance-only property. For a Schreier graph, quotient, or
history-expanded product state, dominators belong to that declared state graph;
they cannot be imported from a different representation without proof.

## 9. Distributed and GPU interpretation

- A stored BFS parent proves one witness path, not dominance.
- Even a complete distributed shortest-path DAG proves only shortest gateways.
- Missing, stale, or late incoming arcs can create false dominators.
- Owner partitions and dominator subtrees are unrelated unless separately
  established; routing ownership is not graph separation.
- A dominance certificate needs a consistent graph epoch and complete relevant
  transition evidence before cross-owner intersections are trusted.

These are semantic requirements. They do not select a GPU or multi-GPU
algorithm, representation, or communication protocol.

## 10. Evidence checklist

1. Directed graph, undirected graph, Cayley graph, Schreier graph, or product
   state graph.
2. Full-path dominator or shortest-path gateway.
3. Reachable-vertex universe and root.
4. Complete graph predecessors or only previous-layer predecessors.
5. Parent tree, shortest-path DAG, or dominator tree.
6. Exact path-count identity and overflow policy if counts are used.
7. Graph epoch and evidence that bypass arcs are complete.
8. Root-relative separation versus global articulation.

## Sources

- T. Lengauer and R. E. Tarjan,
  [*A Fast Algorithm for Finding Dominators in a Flowgraph*](https://doi.org/10.1145/357062.357071),
  ACM TOPLAS 1(1), 1979, 121-141. Definition, unique immediate dominators,
  dominator tree, and classical algorithmic foundation.
- K. D. Cooper, T. J. Harvey, and K. Kennedy,
  [*A Simple, Fast Dominance Algorithm*](https://hipersoft.cs.rice.edu/grads/publications/dom14.pdf),
  Software Practice and Experience 4, 2001. Dominance as an iterative global
  data-flow problem and the dominator-set intersection formulation.
- Notes 11, 20, 28, 30, 41, 52, 55, 57, and 84 provide BFS-tree,
  product-state, shortest-DAG, path-count, reverse-graph, authoritative-state,
  validation, distributed-snapshot, and strong-connectivity context.

## Takeaway

A BFS tree says which shortest witness was selected. A shortest-path DAG can
say which vertices every shortest witness must use. A dominator tree says which
vertices every path must use. These are three distinct structures. Confusing
them turns tie-breaking into false necessity and makes incomplete incoming-edge
views look like valid separation certificates.
