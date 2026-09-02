# BFS trees, fundamental cycles, and Cayley relators

Note 81 studied how a BFS tree distorts non-root distances. The edges omitted
from that tree are not merely discarded adjacency: each one closes a unique
tree path into a fundamental cycle. Together, these cycles describe all binary
cycle parity in a finite undirected graph.

For a labeled Cayley graph they also spell concrete identity words, but the
unlabeled cycle-space statement and the group-relator statement are not the
same object.

## 1. One non-tree edge, one fundamental cycle

Let `G=(V,E)` be a finite connected simple undirected graph and let `T` be a
BFS spanning tree rooted at `s`. For an edge `e={u,v}` not in `T`, the tree has
a unique `u-v` path `P_T(u,v)`. Therefore

```text
C_e = P_T(u,v) union {e}
```

is the unique cycle in `T+e`. Its length is

```text
|C_e| = d_T(u,v)+1
      = d_G(s,u)+d_G(s,v)-2 d_G(s,lca_T(u,v))+1.
```

The original edge has graph distance one between its endpoints, while its
fundamental cycle length records exactly the tree stretch from note 81 plus
that closing edge.

## 2. BFS layers determine fundamental-cycle parity

For every undirected edge `{u,v}`,

```text
|d_G(s,u)-d_G(s,v)| <= 1.
```

Thus a non-tree edge has endpoints either in one layer or in adjacent layers.
Write `a` for the LCA depth.

### Same-layer chord

If both endpoints have depth `d`, then

```text
|C_e| = 2d-2a+1,
```

which is odd. Hence every same-layer edge immediately supplies an odd-cycle
witness, consistent with notes 21 and 31.

### Adjacent-layer non-parent edge

If endpoint depths are `d` and `d+1`, then

```text
|C_e| = 2d+2-2a = 2(d-a+1),
```

which is even. Such an edge can represent an alternative shortest predecessor
without violating bipartiteness.

This parity classification is about an undirected BFS tree. Directed arcs,
self-loops, and parallel labeled edges require separate conventions.

## 3. Fundamental cycles form a binary cycle basis

Represent an edge set by its indicator vector over `F_2`, where addition is
symmetric difference. A connected graph with `n=|V|` and `m=|E|` has exactly

```text
m-(n-1) = m-n+1
```

non-tree edges.

Each fundamental cycle `C_e` contains its own non-tree edge `e` and no other
fundamental cycle contains that edge. The family is therefore linearly
independent over `F_2`.

For any cycle edge set `C`, take the symmetric difference of `C_e` over all
non-tree edges `e` in `C`. Every non-tree edge cancels to match `C`; the
remaining difference lies entirely inside `T`. A tree has no nonempty cycle
edge set, so that remainder is empty. The fundamental cycles span every cycle.

Therefore they form a basis of the cycle space and its dimension is `m-n+1`.
This is an exact structural count, not a statement about how short or useful
the chosen basis cycles are.

## 4. Parent choices change the basis

Changing a valid BFS parent leaves distances and the number `m-n+1` unchanged,
but changes tree paths, LCAs, and fundamental cycles. Therefore it can change:

- individual basis-cycle lengths;
- which original cycles appear directly versus as symmetric differences;
- locality of the edges in a stored cycle witness;
- labeled words obtained in a Cayley interpretation.

A fundamental cycle basis need not be a minimum-total-length cycle basis. A
deterministic parent rule makes one basis reproducible; it does not make it
intrinsically canonical or shortest.

## 5. Cayley edge plus normal forms gives an identity word

Consider a right-action Cayley graph. Let the selected BFS tree word from the
identity to element `g` be `p(g)`. For an oriented non-tree generator edge

```text
u --x--> v,  v=u x,
```

the based closed walk has label

```text
p(u) x p(v)^-1,
```

which evaluates to the identity.

If the two tree words share tree prefix `c`, write `p(u)=c a` and `p(v)=c b`.
The simple fundamental cycle based at their LCA has label

```text
a x b^-1,
```

while the identity-based word is its conjugate

```text
c (a x b^-1) c^-1.
```

Thus parent selection turns every labeled non-tree transition into an explicit
relation witness. Same-layer transitions produce odd-length witnesses;
alternative predecessor transitions produce even-length witnesses under the
undirected symmetric-generator convention.

## 6. Why a cycle basis is not automatically a group presentation

The binary cycle space retains only edge parity. Symmetric difference is
commutative and forgets traversal order, orientation, repeated use, and
generator labels. Group words retain all of those.

Several mismatches follow:

- two generator labels may induce parallel transitions between the same states;
- a generator acting trivially can produce a labeled loop erased by a simple
  graph projection;
- inverse orientation changes a word while leaving the undirected edge set;
- a word relation may traverse an edge more than once and vanish modulo two;
- a fundamental identity word need not be a shortest or defining relator;
- a Schreier-state loop may represent a stabilizer element rather than the
  identity group element.

Consequently, `m-n+1` counts independent binary cycles in the finite simple
graph. It does not count independent group relators, presentation rank, or all
labeled generator-word equalities.

## 7. Bounded BFS sees only locally closed cycles

At depth `r`, a bounded BFS can form a fundamental cycle only when both
endpoints, their selected parent chains, and the closing transition have been
observed under a completed-work contract.

Absence of a non-tree edge in the explored ball does not prove global
acyclicity. Notes 27 and 60 give the sharper girth view: the ball can remain
tree-like until a relation boundary is reached, and the first collision may
appear while generating the next layer or as a same-layer edge in the current
one.

For distributed BFS, a cross-owner non-tree edge is a cycle witness only after
both endpoint identities and parent chains refer to one consistent graph and
checkpoint epoch. Aggregate edge counts are not a replayable cycle
certificate.

## 8. Evidence checklist

For a claimed BFS-derived cycle basis or Cayley relation set, record:

1. simple graph, multigraph, or labeled directed transition model;
2. completed vertex/edge scope and graph version;
3. exact parent tree and tie convention;
4. every non-tree edge with endpoint depths and LCA;
5. fundamental-cycle edge sequence, not only its length;
6. whether algebra is over `F_2` or over ordered generator words;
7. action side, inverse convention, stabilizer, and quotient semantics;
8. whether the claim is a cycle witness, cycle-space basis, identity word, or
   group presentation statement.

## Sources

- R. Diestel, [*Graph Theory*](https://diestel-graph-theory.com/), chapter on
  the cycle and cut spaces. Gives the standard spanning-tree fundamental-cycle
  basis construction.
- D. Peleg and A. A. Schaffer,
  [*Graph Spanners*](https://doi.org/10.1002/jgt.3190130114), Journal of Graph
  Theory 13 (1989), 99-116. Supplies the distance-preserving-subgraph context
  used to contrast tree stretch with retained chords.
- Notes 11, 16, 27, 31, 60, 61, 63, and 81 supply the predecessor, Schreier,
  girth, odd-cycle, relation, stabilizer, same-layer, and tree-stretch contracts
  used here.

## Takeaway

Every edge omitted from a BFS tree closes one fundamental cycle. Same-layer
edges close odd cycles; adjacent-layer non-parent edges close even ones. The
`m-n+1` fundamental cycles form an exact binary cycle-space basis, but their
lengths and shapes depend on parent choices. In a labeled Cayley graph they
yield explicit conjugated identity words, while the unlabeled parity basis is
far too weak to serve automatically as a group presentation.
