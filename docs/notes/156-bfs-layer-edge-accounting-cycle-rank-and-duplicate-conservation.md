# BFS layer-edge accounting: cycle rank and duplicate conservation

Trees, unicyclic graphs, cacti, and theta graphs suggest one general invariant.
In a complete BFS of a finite connected simple undirected graph, every edge is
either inside one layer or crosses one consecutive layer boundary. These two
locations account exactly for every edge outside a BFS spanning tree.

The result turns cycle rank into a conservation law for structural duplicate
work. No experiment is used; the identities are direct counts.

## 1. Radial edge notation

Fix root `s` and exact layers `F_0,...,F_D`. Define

```text
A_d = number of undirected edges with both endpoints in F_d,
B_d = number of undirected edges between F_d and F_(d+1).
```

Set `B_(-1)=B_D=0`. The BFS edge inequality

```text
|dist(s,u)-dist(s,v)| <= 1
```

means there are no other edge classes. Therefore

```text
m = sum_d A_d + sum_(d=0)^(D-1) B_d.
```

For each `v in F_(d+1)`, let

```text
p(v) = number of neighbors of v in F_d.
```

Every reachable nonroot vertex has at least one predecessor, and

```text
B_d = sum_(v in F_(d+1)) p(v).
```

## 2. One BFS tree edge per nonroot vertex

A one-parent BFS tree chooses one of the `p(v)` predecessor edges for every
`v != s`. Hence it contains exactly

```text
sum_d |F_(d+1)| = n-1
```

cross-layer edges. It contains no same-layer edge.

The non-tree edges are therefore exactly:

- all `A_d` same-layer edges;
- `p(v)-1` unselected predecessor edges for every nonroot vertex.

Summing gives the radial cycle-rank identity

```text
beta = m-n+1
     = sum_d A_d + sum_d (B_d-|F_(d+1)|)
     = sum_d A_d + sum_(v != s) (p(v)-1).
```

This count is independent of which equal-depth predecessor wins the parent
race. The distribution across layers depends on the BFS root, but the total
`beta` does not.

### Same-layer does not mean globally geodesically useless

The classification is relative to the chosen root. In the triangle
`s--a--b--s`, BFS from `s` has `F_1={a,b}`, so `a--b` is a same-layer edge.
It cannot belong to a shortest path from `s` to either endpoint: using it after
reaching one endpoint would produce a length-two route to a vertex already at
distance one. Nevertheless, `a--b` is itself the unique length-one shortest
path between `a` and `b`.

Thus a same-layer edge is excluded from the shortest-path DAG rooted at `s`,
not from shortest paths for every possible source-target pair. Changing the
root can also change its radial class: BFS from `a` makes `a--b` a tree-eligible
edge between `F_0` and `F_1`.

## 3. Exact per-level scan conservation

When a complete level `F_d` scans all adjacency lists, its directed edge
occurrences split into:

```text
inward to F_(d-1):       B_(d-1)
within F_d:              2 A_d
outward to F_(d+1):      B_d
```

Thus

```text
sum_(v in F_d) deg(v) = B_(d-1) + 2A_d + B_d.
```

The outward stream contains `|F_(d+1)|` unique states and

```text
B_d-|F_(d+1)| = sum_(v in F_(d+1))(p(v)-1)
```

repeated next-state occurrences beyond one representative per state.

This separates three things often merged under "duplicates":

- reverse scans of already traversed tree/predecessor edges;
- same-layer scans;
- multiple outward proposals for one next-layer vertex.

## 4. Whole-traversal rejection identity

Assume every undirected edge appears in both endpoint adjacency lists and exact
claim-before-enqueue accepts each nonroot vertex once. The complete traversal
scans `2m` directed occurrences and accepts `n-1` frontier insertions. Hence

```text
rejected occurrences = 2m-(n-1)
                     = (n-1) + 2(m-n+1)
                     = (n-1) + 2 beta.
```

The first `n-1` are the inevitable reverse scans of selected BFS-tree edges.
Every non-tree undirected edge contributes exactly two further nonaccepting
occurrences:

- a same-layer edge is rejected once from each endpoint;
- an extra cross-layer predecessor edge gives one duplicate outward proposal
  and one later inward visited scan.

This is a complete-traversal semantic count, not a timing prediction. An
implementation may avoid materializing some occurrences through structural
knowledge, pull traversal, or preprocessing.

## 5. A per-layer cyclomatic charge

Define

```text
q_d = A_d + B_d-|F_(d+1)|.
```

Then `q_d>=0` and

```text
sum_d q_d = beta.
```

`q_d` assigns every non-tree edge to the layer where its non-tree nature first
becomes visible:

- same-layer edge: charged directly in `A_d`;
- extra predecessor of a next-layer vertex: charged in
  `B_d-|F_(d+1)|`.

The sequence `(q_0,...,q_D)` is a root-dependent radial placement of invariant
cycle rank. Two roots or graphs with equal `beta` can expose structural
duplicate pressure at completely different depths.

## 6. Previous exact families reappear

### Tree

All `A_d=0` and `B_d=|F_(d+1)|`, so every `q_d=0` and `beta=0`.

### Odd unicyclic graph

There is one same-layer edge: one `A_d=1`, all predecessor excesses zero, and
`beta=1`.

### Even unicyclic graph

All `A_d=0`; the antipode has `p(v)=2`, so one boundary has predecessor excess
one and `beta=1`.

### Cactus

Every odd cycle contributes one same-layer unit and every even cycle one
predecessor-excess unit. Their sum equals the number of cycle blocks.

### Theta graph

`Theta(3,3,3)` has `p(y)=3`, so `p(y)-1=2`, exactly its cycle rank. In
`Theta(2,3,3)`, two same-layer edges carry the same total rank instead.

These are not separate coincidences; they are instances of one conservation
law.

## 7. Bipartiteness and shortest predecessors

An undirected graph is bipartite exactly when every BFS layer has `A_d=0` for
one, equivalently every, component root. In that case all cycle rank is exposed
as excess predecessor edges:

```text
beta = sum_(v != s)(p(v)-1).
```

This does not mean every vertex has multiple shortest paths. The excess may be
concentrated in a few vertices, and path-count multiplicity can propagate far
beyond those local meetings.

In a nonbipartite graph, `sum A_d` records same-layer odd-cycle witnesses for
the chosen root, but it does not count simple odd cycles.

## 8. Output contract boundary

For distance or one-parent output, each `p(v)-1` predecessor proposal may lose
the frontier claim. For the shortest-path DAG, those same edges are required
output, not disposable noise. For path counts,

```text
sigma(v) = sum_(u in Pred(v)) sigma(u),
```

so an edge's value depends on upstream multiplicity, not merely on its unit
contribution to cycle rank.

The cycle-rank identity counts structural surplus edges. It does not count the
number of shortest paths, which can be exponentially larger.

## 9. GPU and multi-owner interpretation

At level `d`, `B_d-|F_(d+1)|` is the exact amount of outward candidate
multiplicity that a perfect global deduplication could collapse beyond one
record per next state. It does not say where duplicates co-locate physically.

Useful additional questions are:

- are the `p(v)` proposals in one warp, block, GPU, or owner?
- are same-layer edges scanned before the layer is globally stable?
- does one owner receive all predecessor contributions required by the output?
- are retries distinguishable from distinct graph edges?

`q_d` is semantic duplicate pressure, not achievable speedup. Routing,
ordering, representation, and synchronization decide whether the structure can
be exploited.

## 10. Applicability boundary

The formulas above assume:

- finite connected simple undirected graph;
- complete exact BFS from one root;
- every undirected edge represented once per endpoint for scan counts;
- no early target stop;
- vertex identity, not labeled-edge history, defines frontier uniqueness.

Directed arcs can jump arbitrarily between forward-distance layers when viewed
from their head, and self-loops or parallel labeled edges alter occurrence
counts. Implicit Cayley generators may produce the same neighbor through
different labels. Those cases need an occurrence-aware generalization rather
than silent use of `m-n+1` for a simple support graph.

## Sources and internal dependencies

- Notes 10 and 74 define frontier/candidate/visited accounting.
- Notes 11 and 57 separate tree, DAG, count, and path outputs.
- Notes 31 and 82 provide the BFS edge inequality, same-layer parity, and
  fundamental cycle basis.
- Notes 152-155 supply the exact tree, unicyclic, cactus, and theta fixtures.
- The conservation identities are proved by partitioning the complete edge set
  into BFS layer classes.

## Takeaway

Cycle rank is visible directly in a BFS trace:

```text
cycle rank = same-layer edges + excess shortest-predecessor edges.
```

This identity explains where the first duplicates come from and why equal
cycle rank does not imply equal runtime shape: the invariant total can be
placed at different depths and split between fundamentally different kinds of
work.
