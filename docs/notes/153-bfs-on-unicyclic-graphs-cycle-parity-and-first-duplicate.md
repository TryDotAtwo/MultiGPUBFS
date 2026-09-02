# BFS on unicyclic graphs: cycle parity and the first duplicate

A connected simple undirected unicyclic graph has exactly one cycle. It is the
smallest structural step beyond a tree: `m=n` instead of `m=n-1`, or one
non-tree edge instead of none. This makes it an exact laboratory for asking how
the first cycle becomes visible to BFS.

The answer depends on parity. An odd cycle appears as one same-layer edge. An
even cycle appears as one vertex with two shortest predecessors.

No experiment is used below. The results follow from the unique-cycle
decomposition and narrow the general fundamental-cycle treatment in note 82.

## 1. Cycle with trees attached

Every connected simple unicyclic graph consists of a unique cycle `C_l` with a
rooted tree attached to each cycle vertex. Choose a BFS source `s`.

If `s` is not on the cycle, its attached tree has a unique path to one cycle
vertex `c_0`. If `s` is on the cycle, choose `c_0=s`. Let

```text
h = dist(s,c_0).
```

For a cycle vertex `c_j`, numbered in either direction from `c_0`,

```text
dist(s,c_j) = h + min(j, l-j).
```

For a vertex `x` in the tree `T_j` attached at `c_j`, let `t_j(x)` be its tree
depth below `c_j`. If `j!=0`, every path from `s` to `x` must first reach the
cycle at `c_0`, so

```text
dist(s,x) = h + min(j, l-j) + t_j(x).
```

Inside the source attachment tree `T_0`, use the unique tree path instead:

```text
dist(s,x) = dist_T0(s,x)
          = h + t_0(x) - 2 t_0(lca_T0(s,x)),
```

where the LCA uses the tree rooted at `c_0`. When `s=c_0`, this reduces to
`t_0(x)`, and the first formula works for every `j`. When `s` is off-cycle,
forcing a route through `c_0` can overcount. For a triangle with one leaf `s`
attached at `c_0`, setting `x=s` would give `1+1=2` in that incorrect formula;
the tree formula correctly gives `1+1-2=0`.

Thus all ambiguity is localized to the two directions around the unique cycle.
Every attached-tree segment remains a unique path.

## 2. Odd cycle: one same-layer edge

Let `l=2k+1`. The two cycle vertices farthest from `c_0` are `c_k` and
`c_(k+1)`. Both have distance `h+k`, and they are adjacent to each other.

Therefore the unique cycle-closing signature is

```text
one undirected edge inside frontier F_(h+k).
```

Each of those two vertices still has a unique shortest predecessor. The closing
edge does not create a second shortest path from `s`, because traversing it
would add one hop without reducing the preceding distance.

Scanning the full layer sees the same-layer edge twice, once from each endpoint.
Those occurrences are already visited but are not repeated proposals for one
new next-layer vertex.

## 3. Even cycle: one double-parent meeting

Let `l=2k`. There is one antipodal cycle vertex `c_k` at distance `h+k`. Its two
cycle neighbors both lie at distance `h+k-1`.

Hence `c_k` has exactly two shortest predecessors and two shortest paths from
`c_0` around opposite sides of the cycle. At the discovery boundary,

```text
two outward candidate occurrences -> one unique next-frontier vertex.
```

There is no same-layer cycle edge. A one-parent BFS chooses either predecessor;
the shortest-path DAG retains both.

The identity of the winning parent may depend on within-layer scheduling, while
the distance, frontier membership, and existence of two shortest predecessors
do not.

If a nontrivial tree is attached at the antipode, each of its descendants has
only one immediate shortest predecessor but inherits two complete shortest
paths through that predecessor. Thus candidate convergence is localized at one
vertex while shortest-path count multiplicity propagates through an arbitrarily
large subtree. A one-parent BFS tree hides this propagation; the shortest-path
DAG plus path-count recurrence retains it.

## 4. One surplus edge, two BFS manifestations

Any BFS spanning tree of a connected unicyclic graph omits exactly

```text
m-n+1 = 1
```

edge. Note 82 says a same-layer non-tree edge closes an odd fundamental cycle,
whereas an adjacent-layer non-parent edge closes an even one. Here the graph has
only one cycle, so this classification is complete rather than one basis choice
among many:

| cycle parity | omitted edge endpoints | BFS symptom |
|---|---|---|
| odd | same layer | one same-layer adjacency |
| even | adjacent layers | one alternative shortest predecessor |

The selected omitted edge can change in the even case when the antipode's
winning parent changes, but the two-parent meeting cannot disappear.

## 5. Attached trees do not create further convergence

Once BFS enters a tree attached to a cycle vertex, every descendant has a unique
path back to that attachment vertex. Different attached trees cannot meet.

Therefore the cycle contributes the graph's only non-tree duplicate signature:

- odd: the one same-layer edge;
- even: the one double-parent discovery.

In the even case, this means no further immediate-parent convergence. It does
not mean that downstream shortest-path counts return to one.

High degrees inside attached trees can make wide frontiers, but they obey the
collision-free recurrence of note 152. Width and duplicate convergence remain
separate quantities.

## 6. Visited remains semantically necessary

A traversal specialized to a certified tree can exclude only the incoming
parent. Applying that rule unchanged to a unicyclic graph fails:

- on an odd cycle, the two waves traverse the same-layer closing edge;
- on an even cycle, both waves propose the antipode;
- without exact visited or an equivalent cycle-aware rule, later expansion can
  circulate around the cycle.

One cycle is enough to invalidate the generic claim that parent exclusion
replaces visited. The amount of duplicate work is tiny, but the correctness
boundary is qualitative.

## 7. Frontier counts are tree contributions plus one parity event

Away from the cycle meeting, each frontier vertex contributes its nonparent
tree neighbors uniquely. The exact tree recurrence fails only at the unique
cycle signature:

- odd cycle: subtract the two directed same-layer scan occurrences from any
  naive `deg(v)-1` outward interpretation at the terminal cycle layer;
- even cycle: two outward occurrences at depth `h+k-1` collapse to one new
  antipode; when that antipode is expanded at depth `h+k`, both of its cycle
  neighbors are old, so its new tree neighbors number `deg(antipode)-2`, not
  `deg(antipode)-1`.

This is a minimal example of why scanned edges, candidate occurrences, and
unique next states need separate counters even when their difference is only
one structural event.

## 8. Multi-owner interpretation

If the two arcs of the cycle lie on different owners, the waves may meet
remotely:

- odd parity routes or scans a same-layer cross-owner edge;
- even parity can send two proposals for the antipode to its authoritative
  owner.

Owner authority resolves parent selection, but it must not change distance or
frontier membership. A single duplicate is not a throughput concern by itself;
it is a transparent correctness fixture for routing, deduplication, and
complete-level semantics.

## 9. Cayley boundary

A finite connected simple undirected Cayley graph is regular. If it is also
unicyclic, `m=n`, so its average degree is two. Regularity then forces every
vertex to have degree two, and connectedness forces the whole graph to be a
cycle.

Thus there is no finite simple undirected "cycle with irregular trees attached"
Cayley case. The unicyclic Cayley specialization is exactly a cycle graph, where
the odd/even signatures above become global and root-independent by
vertex-transitivity.

Schreier graphs, directed generator sets, loops, or parallel labeled edges need
their own conventions and are not covered by this reduction.

## 10. Relation to Cayley words

On a cycle Cayley graph, the two traversal directions are two word families.
For even length they meet at the antipode with equal-length words; for odd
length the farthest pair is joined by a same-layer generator edge. This is the
simplest parity-controlled example of a relation appearing at a BFS boundary.

In larger Cayley graphs, many translated cycles and interacting relators make
the signature repeat and overlap. The unicyclic graph isolates one relation
without pretending that a realistic Cayley graph has only one global cycle.

## Sources and internal dependencies

- Note 31 gives the same-layer odd-cycle witness.
- Note 74 separates candidate occurrences, claims, and accepted states.
- Note 82 proves the general same-layer/adjacent-layer fundamental-cycle parity
  classification.
- Note 152 gives the collision-free tree recurrence that one cycle minimally
  violates.
- Notes 16, 32, 60, and 67 provide the Cayley action, distance-regular cycle,
  relation-onset, and parity context.

## Takeaway

The first edge beyond a tree has exactly two possible BFS faces. Odd parity
places it inside one frontier; even parity makes two frontier vertices propose
the same next vertex. This tiny graph class exposes why duplicate type, not
just duplicate count, matters to visited semantics and shortest-path output.
