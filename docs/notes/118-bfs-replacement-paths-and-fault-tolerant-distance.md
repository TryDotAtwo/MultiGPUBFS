# BFS after graph failures: replacement paths and fault-tolerant distance

A failure can mean that the graph changed, or that the machine evaluating the
same graph stopped. These are different mathematical problems. This note studies
the first problem and keeps the execution-fault semantics of note 30 separate.
It adds no implementation, optimizer, benchmark, or GPU code.

## 1. Declare what failed

For `G=(V,E)` and a failure set `F`, the surviving graph must be explicit:

- an edge failure uses `G-F` for `F subset E`;
- a vertex failure deletes the failed vertices and all incident edges;
- a local labeled-edge failure deletes one concrete transition;
- a generator or move-label failure deletes every transition carrying that
  label, with the orientation convention stated;
- a temporal unavailability model changes which edges exist at which time;
- a GPU, rank, or process failure changes the execution of BFS but not `G`.

Calling all six cases "fault-tolerant BFS" hides the object whose distance is
being requested. In particular, loss of a partition owner is not permission to
reinterpret its states as deleted graph vertices.

## 2. Replacement distance and deletion monotonicity

For surviving vertices `s,t`, define

```text
d_F(s,t) = d_(G-F)(s,t).
```

A replacement path is a shortest surviving path realizing that distance. Edge
or vertex deletion cannot create a shorter path, so

```text
d_G(s,t) <= d_(G-F)(s,t),
```

where disconnection is represented by infinity. The original distance is thus
a lower bound after deletion, not generally the new answer.

The classical replacement-paths problem is narrower than arbitrary sensitivity:
fix a shortest `s-t` path `P`, and for every edge `e` of `P`, compute the
shortest `s-t` path in `G-e`. Variants that fail every graph edge, fail vertices,
answer all targets from one source, answer all pairs, or allow several
simultaneous failures are different output contracts.

## 3. Why a BFS tree is not a one-fault certificate

Consider the diamond

```text
    a
   / \
  s   t
   \ /
    b
```

Both `s-a-t` and `s-b-t` have length two. A BFS tree may select `s-a-t` and
retain only `s-b` from the other branch. Deleting `s-a` disconnects `t` in that
tree, although `d_(G-{s-a})(s,t)=2` in the full graph.

Three facts must therefore remain separate:

1. failure of an edge outside a selected tree path leaves that selected path
   usable, although path counts or canonical alternatives may change;
2. failure of a tree edge invalidates the selected witness but may leave an
   equal-length or longer detour in the graph;
3. failure of a bridge genuinely disconnects its two sides.

The selected tree alone cannot distinguish cases 2 and 3. Note 83's complete
fundamental-cut evidence can: a tree edge is a bridge exactly when it is the
only original edge crossing its subtree cut.

## 4. The old shortest-path DAG is also insufficient

The predecessor DAG records all original shortest paths when it is complete.
It can expose an equal-length replacement that avoids a failed item. It need
not contain a longer detour, because vertices and edges off every original
shortest path are absent from that DAG.

Thus:

- surviving path in the old DAG proves an upper bound equal to the old distance
  and hence exactness by deletion monotonicity;
- absence of such a path proves only that no recorded original shortest path
  survived;
- computing a larger replacement distance requires evidence outside the old
  shortest-path DAG.

This is another instance of a recurring BFS lesson: an artifact sufficient for
one output contract is not automatically sufficient for a stronger query.

## 5. Rerunning BFS versus preprocessing failures

For one declared unweighted failure scenario, ordinary BFS on `G-F` gives the
new exact source distances. That is a semantic baseline, not a claim that
recomputation is always the best method.

Preprocessing asks a different question: what retained subgraph or index lets
many future failure scenarios be answered? An exact single-source, single-fault
FT-BFS subgraph `H subset G` requires

```text
d_(H-F)(s,v) = d_(G-F)(s,v)
```

for every allowed single failure `F` and every surviving `v`. Parter and Peleg
show that exactness has a real sparsity cost: there are graphs for which any
single-edge or single-vertex FT-BFS structure needs `Omega(n^(3/2))` edges,
while matching-order constructions exist in the relevant worst-case regime.
The word "tree" in early terminology therefore denotes the BFS service being
preserved, not an ordinary `n-1`-edge tree.

Approximate FT-BFS changes the equality to an explicit stretch inequality. It
must not be reported as exact fault-tolerant distance.

## 6. Small calibration graphs

- **Path:** failure of an internal edge disconnects the two sides. No detour
  exists.
- **Diamond:** failure of one branch edge leaves an equal-length replacement.
- **Cycle:** every single edge is non-bridging, but deleting one can increase
  distances for pairs whose short arc used it.
- **Complete graph:** deleting the direct `s-t` edge changes distance from one
  to two when another vertex exists.

These examples separate connectivity, preserved distance, and increased but
finite distance. "Still reachable" is weaker than "old BFS distance survives."

## 7. Certificates after a failure

A positive path certificate for scenario `F` must:

1. start and end at the declared surviving states;
2. replay through original transitions that remain in `G-F`;
3. avoid every failed edge, vertex, or label under the declared identity rule;
4. have the claimed surviving length.

This proves an upper bound. Exactness additionally needs a lower bound in the
surviving graph. A completed BFS through depth `L-1` in `G-F`, followed by a
length-`L` witness, supplies one. Old labels supply only the weaker bound
`d_G <= d_(G-F)`.

Claiming disconnection is stronger still: failure to find a path in the old
tree or DAG is insufficient. Exact disconnection needs exhaustive reachability
in `G-F` or an independent surviving-cut proof.

## 8. Cayley and Schreier failures

A local edge failure usually destroys Cayley translation symmetry: translating
the missing transition produces other transitions that did not fail. A global
generator-label failure is different. Removing the same generator everywhere
leaves a Cayley or Schreier graph for the surviving alphabet, though it may no
longer be connected.

For example, take `Z_6` with symmetric moves `+/-1` and `+/-2`. If the entire
`+/-1` move family fails, the surviving `+/-2` moves generate `{0,2,4}`. The
graph splits into the even and odd cosets. This is a global algebraic change,
not six unrelated local edge failures.

The failure identity must also say whether a physical reversible move removes
both directed orientations. In a Schreier action, connectivity is governed by
the orbit of the subgroup generated by surviving moves; group-level redundancy
and action-specific stabilizer coincidences must not be conflated.

Note 116 asks whether removed generators have retained-generator words. Here
the same evidence has a failure interpretation: a bounded replacement word
proves a detour for every translated generator edge, while a state-local detour
does not prove a global label-failure guarantee.

## 9. GPU and multi-GPU boundary

Graph failures and compute failures require different recovery evidence:

- deleting an edge, state, or move label changes the mathematical input and the
  correct distances;
- losing a GPU or worker should preserve the same input and reproduce the same
  output through checkpoint, replay, or another consistent recovery protocol;
- losing the rank that owns a state does not make that state unreachable in
  the graph;
- batching many graph-failure scenarios may share preprocessing or execution,
  but every answer remains indexed by its own `F`;
- a recovered distributed run must not mix messages or visited claims from
  different graph/failure epochs.

Report preprocessing, retained structure, per-scenario work, recovery work,
and independent exact validation separately. Throughput across many failure
scenarios is not ordinary single-instance BFS throughput.

## 10. What this changes in the mental model

Ordinary BFS answers a distance question in one fixed graph. Replacement paths
ask a family of nearby distance questions. FT-BFS structures retain enough
redundancy to answer a declared family after failures. Checkpoint/replay keeps
one fixed question intact despite machine failure.

The reusable discipline is:

```text
failure identity -> surviving graph -> requested distance family
                 -> retained evidence -> exact or approximate guarantee
```

Skipping the first two arrows is how an execution fault gets confused with a
graph mutation, or an old BFS tree gets mistaken for a fault-tolerant oracle.

## Sources

- J. Hershberger, S. Suri, and A. Bhosle,
  [*On the Difficulty of Some Shortest Path Problems*](https://doi.org/10.1145/1186810.1186815),
  ACM Transactions on Algorithms 3(1), 2007. Gives the classical
  edge-on-one-shortest-path replacement-paths contract.
- M. Parter and D. Peleg,
  [*Sparse Fault-Tolerant BFS Structures*](https://doi.org/10.1145/2976741),
  ACM Transactions on Algorithms 13(1), 2016. Defines exact single-source
  edge/vertex FT-BFS structures and proves matching sparsity bounds.
- M. Parter and D. Peleg,
  [*Fault Tolerant Approximate BFS Structures*](https://arxiv.org/abs/1406.6169),
  SODA 2014. Makes the exact-versus-approximate guarantee explicit.
- Notes 30, 42, 53, 83, 92, and 116 supply this repository's checkpoint,
  bounded-unknown, shortest-path-DAG, bridge, reachability-preserver, and
  generator-substitution distinctions.

## Takeaway

A replacement path is shortest in a specifically damaged graph. An ordinary
BFS tree certifies the undamaged root distances but usually cannot answer even
one edge failure. Exact FT-BFS preserves all declared post-failure source
distances and may need much more than tree size. Graph mutation and GPU failure
remain separate: one changes the answer, the other must recover the same one.
