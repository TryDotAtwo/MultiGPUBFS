# Directed BFS arc surplus: back depth and arborescence accounting

In an undirected graph, BFS edges stay within one layer or cross one adjacent
boundary, and non-tree edges form a binary cycle-space basis. Directed BFS keeps
only half of that geometry: an arc from depth `d` cannot reach deeper than
`d+1`, but it may point to any earlier layer.

There is still an exact arborescence-surplus identity. Its value must not be
called the number of directed cycles.

No experiment is used. The fixtures are finite graphs checked directly from
their arc lists.

## 1. Reachable directed support graph

Fix source `s` and restrict attention to its finite reachable set `R`. Every
outgoing arc from a reachable vertex also ends in `R`. Let

```text
n = |R|,
m = number of directed support arcs with tail in R.
```

Let `F_0,...,F_D` be exact forward BFS layers and `B_d` their cumulative ball.
For arcs leaving `F_d`, define

```text
P_d = number of arcs from F_d to F_(d+1),
Q_d = number of arcs from F_d to B_d.
```

For every arc `u->v` with `u in F_d`,

```text
dist(s,v) <= d+1.
```

Therefore these classes are complete and disjoint:

```text
m = sum_d P_d + sum_d Q_d.
```

## 2. Directed predecessor excess

For `v in F_(d+1)`, let

```text
p(v) = number of distinct support arcs u->v with u in F_d.
```

Every nonroot reached vertex has at least one such arc, so

```text
P_d = sum_(v in F_(d+1)) p(v).
```

A one-parent BFS arborescence selects one predecessor arc per nonroot state and
contains exactly `n-1` arcs. All other support arcs are:

- `Q_d` arcs to an already reached layer;
- `p(v)-1` unselected equal-depth predecessor arcs into each new state.

Hence

```text
m-(n-1)
  = sum_d Q_d + sum_d(P_d-|F_(d+1)|)
  = sum_d Q_d + sum_(v != s)(p(v)-1).
```

This is the directed support analogue of note 156's radial edge accounting.

## 3. Complete vertex-frontier rejection count

If every directed support arc is scanned once and claim-before-enqueue accepts
each nonroot endpoint once, then exactly `n-1` arc attempts win a frontier
insertion. Therefore

```text
nonaccepting support-arc attempts = m-(n-1).
```

The identity counts support arcs, not labeled occurrences. Parallel generator
labels add the note-157 representation term before this support-level count.

Unlike the undirected formula, a non-tree arc contributes one scan, not a
paired outward/backward scan.

## 4. Back-depth spectrum

Refine `Q_d` by endpoint layer:

```text
Q_(d,k) = number of arcs from F_d to F_k,  0<=k<=d,
Q_d = sum_(k=0)^d Q_(d,k).
```

Define lag

```text
lambda = d-k.
```

Then:

- `lambda=0`: same-layer arc;
- `lambda=1`: arc to the previous layer;
- larger `lambda`: long return toward a shallower BFS region.

This lag is radial relative to one root. It is not a topological-order distance,
cycle length, or proof that the arc lies on a directed cycle.

## 5. Acyclic complete-DAG counterexample

Take vertices `0,...,n-1` and every arc

```text
i -> j whenever i<j,
```

with source `0`. This is a DAG. BFS layers are

```text
F_0={0},
F_1={1,...,n-1}.
```

The `n-1` arcs from zero are tree candidates. Every other arc lies inside
`F_1`, so

```text
Q_1 = (n-1)(n-2)/2,
m-n+1 = (n-1)(n-2)/2.
```

The surplus can be quadratic while the number of directed cycles is zero.
Same-layer directed arcs are not odd-cycle witnesses.

## 6. Diamond versus directed cycle

### Directed diamond

```text
s -> a, s -> b, a -> t, b -> t
```

This DAG has one surplus arc. It appears entirely as predecessor excess:
`p(t)-1=1`, with every `Q_d=0`.

### Directed cycle

```text
0 -> 1 -> ... -> n-1 -> 0
```

Every forward boundary has one accepted arc until the last vertex. The closing
arc contributes one visited-ball term

```text
Q_(n-1,0)=1.
```

Both examples have surplus one. In the diamond it represents alternate
reachability in a DAG; in the cycle it is a genuine return arc. The scalar
surplus does not distinguish them.

## 7. SCC and cycle interpretation

An arc `u->v` lies on a directed cycle exactly when `v` can reach `u`. BFS depth
alone does not answer that return-reachability question.

Consequently:

- a same-layer arc may be acyclic;
- a long back-depth arc may or may not close a directed cycle;
- predecessor excess may occur in a DAG;
- SCC membership needs reverse reachability or an SCC algorithm.

`m-n+1` is best called arc surplus over a spanning arborescence in this note,
not directed cyclomatic number.

## 8. Condensation does not remove radial back arcs

The SCC condensation is a DAG, but a DAG arc can still go from a deep BFS layer
to a shallow one. The head may have a short route from the source while the tail
was reached through a long branch.

Therefore condensation removes directed cycles but need not make every arc go
from depth `d` to `d+1`. `Q_(d,k)` can remain nonzero in an acyclic condensation.

## 9. Output semantics

For distance or one-parent output:

- every `Q_d` arc is already visited;
- one predecessor per new state is retained;
- `p(v)-1` equal-depth predecessor arcs may lose the claim.

For a complete shortest-path DAG, all `p(v)` arcs from `F_d` to `v in
F_(d+1)` are required. `Q_d` arcs are not shortest predecessors because they do
not increase distance by one, though they may be needed for SCC, cycle, or full
graph output.

Thus "nonaccepting for frontier" does not mean semantically irrelevant to every
analysis.

## 10. GPU and multi-owner interpretation

The two surplus terms meet at different boundaries:

- `Q_d` can be rejected by exact visited knowledge because its endpoint is
  already in `B_d`;
- `P_d-|F_(d+1)|` requires convergence among distinct parents of a new state.

Their physical cost depends on whether visited is local, replicated, or owner-
authoritative and whether parent arcs meet before or after routing. A large
`Q_d` is not automatically removable before communication if the producer lacks
authoritative membership.

Lag histograms can also matter: long-return arcs may route to owners associated
with old regions while same-layer arcs can create current-frontier traffic. But
lag alone predicts neither bytes nor locality without an owner map.

## 11. Forward and reverse profiles

Reverse BFS uses transpose arcs and creates its own layers, `P_d`, `Q_(d,k)`,
and predecessor excess. The forward and reverse surplus totals both count all
support arcs minus their respective reachable arborescence sizes only when both
sides reach the same finite vertex set.

Their radial distributions can be entirely different. This is another reason
frontier cardinality alone does not determine bidirectional side work.

## 12. Applicability boundary

The identities assume:

- finite complete traversal of the source-reachable support graph;
- distinct directed support arcs, not raw generator labels;
- one accepted vertex frontier record per nonroot state;
- no early stopping or omitted successor arcs.

Loops are support arcs in `Q_(d,d)`. Parallel labeled occurrences, retries, and
multiarcs need the occurrence identity of note 157. Weighted relaxation and
asynchronous tentative labels need different finalization accounting.

## Sources and internal dependencies

- Notes 05 and 75 define directed and reverse BFS orientation.
- Note 74 gives claim and duplicate queue semantics.
- Note 84 separates SCC, condensation, reachability, and BFS depth.
- Notes 148 and 159 provide directed-random and Schreier direction asymmetry.
- Notes 156-157 supply undirected and occurrence-level conservation laws.
- The arc-surplus identity follows by partitioning every support arc by its BFS
  endpoint layer and selecting one predecessor per nonroot vertex.

## Takeaway

Directed BFS still gives an exact surplus ledger:

```text
arcs beyond one BFS arborescence
= arcs to the visited ball
 + excess predecessors of new states.
```

What disappears is the undirected cycle interpretation. A large surplus can
live entirely inside a DAG, and only return reachability—not BFS layer position—
decides whether an arc belongs to a directed cycle.
