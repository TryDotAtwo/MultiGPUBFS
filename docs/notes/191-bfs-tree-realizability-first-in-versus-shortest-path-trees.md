# BFS-tree realizability: first-in trees versus shortest-path trees

## Question

If every chosen parent edge decreases distance by one, must the resulting
shortest-path tree be producible by some ordinary FIFO BFS run?

No. Parent validity is pointwise; FIFO discovery history couples choices across
several children.

## 1. Three different objects

For a rooted graph, distinguish:

1. the canonical scalar distance map `d(s,v)`;
2. an arbitrary **shortest-path tree**, where every non-root vertex `v` chooses
   some edge `parent(v)->v` with `d(parent(v))=d(v)-1`;
3. a **first-in BFS tree**, where `parent(v)` is the first already dequeued
   predecessor that discovers `v` in one legal FIFO BFS history.

Every first-in BFS tree is a shortest-path tree. The converse fails.

## 2. Smallest coupling pattern

Let `s` reach `u,v`, and let both `u` and `v` reach both `x,y`:

```text
        u ----> x
      /   \  /
     s     XX
      \   /  \
        v ----> y
```

All of `u,v` have depth one and all of `x,y` have depth two. The proposed
parents

```text
parent(x)=u,
parent(y)=v
```

are individually shortest-valid. But whichever of `u,v` is dequeued first
discovers both `x` and `y`; the later vertex cannot become first discoverer of
one child. Reordering the adjacency list of the first parent changes the order
in which it discovers `x,y`, not the fact that it discovers both.

The obstruction is therefore not a wrong distance and not a cycle in the
proposed tree. It is an inconsistent set of precedence demands on the same BFS
layer:

```text
parent(x)=u requires u before v,
parent(y)=v requires v before u.
```

## 3. Why local precedence is not the whole recognition problem

For one fixed layer, a chosen parent must precede every competing predecessor
of the same child. These constraints are necessary. They expose the example's
cycle immediately.

But FIFO order at depth `d` is not freely chosen in isolation. It is itself the
concatenation of discovery blocks emitted while depth `d-1` vertices are
dequeued, with freedom only inside declared neighbor orders. Thus a complete
recognition theorem must ask whether compatible choices exist across all
layers, not merely whether every parent edge is geodesic.

Manber formalized exactly the undirected recognition problem: given `G` and a
spanning tree `T`, decide whether `T` can be the outcome of BFS, and find all
possible roots. The paper gives a linear-time algorithm. Its existence is
evidence that BFS-tree realizability is a structured global property distinct
from shortest-path-tree validation. The full paper was not available in this
pass, so this note does not reconstruct its algorithm from the abstract.

## 4. Post-layer reduction deliberately changes the tree family

A level-synchronous implementation can gather every depth-`d` proposal for a
child and use a deterministic child-dependent preference, for example,

```text
for x: prefer u to v;
for y: prefer v to u.
```

The result is deterministic and every edge decreases distance by one. This
rule chooses the crossed tree above and therefore is not the first-in tree of
any serial FIFO execution.

This example does not refute the fixed lexicographic rule
`min(parent_state_id, move_id)` with a common global parent-ID order. Both
children here have the same candidate-parent set, so that rule selects the same
minimum-ID parent for both. FIFO realizability of another reduction or a larger
graph needs its own argument; determinism alone neither proves nor disproves it.

That is not a semantic defect when the declared output is:

```text
exact distances + one deterministic shortest witness.
```

It is a defect only if the declared output is:

```text
the discovery tree of some ordinary FIFO BFS history.
```

“Reproducible BFS tree” is therefore underspecified until it says whether
reproducibility applies to a shortest-path reduction or to a realizable search
history.

## 5. GPU and multi-GPU consequence

Parallel first arrival is normally a hardware race and does not define a stable
semantic FIFO history. Owner-side post-layer reduction removes that race but
also severs the implication that the selected parents came from any one serial
first-discovery order.

Useful validation fields are consequently separate:

```text
distance validity
parent-edge depth validity
replay validity
deterministic reduction parity
first-in BFS-tree realizability, if actually required.
```

Comparing only with one CPU run is too strong for an arbitrary-shortest-tree
contract and too weak for a canonical contract: another CPU neighbor order may
legitimately choose a different first-in tree, while a stable but incorrectly
defined reduction may repeat the same noncanonical answer everywhere.

## 6. Cayley and labeled-action boundary

In a Cayley or Schreier graph, an incoming parent occurrence also has a
generator label. A generator order can choose a shortlex word or another
canonical witness, but a state-level post-layer minimum and a serial
generator-ordered first-discovery history are still different reductions.

Relations and stabilizer aliases increase the number of competing occurrences.
Several labels can realize one support edge, and several parents can reach one
state. A tree of semantic states discards that labeled multiplicity; replay
requires retaining a concrete move occurrence even when the parent vertex is
already fixed.

## 7. Exact Cayley counterexample in `S_3`

The obstruction is not an artifact of an irregular hand-built graph. Take the
right Cayley graph of `S_3` generated by all three transpositions:

```text
S = {(12),(13),(23)}.
```

The cycle-count metric from note 138 gives:

```text
F_0 = {e},
F_1 = the three transpositions,
F_2 = {(123),(132)}.
```

Multiplying any transposition by either of the other two transpositions yields
one of the two 3-cycles, and the two choices yield different orientations.
Hence the support graph between `F_1` and `F_2` is exactly `K_(3,2)`: every
depth-one state is a shortest predecessor of both depth-two states.

Choose two different transpositions `u,v` and propose

```text
parent((123)) = u,
parent((132)) = v.
```

Both edges are geodesic. Nevertheless the first of `u,v`—indeed the first of
all three `F_1` states—to be dequeued discovers **both** 3-cycles. Therefore
every serial first-in FIFO tree must give the two `F_2` states the same parent.
An arbitrary or child-dependent deterministic post-layer rule may validly give
them different parents. A fixed `min(parent_state_id, move_id)` rule cannot do
so in this example: both children have all three transpositions as candidates
and select the same minimum-ID parent.

This example adds two useful intuitions:

- vertex transitivity and a conjugacy-invariant generator set do not make
  shortest-parent choices independently serializable;
- the same relations that create shortest-word multiplicity also create
  coupled first-arrival constraints across states.

No enumeration is needed: the six group elements and the product of two
distinct transpositions prove the whole layer incidence exactly.

## 8. Rejected implications

- Every shortest-path tree is a BFS discovery tree.
- Valid parent depth plus replay proves FIFO realizability.
- Sorting adjacency lists can realize arbitrary independent parent choices.
- A deterministic GPU parent tree represents some deterministic serial BFS.
- Failure to match one CPU parent tree means distances are wrong.
- Matching one CPU parent tree proves a canonical or first-in-tree contract.

## 9. Evidence boundary

The crossed-parent graph and the all-transposition `S_3` Cayley graph are exact
conceptual counterexamples. Manber's publisher abstract confirms the formal
recognition problem and linear-time result, but the full recognition
construction was not inspected. No code, enumeration, or runtime claim is made
here.

## Sources

- Udi Manber, [*Recognizing Breadth-First Search Trees in Linear
  Time*](https://www.sciencedirect.com/science/article/pii/002001909090155Q),
  *Information Processing Letters* 34(4), 167--171, 1990,
  DOI 10.1016/0020-0190(90)90155-Q. Publisher abstract inspected; full text not
  obtained in this pass.
- Robert Scheffler, [*On the Recognition of Search Trees Generated by BFS and
  DFS*](https://doi.org/10.1016/j.tcs.2022.09.018), *Theoretical Computer
  Science* 936, 116--128, 2022. Open-access publisher abstract supplies the
  modern “first-in tree” terminology and historical placement of BFS-tree
  recognition.
- Notes 19, 57, 138, and 175 supply the existing crossed-parent counterexample,
  output-contract lattice, exact all-transposition `S_n` metric, and shortlex
  distributed reduction used here.

## Compact conclusion

Distances constrain each parent edge locally, but a serial BFS discovery tree
also records a globally compatible first-arrival order. Parallel post-layer
parent reduction may preserve exact distances and deterministic replay while
producing a tree outside the family of serial first-in BFS trees. These are
different valid contracts and must be tested separately.
