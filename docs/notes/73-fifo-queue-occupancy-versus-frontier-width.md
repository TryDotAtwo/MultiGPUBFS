# FIFO queue occupancy is not frontier width

The exact BFS layers `F_d` are mathematical sets. A sequential FIFO queue is
one schedule that realizes those layers, but its instantaneous contents usually
do not equal one complete frontier. This distinction matters when interpreting
memory measurements.

## Scope

Assume a fixed directed or undirected unit-edge graph, exact identity, and
**mark on first enqueue**: a state enters the queue only once. Neighbors of each
dequeued vertex are processed completely. The result below applies to one or
many initial sources after treating their distinct set as `F_0`.

## The two-layer queue invariant

FIFO BFS dequeues vertices in nondecreasing distance. While it is processing
layer `F_d`, its queue has the form

```text
unprocessed suffix of F_d
followed by
already discovered prefix of F_(d+1).
```

No `F_(d+2)` vertex can yet be enqueued: producing one requires dequeuing an
`F_(d+1)` parent, and FIFO places every discovered `F_(d+1)` vertex behind the
remaining `F_d` suffix. Once that suffix is empty, the queue is exactly the
complete `F_(d+1)` in its discovery order. The same argument then repeats.

Thus the queue contains at most two consecutive distance layers, but it need
not equal either full layer during the transition.

## Exact occupancy formula

Let

```text
m = |F_d|,
n = |F_(d+1)|.
```

After the first `k` parents of `F_d` have been completely expanded, let `D_k`
be the number of distinct `F_(d+1)` vertices discovered from their combined
successors. Then

```text
Q_k = m-k+D_k,
D_0=0,
D_m=n.
```

The first term is the remaining current-layer suffix; the second is the unique
next-layer prefix. This gives

```text
max(m,n) <= peak during/at the layer boundaries <= m+n-1
```

when both layers are nonempty. The lower expression follows because the queue
equals `F_d` at the start and `F_(d+1)` at the end. For `1<=k<=m`, at least one
current parent has been removed, so `Q_k<=m-1+n`.

The upper bound can be attained when the first processed parent discovers every
next-layer vertex. It is not generally equal to `max(m,n)`.

## One graph, two queue peaks

Construct a three-layer tree:

- root `s`;
- `m=100` vertices in `F_1`;
- `n=100` vertices in `F_2`;
- one distinguished `F_1` hub is parent of every `F_2` vertex;
- the other 99 `F_1` vertices have no forward children.

The root adjacency order determines the FIFO order of `F_1` but changes no
distance or frontier set.

If the hub is processed first, its expansion occurs after one current parent
is removed and all 100 next states are appended:

```text
Q_1 = 99+100 = 199.
```

If the hub is processed last, the queue shrinks from 100 to one before the 100
children appear. The peak is then only 100. The graph, visited set, exact
frontiers `[1,100,100]`, and total edge work are identical.

This is a scheduling-memory effect, not a semantic change in BFS.

## Discovery order matters through prefix coverage

The whole occupancy trajectory is governed by `D_k`, not merely by `m`, `n`,
or total boundary incidence. Parents whose child sets cover new endpoints early
inflate the mixed queue; parents whose successors mostly repeat already found
children add little.

Therefore two parent orders with the same aggregate duplicate count can still
have different queue peaks. The relevant object is the prefix union curve

```text
k -> |union of forward-neighbor sets of the first k parents|.
```

This curve is order-dependent even though its endpoint `D_m=n` is canonical.

## When the bound does not apply

### Mark on dequeue

If every unseen-looking occurrence is enqueued and identity is finalized only
when popped, the queue may contain several copies of one semantic state. Its
next-layer part is then an occurrence multiset, not a unique-state prefix, and
the `m+n-1` bound using frontier cardinalities can fail badly.

### Incomplete or pruned traversal

Capacity drops, beam selection, early target stopping, omitted successors, and
probabilistic false positives change `D_k` or prevent `D_m=n`. A small queue is
not evidence of exact BFS unless the semantic gate is established separately.

### Asynchronous relaxation

A reactivating asynchronous shortest-path schedule can hold labels from more
than two tentative depths and can enqueue a state repeatedly. It is not governed
by the strict FIFO two-layer invariant.

### Bulk-synchronous frontiers

A level-synchronous implementation may store complete current and next arrays
simultaneously. Its buffer occupancy is a different quantity from sequential
FIFO queue occupancy even when both compute identical `F_d` sets. Scratch,
candidate, parent, and visited storage remain additional memory categories.

## Interpretation for GPU and multi-GPU study

The theorem does not recommend a queue ordering. It establishes which memory
quantity is being measured:

- semantic frontier width: canonical `|F_d|`;
- FIFO occupancy: order-dependent mixed suffix plus prefix;
- candidate storage: occurrence-dependent and potentially much larger;
- bulk frontier buffers: representation-dependent;
- visited and output state: persistent across layers.

Distributed ownership adds another distinction: each rank may observe a local
mixed queue or local current/next buffers whose peaks do not coincide in time.
The sum of per-rank maxima is not a globally simultaneous memory snapshot, and
the maximum global frontier does not by itself bound every local allocation.

No performance conclusion follows from the combinatorial occupancy bound. It
only prevents `peak queue`, `peak frontier`, and `peak candidates` from being
reported as interchangeable measurements.

