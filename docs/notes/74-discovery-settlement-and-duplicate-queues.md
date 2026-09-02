# Discovery, settlement, and duplicate-tolerant BFS queues

“Visited” can mean that a state was first generated, successfully claimed,
published into the work queue, or finally popped and expanded. Those events
coincide in a simple textbook presentation but separate in real sequential,
parallel, distributed, and recoverable executions.

This note compares two exact sequential FIFO contracts. It does not select or
optimize an implementation.

## Contract A: claim before enqueue

For each generated child:

```text
if child is not visited:
    atomically claim child
    enqueue child once
```

In a sequential implementation the test and mark are one logical operation.
The queue contains at most one record per semantic state. Note 73's occupancy
formula therefore uses unique next-frontier prefixes.

In a parallel or distributed implementation, “atomically claim” identifies the
winner but is not by itself the entire publication protocol. If a worker marks
visited and fails before making the work record recoverably visible, the state
can become a claimed orphan that no worker expands. Note 30 treats this
claim-versus-publication failure boundary.

## Contract B: settle on dequeue

A duplicate-tolerant sequential version can instead enqueue every child whose
settled bit is still false:

```text
pop (v,d)
if v is already settled:
    discard this stale occurrence
else:
    settle v at distance d
    enqueue eligible neighbor occurrences at distance d+1
```

Several records for the same state may coexist. The first popped record settles
the state; later copies are discarded without expansion.

An additional exact `in_queue` or claimed set would collapse these copies, but
that is logically a form of claim-before-enqueue even if the final distance
array is written later.

## Why strict FIFO still gives shortest distances

Every enqueued record has candidate distance one more than its expanded parent.
Strict FIFO expands settled parents in nondecreasing distance. Therefore no
candidate reached by a longer path can be popped before all candidates reached
from shallower parents. The first popped occurrence of each state has minimum
hop length and may safely settle its distance.

This proof relies on:

- unit edge costs;
- one global FIFO order or an equivalent nondecreasing-distance discipline;
- stale duplicate copies not being expanded;
- complete successor generation for every first-settled state.

It does not justify an arbitrary stack, priority inversion, lossy queue, or
asynchronous irrevocable first arrival.

## A quadratic queue witness

Consider two consecutive BFS layers:

```text
|F_d|=m,
|F_(d+1)|=n,
```

and put every possible simple edge between them. All `m` current parents are
ahead of all next-layer records in FIFO order. Under settle-on-dequeue, none of
the children is settled while those parents are processed. Each parent thus
enqueues all `n` child occurrences.

Immediately after the last current parent, the queue contains

```text
m n
```

records representing only `n` semantic states. For `m=n=w`, queue occupancy is
`w^2` while each exact frontier has width `w`.

Under exact claim-before-enqueue, the first proposal for each child wins and
the corresponding next-layer queue contains `n` records. Both contracts can
compute identical distances and frontier sets; their physical record counts
are different objects.

The witness is quadratic in layer width, not an assertion that every
mark-on-dequeue run is quadratic. Its purpose is to refute a bound based only on
unique frontier cardinality.

## Explicit-graph work can remain `O(V+E)`

Let `R` be the reached vertices of a finite explicit graph and let `A_R` count
the stored outgoing adjacency occurrences scanned from them. With stale-copy
suppression:

- each semantic vertex is first-settled and expanded at most once;
- its adjacency is therefore scanned once;
- each scanned occurrence creates at most one queued record;
- every queued record is popped once, either as the winner or as stale.

Total expansion, enqueue, and pop work is consequently `O(|R|+A_R)`. For a
simple undirected graph stored in both orientations, `A_R` is the relevant
directed adjacency count, approximately twice the reached edge count.

The memory conclusion is different. Total records are bounded by the initial
source records plus `A_R`, so settle-on-dequeue has an
`O(|sources|+A_R)` queue bound rather than the `O(|R|)` unique-record bound. The
complete-bipartite boundary attains `Theta(A_R)` live records at one transition.

Thus quadratic-in-frontier memory and linear-in-explicit-input work are fully
compatible statements. `O(V+E)` does not mean the queue contains only `O(V)`
unique records, and it remains an incomplete physical cost model for implicit
wide-state generation as discussed in note 29.

## Expanding stale copies turns graph search into walk expansion

Consider a directed acyclic layered graph with one root and two vertices in
every later layer. Between each pair of consecutive two-vertex layers, include
all four directed edges; connect the root to both first-layer vertices.

Its unique BFS frontier width is two at every positive depth. Through depth
`D`, the graph has `O(D)` vertices and edges. But if every duplicate queue copy
is expanded again, record multiplicity doubles at every layer:

```text
depth 1: 2 records
depth 2: 4 records
depth 3: 8 records
...
depth D: 2^D records.
```

These records represent distinct directed walks or path prefixes, not distinct
states. First-settlement distance labels may still happen to be correct on this
acyclic fixture, but the execution no longer has ordinary graph-BFS work. If
re-expansion also re-enqueues occurrences without an authoritative
settled-target filter, a cyclic graph can fail to terminate at all without an
external depth bound.

This gives a useful semantic diagnostic:

```text
duplicate records may be queued;
duplicate semantic states must not be expanded as fresh graph states.
```

If the intended object is instead every walk or every labeled path, that is a
different output contract whose multiplicity can be exponential even when the
state graph is small.

## Candidate buffers are often occurrence multisets

A bulk level-synchronous pipeline may first materialize every edge-generated
candidate and only then sort, hash, or claim unique children. Its candidate
buffer behaves like the `m n` occurrence multiset even if its final next
frontier is unique.

Therefore these counters should remain distinct:

```text
generated transition occurrences
candidate records materialized
records enqueued or routed
unique states claimed
first-settled states
stale duplicate pops
stale duplicate expansions (required to be zero for this graph-BFS contract)
accepted next-frontier states.
```

Equality between two of them is a property of a declared schedule, not BFS
semantics.

## Parent and path-output consequences

Distance-only correctness does not settle richer outputs:

- claim-before-enqueue usually gives the first successful proposal one parent;
- settle-on-dequeue gives the first popped occurrence one parent;
- both may discard other same-depth shortest predecessors;
- neither alone constructs the complete shortest-path DAG or exact path count;
- deterministic or shortlex parent semantics need their own completed tie
  reduction, not an unspecified race winner.

Thus two runs may agree on every distance and frontier while selecting different
replayable shortest paths.

## Capacity and failure semantics

Sizing a physical queue to maximum unique frontier width is unsafe for a
duplicate-tolerant occurrence queue. If excess records are silently dropped,
the correctness question becomes whether at least one copy of every required
state survives; aggregate overflow count cannot prove that.

Conversely, reserving a visited claim without guaranteed work publication can
lose the only copy before it reaches any queue. Exact recovery needs a protocol
that couples or reconciles:

```text
claim -> durable/visible work publication -> eventual expansion.
```

Retries must also distinguish the same occurrence from a genuinely different
shortest-parent contribution when the output contract retains multiplicity.

## GPU and multi-GPU interpretation

On a GPU, a visited atomic may collapse proposals before compaction, while a
sort/unique pipeline may materialize them first. Across GPUs, several sources
may route equal children before the authoritative owner settles one identity.
These are locations where multiplicity is removed, not different BFS metrics.

The complete-bipartite witness predicts possible record amplification but no
runtime winner. Atomics, sorting, routing, batching, memory locality, and
contention require measurement under the same semantic contract. The useful
first question is simply: at which event does a candidate become unique and
authoritative?

## Bounded observation

REF-032 validates the two synthetic witnesses in a read-only-mounted Rust
Docker run. On the 100-by-100 complete boundary, claim-before-enqueue used 201
records total and peak queue 199; settle-on-dequeue used 10,101 records, 9,900
stale pops, and peak queue 10,000 while still expanding exactly 201 states.

On the depth-12 two-vertex DAG, stale suppression expanded 25 states and popped
22 stale records, whereas expanding every occurrence produced exact layer
multiplicities `1,2,4,...,4096`. The stale-suppressed peak was six rather than
the initially predicted four because winner expansions appended next-layer
records while stale current-layer copies remained live. These are fixture
observations, not performance results.
