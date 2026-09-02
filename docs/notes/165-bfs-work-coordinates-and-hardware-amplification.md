# BFS work coordinates and hardware amplification

The phrase “BFS did `N` work” is incomplete. A traversal moves through several
different objects: frontier states, logical successor occurrences, support
arcs, unique candidate states, accepted states, physical records, memory
transactions, owner messages, and synchronization cuts. They are related, but
they are not interchangeable units.

This note builds a vocabulary for later one- and multi-GPU measurements. It is
an accounting framework, not an optimizer or implementation proposal.

## 1. Semantic coordinates at one completed level

For exact level `d`, retain at least:

```text
f_d = |F_d|                         frontier states
g_d = logical successor occurrences
p_d = distinct nonloop support arcs
c_d = support arcs to F_(d+1)
n_d = |F_(d+1)|                    accepted new states
```

For the stabilizer-aware waterfall of note 160,

```text
g_d = L_d + R_d + V_d + D_d + n_d,
p_d = V_d + D_d + n_d,
c_d = D_d + n_d.
```

Hence

```text
0 <= n_d <= c_d <= p_d <= g_d.
```

These are exact semantic counts under their declared occurrence and support
contracts. They say nothing by themselves about bytes, instructions, atomics,
or elapsed time.

In a free regular Cayley graph with generator set `S`, `g_d=|S|f_d`. In an
explicit irregular graph, `g_d` is the sum of expanded outdegrees. Equal
frontier widths therefore need not imply equal logical expansion work.

## 2. Physical execution coordinates

An implementation may additionally perform:

```text
a_d  physical expansion attempts
w_d  candidate-record writes
q_d  identity/visited probes
t_d  atomic or serialized claim attempts
r_d  routed records
m_d  transport messages
b_d  bytes by field, memory tier, and direction
s_d  synchronization or dependency events
```

There is no universal equality between these quantities and the semantic
counts. Examples:

- one logical occurrence can be recomputed after retry or reactivation;
- a fused pipeline may generate an occurrence without writing a candidate;
- one candidate can require several hash probes or memory transactions;
- several candidates can share one block reservation or message;
- one state record can be split into key, payload, parent, and routing fields;
- a bitmap operation may touch one word while contending with many claims.

Thus `g_d` is neither “number of GPU threads” nor “number of bytes” nor
“number of atomics.” Those are properties of a particular realization.

## 3. Useful amplification and yield ratios

Ratios become meaningful only after numerator and denominator are named:

```text
semantic acceptance yield       n_d / g_d
support yield                    p_d / g_d
new-endpoint convergence         c_d / n_d       (when n_d>0)
execution amplification          a_d / g_d
record materialization           w_d / g_d
probe amplification              q_d / g_d or q_d / w_d
claim contention                 t_d / n_d
routing records per new state    r_d / n_d
bytes per logical occurrence     b_d / g_d
bytes per accepted state         b_d / n_d
```

The first three describe graph/action geometry under a fixed output contract.
The others describe a concrete execution. A high duplicate ratio predicts only
semantic convergence; it does not prove that duplicates reached the same warp,
block, batch, GPU, or owner soon enough to save physical work.

Ratios with `n_d` in the denominator diverge at an empty terminal frontier.
Terminal levels should report their absolute work and `n_d=0`, not an invented
finite “cost per accepted state.”

## 4. Conservation versus conversion

Semantic waterfall equations are conservation identities. Moving from a
semantic count to hardware work is a conversion model:

```text
logical objects
  --representation/layout/protocol-->
records, probes, transactions, bytes, messages, time.
```

Conversion factors can vary by level even when the graph is regular. Cache
state, frontier ordering, key distribution, contention, batch boundaries,
owner skew, compression ratio, and output metadata all intervene.

Therefore a fixed estimate such as

```text
time = generated_edges / peak_edges_per_second
```

is not a BFS law. At most it is a measured model within a specified regime.

## 5. Sum work, critical work, and dependency depth

Total work and elapsed time answer different questions. For owners `j`, record

```text
sum_j work_(d,j),
max_j work_(d,j),
sum_j bytes_(d,j),
max_j bytes_(d,j).
```

The sums describe resource consumption. The maxima are closer to the
bulk-synchronous critical owner, but still omit overlap and dependency order.
The number of completed BFS levels is semantic dependency depth for a strict
level schedule; kernel launches, collectives, and messages are physical events
that may be fused, overlapped, or multiplied without changing that depth.

Consequently:

- balanced frontier-state counts do not prove balanced generator or edge work;
- balanced logical occurrences do not prove balanced bytes or contention;
- balanced bytes do not prove balanced critical time;
- fewer synchronization calls do not prove a shorter semantic dependency
  chain.

## 6. Communication has several denominators

For an owner function, distinguish:

```text
remote logical occurrences,
remote distinct support arcs,
remote unique candidate states,
remote accepted states,
routed physical records,
wire payload bytes,
protocol/control/retry bytes.
```

Early local convergence can reduce routed records. Owner-side convergence can
route multiple records for one state. Replication can avoid a query while
adding update traffic. Recomputing a state from a compact move history can
trade wire bytes for compute. None of these transformations changes which
global state identities must be resolved authoritatively.

There is no representation-independent rule that every remote logical arc must
send a full state. Nor is there a rule that fewer records means fewer bytes:
the surviving records may carry wider keys, provenance, counts, or replay
metadata.

## 7. One GPU versus many GPUs

The same semantic vector should be reported for one and many GPUs. The physical
vector then explains the difference:

- one GPU exposes memory hierarchy, contention, materialization, and level
  control;
- many GPUs add source/destination owner matrices, routing, skew, replication,
  retry/idempotency, and global completion evidence;
- asynchronous label correction additionally adds reactivation and stale-work
  amplification from note 164.

Worker-count parity of final frontiers checks semantics. Comparing physical
vectors shows where additional work entered. Neither alone establishes a
universal scaling law.

## 8. A minimal measurement record

For each completed level, a future probe should preserve:

1. graph/action version, source, direction, output contract, and state identity;
2. the semantic vector `(f,g,p,c,n)` and waterfall categories where available;
3. the physical vector by stage and record field;
4. per-owner sums, maxima, and source-to-destination matrices;
5. peak resident and intermediate capacities plus overflow outcome;
6. completed-cut evidence and separation of retry/replay/reactivation work;
7. end-to-end level time alongside component timings;
8. exact validation scope from note 163.

This is deliberately a vector, not one score. Collapsing it to TEPS, states/s,
or bytes/s is useful only for a named question and workload.

## 9. Rejected implications

- Equal frontier sizes imply equal BFS work.
- Equal generated-occurrence counts imply equal hardware cost.
- Duplicate ratio predicts saved atomics or bytes.
- Fewer routed records always means fewer wire bytes.
- Perfect aggregate balance implies good level time.
- Fewer barriers means less global coordination.
- Higher TEPS implies more accepted-state progress.
- Final-frontier parity explains scaling behavior.

Each statement can hold in a measured regime; none follows from BFS semantics
alone.

## 10. Current synthesis

BFS geometry determines semantic volumes. Representation and protocol convert
those volumes into physical records and bytes. Hardware and scheduling convert
the physical dependency graph into time. Keeping these three layers separate
makes performance observations interpretable without pretending to know an
optimal implementation in advance.

This note consolidates and sharpens the cost boundaries in notes 07, 29, 36,
47, 51, 54, 160, 163, and 164.

