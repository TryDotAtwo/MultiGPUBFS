# BFS schedule contracts: layer-setting and label-correcting execution

BFS is defined by shortest-hop distances, not by a particular queue data
structure. But removing or changing the queue is safe only when another proof
replaces the ordering guarantee it supplied.

Three correctness contracts clarify the design space. They can compute the same
distance function on a finite unit-cost graph, while their intermediate states,
stopping rules, parent semantics, and amount of work differ.

No implementation or performance recommendation is made here.

## 1. Contract L: completed layers and irrevocable first claim

At the start of round `d`, the active set is exactly `F_d` and visited is
exactly `B_d`. Every logical successor of `F_d` is processed before any member
of `F_(d+1)` is expanded. A state first claimed from `F_d` therefore has exact
distance `d+1` and its distance never needs correction.

Physical order inside `F_d` is arbitrary. The barrier is not valuable because
it is a barrier; it is one mechanism for proving that no shallower predecessor
work remains. A distributed implementation may realize the same semantic cut
without a host-side barrier, but it still owes equivalent closure evidence.

## 2. Contract K: nondecreasing-key label setting

Maintain tentative distance keys and settle a state only when it is removed at
the globally minimum unsettled key. For unit edges, a strict FIFO queue is a
special compact realization: records appear in nondecreasing hop distance and
contain at most two adjacent levels.

This contract may mix storage for several levels, but cannot settle a larger
key while a smaller reachable key remains eligible. The proof is the
label-setting proof used by unit-weight Dijkstra, rather than an explicit
frontier-array induction.

Duplicate-tolerant FIFO queues fit here when stale copies are discarded and
only the first minimum-distance pop settles the state. Queue membership,
discovery, and settlement are then different events.

## 3. Contract R: arbitrary-order label correction

Allow work to run in arbitrary order. Store tentative labels and apply

```text
D[v] <- min(D[v], D[u] + 1).
```

Every strict decrease must reactivate the affected state's outgoing
propagation. With real-edge witnesses, atomic/serialized minima, eventual
delivery, fair processing, no lost activation, and genuine global quiescence,
the fixed point equals the BFS distance function.

Here discovery does not imply finality. A state may be expanded repeatedly,
and a parent record must be tied to the distance version that won. The final
distances can be schedule-confluent while parents, discovery order, transient
frontiers, messages, and total work remain schedule-dependent.

For a finite graph, once a vertex receives a finite integer witness label, it
can decrease only finitely many times in that execution. This observation does
not replace fairness or termination detection: an unprocessed improvement or
an in-flight message can leave a nonfixed state that merely looks idle.

## 4. The unsafe hybrids

These combinations lack a shortest-distance proof:

- arbitrary-order execution plus irrevocable first claim;
- `atomicMin` updates without reactivating improved descendants;
- local minimum settlement without a valid global lower-bound condition;
- empty resident queues while messages, kernels, retries, or publications are
  still in flight;
- a stale work record allowed to overwrite a better label or parent version.

Atomicity answers which race wins. It does not establish that the winner has
minimum distance.

## 5. The same physical event has different semantic meanings

| Event | Contract L | Contract K | Contract R |
|---|---|---|---|
| first generation | exact next-layer candidate | tentative candidate | tentative upper bound |
| successful first claim | final distance after layer premise | not necessarily settled | not final |
| pop/activation | expand exact layer member | settle only at global minimum | process current version |
| later smaller label | impossible under invariants | impossible after valid settlement | expected and must propagate |
| empty local queue | local round progress only | not global completion | not quiescence |
| target first seen | exact only from a closed shallower ball | upper bound until minimum-key finalization | upper bound |

Calling every one of these events `visited` erases the very distinction needed
for a correctness proof.

## 6. Target finalization differs by contract

Under Contract L, a target generated while expanding exact `F_d` has scalar
distance `d+1`; richer equal-depth outputs can still require layer closure.

Under Contract K, the target becomes final when its key is validly settled at
the global minimum, or when an equivalent lower-bound proof excludes a shorter
unsettled route.

Under Contract R, first discovery supplies only an upper bound. Exact target
finalization requires a global lower-bound certificate over all unfinished
work, or full fair quiescence. A rank-local hit cannot provide this alone.

## 7. One GPU and many GPUs

Device count does not alter the three semantic contracts, but it changes what
must be observed to prove them:

- Contract L needs global completion of the current logical layer and durable
  publication of every accepted next-layer state.
- Contract K needs a globally sound minimum unsettled key despite partitioned
  queues and delayed messages.
- Contract R needs distributed termination detection and accounting for every
  improvement and reactivation in flight.

A fast kernel launch order, stream event, collective return, or local queue
state is physical evidence. It becomes a semantic boundary only when the
protocol connects it to one of these proof obligations.

## 8. Measurements that should not be conflated

For Contract L, useful work is naturally described by completed depths,
successor occurrences, unique accepted states, and duplicate categories.

For Contract K, also record tentative insertions, stale pops, and settled keys.

For Contract R, also record strict decreases, reactivations, version-stale
work, repeated edge expansion, label overestimation, in-flight maxima, and
termination-detection traffic.

Equal final distances do not imply equal work. Conversely, repeated work is not
a correctness defect under Contract R if the fixed-point and termination
obligations are satisfied.

## 9. Current synthesis

The queue is replaceable; its proof obligation is not. Exact BFS distances can
come from closed metric layers, nondecreasing-key settlement, or fair
label-correcting convergence. A design becomes incorrect when it borrows the
cheap finality rule from one contract while using the weaker schedule of
another.

This taxonomy refines notes 03, 12, 18, 57, 74, and 162. It is intended as a
semantic checklist for later bounded one- and multi-GPU probes.

