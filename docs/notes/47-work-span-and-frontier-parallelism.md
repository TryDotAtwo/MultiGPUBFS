# Work, span, and frontier parallelism in BFS

Parallel BFS is not characterized by total edge work alone. Its elapsed-time
limits depend on how that work is arranged across causally ordered distance
layers and on how much parallel work each layer exposes.

This note develops a work-span model for understanding GPU and multi-GPU BFS.
It is not an implementation or optimization proposal.

## Work and span answer different questions

For a completed search, let

```text
W = total primitive work
S = span = length of the critical dependency path
P = number of effective parallel workers
T_P = elapsed parallel time in the abstract machine model.
```

Every schedule obeys the two elementary lower bounds

```text
T_P >= W/P
T_P >= S,
```

and therefore

```text
T_P >= max(W/P, S).
```

The ratio

```text
A = W/S
```

is average available parallelism. Even with zero overhead, adding workers far
beyond `A` cannot keep producing proportional speedup on that fixed workload.

These are dependency bounds, not a performance prediction. Real time adds
memory latency, load imbalance, atomics, launches, communication, and capacity
effects.

## A level profile is more informative than one average

For exact level-synchronous BFS, define at depth `d`

```text
w_d = number of frontier states
e_d = transition occurrences expanded from that frontier
q_d = exact identity/dedup work induced by those candidates.
```

A useful logical work profile is

```text
W_d = transition work_d + identity work_d + output/control work_d.
```

Total work is approximately `sum_d W_d`, but instantaneous parallelism changes
with `d`:

- a one-state frontier exposes at most that state's move-level parallelism;
- a wide frontier may expose millions of independent transition occurrences;
- convergence can move the bottleneck from expansion to contested identity;
- a nearly saturated graph can have abundant edge work and almost no accepted
  new states.

Thus two BFS runs with equal `W` and equal number of levels can have different
elapsed times because their work is distributed differently across layers.

The scalar `W/S` hides this profile. Per-level `W_d`, `w_d`, `e_d`, and stage
times show whether parallel resources are underfilled early/late and saturated
only around the middle.

## The logical layer dependency

In ordinary first-discovery BFS, expansion of semantic layer `d+1` depends on
knowing which candidates from layer `d` are genuinely new after exact visited
resolution. Otherwise a longer or duplicate arrival can be expanded as though
it were the shortest discovery.

Under the local successor-oracle/frontier-expansion model, reaching a vertex at
distance `D` therefore contains a causal chain of `D` edge discoveries:

```text
s -> v_1 -> ... -> v_D.
```

Unlimited processors can expand all independent work within a layer, but they
cannot ask the oracle for successors of `v_i` before `v_i` itself has been
generated from the preceding chain.

This is a model-relative lower bound. It is not a theorem that every possible
parallel algorithm on every graph representation has span `Omega(D)`:

- a fully materialized adjacency matrix can use algebraic transitive-closure or
  repeated-squaring techniques that shift work into large global operations;
- preprocessing can store distances or shortcut indexes before the query;
- symbolic representations can apply one operation to many implicit edges;
- graph powers change which logical paths one physical step represents.

Such methods change the primitive access model, preprocessing bill, work, or
output contract. They do not refute the causal statement for on-demand local
frontier expansion.

## One level is not one unit of span

A bulk level may itself require parallel stages such as

```text
expand -> validate -> identify/dedup -> compact -> publish next frontier.
```

If a stage uses a tree reduction, scan, sort network, or distributed collective,
its span can grow with the layer size even though its work is parallel. In a
PRAM-style analysis, one might write

```text
S_total = sum_d S_d,
```

where `S_d` is the critical path through the physical primitives implementing
logical level `d`.

Counting only the number of BFS levels therefore underestimates span. Conversely,
counting every kernel launch as a logical level confuses the algorithm with one
physical decomposition: several kernels can implement one level, or a persistent
kernel can implement several levels.

## Physical barriers versus logical dependencies

A level barrier is one way to enforce that every depth-`d` proposal is resolved
before depth-`d+1` expansion. The logical obligation is ordered finalization;
the physical mechanism may be:

- host-separated launches;
- a device-wide cooperative phase;
- an epoch counter and queue protocol;
- a distributed collective;
- asynchronous label correction with eventual quiescence.

Removing a visible barrier does not erase the dependency. It either moves the
coordination elsewhere or changes the correctness proof to the relaxation model
from note 18, where vertices may reactivate and work may repeat.

Likewise, fusing `k` logical microlevels into one physical superstep can reduce
launch or network-round count while retaining an internal dependency chain of
length `k`. Note 26 distinguishes this from silently traversing a graph power
and changing reported distances.

## Narrow-frontier strong-scaling ceiling

Suppose a level exposes only `e_d` independent transition occurrences before
they converge at visited. Even infinitely many processors cannot usefully assign
more than `O(e_d)` workers to those occurrences without finding finer parallelism
inside move application or identity processing.

For a chain graph,

```text
w_d = 1, e_d <= 2,
```

so BFS has linear work and linear dependency depth but essentially constant
frontier parallelism. A GPU can be mostly idle even though the implementation is
perfectly work-efficient.

For a tree-like middle layer, `e_d` can be enormous, exposing high parallelism
but also candidate/frontier memory pressure. Note 46 shows that bounded-degree
constant expanders must contain a linear-width layer. High parallelism and high
memory demand can therefore be two sides of the same geometry.

## Work efficiency and time efficiency can disagree

Consider two abstract schedules:

- schedule A inspects every stored adjacency once but performs several global
  phases per level;
- schedule B redundantly reinspects or regenerates some transitions while
  reducing coordination or improving locality.

Schedule A is more work-efficient. Schedule B may still finish sooner on a
particular machine. The reverse is also possible if redundant work exhausts
bandwidth or capacity.

Therefore these claims are independent:

```text
less total work
less critical span
less data movement
less elapsed time
better energy efficiency
larger solvable graph.
```

No one of them should silently stand in for the others.

## Candidate work is not useful search progress

At level `d`, generated transitions split into

```text
already-visited hits
same-batch duplicate occurrences
accepted new states.
```

All may supply GPU parallel work, but only accepted states enlarge the exact
ball. Near saturation, a kernel can maintain high transition throughput while
discovery yield approaches zero.

This suggests two complementary parallelism profiles:

```text
physical parallelism supply: generated/identity operations available now
semantic progress: new exact states or completed distance layer per unit time.
```

Optimizing the first metric alone can make a search look busy without moving
the stopping certificate much closer.

## Target stopping changes the critical path

For a target at distance `D`, candidate-stop can avoid processing the remainder
of the target-producing expansion when the level invariant proves no shorter
candidate remains possible. Complete-level output, all shortest parents, or a
full radius-`D` ball requires more work after the first hit.

Consequently the span/work boundary depends on output:

- one target distance/path needs the causal discovery plus a valid lower-bound
  stopping certificate;
- all distance-`D` states need completion of the whole boundary;
- exhaustion needs the final empty frontier after expanding the last layer.

First device-local hit is not automatically the global critical-path endpoint
in a multi-GPU run; smaller-depth work or messages may remain in flight.

## Multi-GPU critical path

With owner-computes visited, one bulk-synchronous level conceptually contains

```text
local expand
-> local pre-dedup/pack
-> route candidate records
-> owner exact dedup/claim
-> global completion/target reduction
-> next-level publication.
```

The level duration is constrained by the slowest relevant owner and the
dependency chain across communication and collective phases, not by aggregate
GPU work divided by GPU count.

Useful per-level quantities include:

```text
max owner work, not only sum owner work
max owner bytes and frontier capacity
critical communication latency
aggregate and peak-link traffic
collective/termination span
overlap that lies off versus on the critical path.
```

Adding GPUs can lower `W/P` while leaving the per-level latency chain nearly
unchanged or even increasing collective cost. This is the dependency form of a
strong-scaling ceiling. Capacity scaling may remain valuable even when latency
speedup saturates.

## Why overlap must be causal, not cosmetic

Two activities overlap in a timeline only if they execute concurrently. They
shorten elapsed time only when the overlapped portion would otherwise lie on the
critical path.

For example, communication of already packed candidates can overlap expansion
of other partitions. But next-level expansion cannot safely consume a state
whose authoritative visited decision has not completed under the level contract.
The final unresolved owner/collective tail can still determine `S_d`.

Thus report both:

```text
sum of component durations
critical-path level duration.
```

Subtracting all visually overlapping intervals from all work can overstate the
benefit when dependencies serialize the tail.

## Counterexamples and rejected shortcuts

### `W/P` predicts GPU time

A chain has `W=Theta(n)` but span `Theta(n)` in the local frontier model, so
unbounded processors do not give constant time.

### BFS has span exactly equal to its number of levels

Each level may contain scans, reductions, routing, or collectives with their own
critical paths. The number of levels is only a logical lower-bound component.

### Removing kernel launches removes the level dependency

A persistent kernel can hide launches while retaining the same ordered
first-discovery phases internally.

### More frontier parallelism is unconditionally better

Wide boundaries improve occupancy but increase frontier, candidate, visited,
and routing pressure.

### More GPUs must reduce latency

Fixed-work strong scaling stops helping when span, skew, communication, or
collectives dominate. More aggregate memory can still enlarge capacity.

### Any `Omega(D)` statement applies to every graph algorithm

It is valid here for on-demand local successor expansion. Global algebraic
access or preprocessing changes the model and can trade much more work/storage
for smaller query depth.

## Sources

- [A Work-Efficient Parallel Breadth-First Search Algorithm](https://doi.org/10.1145/1810479.1810534),
  supplies a structured work-efficient parallel BFS context.
- Duane Merrill, Michael Garland, and Andrew Grimshaw,
  [Scalable GPU Graph Traversal](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf),
  exposes the practical interaction among frontier regimes, work efficiency,
  duplicate handling, and multi-GPU coordination.
- Richard Brent,
  [The Parallel Evaluation of General Arithmetic Expressions](https://doi.org/10.1145/321812.321815),
  is the classical source for relating work, depth, and bounded processors.
- Notes 18, 26, 29, 44, and 46 provide the asynchronous, k-hop, complexity,
  source-transfer, and expansion boundaries used here.

## Current conclusions

1. Parallel BFS time is lower-bounded by both total work per worker and the
   critical dependency span.
2. Frontier geometry creates a time-varying parallelism profile; one average
   cannot describe narrow tails and wide middle layers.
3. In the local implicit-successor model, distance layers impose a causal chain,
   but global representations or preprocessing can change that model by paying
   elsewhere.
4. A physical barrier is not the dependency itself, and removing its visible
   form does not remove ordered finalization or quiescence obligations.
5. Work efficiency, elapsed time, memory capacity, and semantic discovery yield
   are separate performance claims.
6. Multi-GPU strong scaling is limited by the slowest owner and the per-level
   communication/collective critical path, even when aggregate work is balanced.
