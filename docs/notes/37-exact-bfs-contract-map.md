# Exact BFS contract map: what must agree before the name is earned

Exact BFS is not identified by a queue, a kernel, a visited table, or a frontier
count in isolation. It is a stack of contracts connecting the requested graph
problem to a complete mathematical result and then to its physical execution.
Breaking any one layer can leave the others looking plausible.

This note synthesizes the preceding study. It introduces no implementation.

## The core mathematical statement

For a fixed directed unweighted graph `G=(V,E)` and source set `S`, define

```text
B_d = {v | dist(S,v) <= d}
F_d = B_d minus B_(d-1).
```

The exact level recurrence is

```text
F_(d+1) = unique(Post(F_d)) minus B_d
B_(d+1) = B_d union F_(d+1).
```

Everything called an implementation detail must still realize the same sets,
or a separately declared variant. The contract map answers what the symbols
`V`, `E`, `dist`, `unique`, `Post`, `S`, and completion mean in the actual
workload.

## Nine independent contract layers

| Layer | Must declare | Typical silent substitution |
|---|---|---|
| problem | source, target, requested output | one path substituted for all paths |
| graph | vertex identity, edges, direction, labels, version | quotient or symmetrized graph substituted for original |
| metric | unit step and logical depth | incidence half-step, k-hop macro-edge, or weight treated as one hop |
| identity | exact equality/canonical key | hash/fingerprint equality or unintended orbit equality |
| frontier transaction | complete expansion, dedup, old-ball subtraction | partial/top-k/capacity-truncated next layer |
| schedule | level barrier or relaxation/fairness rule | first asynchronous claim frozen before shorter proposal |
| output metadata | parents, ties, labels, counts, determinism | reached set assumed to preserve shortlex/counts |
| completion | target lower bound or global quiescence | local empty queue or first meet treated as proof |
| physical evidence | capacity, versions, routing, replay, measurements | fast kernel or matching count treated as end-to-end correctness |

The layers constrain one another but are not interchangeable. A valid parent
chain is an upper-bound witness, not by itself a proof that no shorter path was
missed. A correct distance map does not imply a correct path count. A globally
empty frontier under one graph version says nothing about another version.

## 1. Problem and output contract

Before choosing an algorithm, name the requested object:

- distance to one target;
- every distance through a radius;
- the complete reachable component;
- one arbitrary shortest path;
- a deterministic/shortlex path;
- all shortest parents or a predecessor DAG;
- shortest-path counts;
- nearest-source labels and ties;
- component, bipartiteness, eccentricity, diameter, or girth certificate.

These outputs have different stopping boundaries and reduction algebras. The
phrase "find a solution" does not choose among them.

## 2. Graph contract

The search graph is determined by more than a state struct:

```text
semantic vertex identity
successor/predecessor relation
directedness and inverse convention
generator/edge labels and multiplicity
legality and graph version
quotient, product, or symmetry frame
```

Explicit CSR, an implicit successor function, and a Cayley action can present
the same abstract graph, but only if successor completeness and identity agree.
Changing canonicalization, generator order/set, temporal snapshot, or action
side can change the graph even when record bytes remain compatible.

## 3. Metric and logical-depth contract

Ordinary BFS minimizes the number of unit graph edges. Variants must say when
one physical operation is not one logical edge:

- 0-1 BFS includes zero-cost edges;
- weighted SSSP uses relaxation beyond FIFO BFS;
- k-hop batching contains several logical depths in one superstep;
- incidence-graph hyperedge traversal uses two physical edges per logical
  Berge step;
- temporal/product states change what a vertex and a step remember;
- lazy random walks add waiting steps and do not preserve exact-length support.

A counter printed by a loop is not automatically the requested distance.

## 4. Exact identity contract

`unique` means semantic equality under the requested graph:

- an injective dense rank is exact;
- a collision-resolving table can be exact;
- a bare fingerprint or Bloom-positive decision is approximate;
- symmetry canonicalization is exact only for the quotient problem or after a
  distance/path-lifting proof;
- product-state components may not be dropped merely because base states match.

Owner routing and visited equality must use compatible keys and epochs so all
equal states meet at one authoritative decision.

## 5. Frontier transaction contract

A completed next frontier must contain every new semantic state and no old one.
Physical candidates may be bags, partitions, sorted runs, bitmaps, or messages.
The proof obligation remains

```text
exact unique of every successor occurrence
minus the complete old ball
with explicit capacity/overflow status.
```

Queue versus bitmap, push versus pull, local pre-deduplication, external sort,
and GPU compaction can preserve this predicate. Beam top-k, silent truncation,
false-positive visited, or incomplete message delivery cannot.

## 6. Schedule contract

Level-synchronous BFS starts expansion of `F_d` with the exact known ball
`B_d`. A genuinely new child then has final distance `d+1` immediately: its
parent supplies a length-`d+1` witness, and nonmembership in `B_d` excludes a
shorter path. Completing every depth-`d` producer closes the whole next layer
and its requested metadata; it is not needed to finalize each scalar label.

An explicit barrier is one mechanism, not a semantic requirement. Sequential
FIFO preserves nondecreasing expansion depths without a separate barrier.
If an execution abandons that order and allows a longer proposal to arrive
before a shorter one, irreversible first discovery is no longer justified.
An exact corrective alternative needs:

- minimum-distance relaxation;
- reactivation after improvements;
- fair eventual processing;
- output-specific tie completion;
- distributed termination detection.

An asynchronous execution can compute exact distances without preserving the
same parent or processing order. Determinism is a separate reduction.

## 7. Output-metadata contract

The reached set is idempotent set union. Richer outputs differ:

| Output | Required retained/reduced information |
|---|---|
| arbitrary parent | one real depth-decreasing edge |
| canonical parent | every equal-depth contender under a total rule |
| all parents | exact predecessor-edge set |
| path count | one contribution per predecessor edge with overflow semantics |
| labeled path | replayable move and action/frame convention |
| nearest source | complete equal-depth label/tie reduction |

Membership deduplication cannot silently stand in for these reductions.

## 8. Stopping and completion contract

The stopping proof depends on the requested output:

- target BFS: all work capable of producing a shorter target must be excluded;
- bidirectional BFS: best meeting upper bound must meet a completed lower bound;
- exhaustive BFS: globally no pending frontier, candidates, messages, or failed
  owner remains;
- bounded BFS: the declared final radius is complete;
- asynchronous BFS: no improvement/reactivation obligation remains;
- checkpointed BFS: durable state proves the same property after recovery.

Local emptiness, first target observation, first bidirectional intersection,
or a persisted timestamp are observations, not universal termination proofs.

## 9. Physical and evidential contract

The physical run must bind:

```text
graph/state/generator versions
record schema and exact capacity
frontier/visited/parent artifacts
owner count/function/epoch
overflow and failure flags
checkpoint/message state where relevant
hardware, software, command, and workload parameters.
```

Correctness evidence should test membership and paths, not only totals. A
matching frontier count can hide one missing and one spurious state. A matching
target distance can coexist with lost branches. A replayable path can coexist
with an incorrect claim of exhaustive completion.

## Three classes of variation

### Same mathematical result, different evaluation

Under their proof conditions, these can preserve exact `B_d,F_d`:

- FIFO queue versus bulk level sets;
- sparse list versus exact bitmap;
- push versus pull;
- sort/unique versus exact claims;
- CPU, GPU, and owner-partitioned multi-GPU execution;
- internal k-hop batching that still reconstructs every logical layer;
- rollback/replay from a consistent checkpoint.

They may change order, work, parents, locality, or communication unless those
are separately constrained.

### A different but exact mathematical problem

These can be exact while not returning ordinary single-source BFS output:

- multi-source distance to a set;
- 0-1 or weighted shortest paths;
- quotient/orbit distance;
- product-state constrained paths;
- temporal earliest-arrival metrics;
- hypergraph Berge distance;
- directed positive-generator reachability;
- LexBFS ordering.

The error is not using a variant. It is retaining the old name and guarantees
after the problem object changed.

### Incomplete or approximate execution

These require an explicit weakened claim:

- beam/top-k pruning relative to the original graph;
- depth or capacity truncation;
- false-positive approximate visited decisions;
- dropped candidates/messages;
- sampled edge expansion;
- heuristic bidirectional stopping;
- unfinished owner/global reduction.

An approximate method can be useful. It is not exact BFS merely because every
retained state has a valid path.

## A minimal exact-BFS passport

Before accepting a run, fill these fields:

```text
problem/output:
vertex identity:
edge/successor version:
direction, labels, and unit cost:
source/target semantics:
logical level invariant:
exact visited mechanism:
candidate completeness and capacity:
parent/tie/count contract:
ownership/routing epoch:
stopping/quiescence proof:
recovery semantics:
reference oracle/certificate:
work and hardware metrics:
known approximations or omissions:
```

An empty field is not automatically an error, but it identifies an unproved
link in the claim.

## Validation ladder

Validation should rise with claim strength:

1. **transition replay:** every stored edge/move is real;
2. **parent upper bound:** parent chains realize recorded distances;
3. **edge lower bound:** no explored edge violates shortest-label inequality;
4. **frontier equality:** exact small-oracle sets match at every depth;
5. **exhaustion:** no reachable successor escapes the final ball;
6. **rich output:** ties, labels, DAG edges, and counts match their oracle;
7. **distributed parity:** results agree across owner/device counts;
8. **failure parity:** replay/restart preserves the declared output;
9. **performance comparability:** compared runs share every semantic field.

Later rungs do not excuse missing earlier semantics. Performance comparison is
the last gate, not a substitute for correctness.

## How hardware measurements attach to the map

Measurements become interpretable when attached to the contract layer they
serve:

| Measurement | Explains | Does not prove |
|---|---|---|
| generated transitions | expansion work | unique states or completeness |
| accepted states | visited outcome | correct rejected identities |
| frontier bytes | one representation | candidate/visited/output peak |
| TEPS | normalized traversal rate | actual inspections or semantic parity |
| atomics/hash probes | physical contention | graph convergence alone |
| communication bytes | routing payload | global termination |
| GPU utilization | device occupancy | end-to-end optimality |
| matching counts | aggregate consistency | exact set equality |

This prevents a fast primitive from being promoted into a broad traversal
claim without the intervening evidence.

## Cross-layer counterexamples

- A valid length-three parent chain can coexist with an ignored length-two
  edge: output witness without distance minimality.
- A colliding visited hash can preserve most counts while losing the only path
  to a target: compact representation without exact identity.
- A complete local frontier can coexist with an in-flight remote candidate:
  local schedule without global completion.
- A quotient path can be shortest in orbit space but miss the fixed target
  representative: exact search on the wrong graph.
- An incidence BFS can be exact at graph depth two while the requested
  hyperedge distance is one: exact execution under the wrong metric.
- A bitmap can preserve reached membership while dropping two parent labels:
  exact scalar BFS with incorrect richer output.
- Equal final counts can hide swapped membership: aggregate validation without
  identity validation.

## Sources inside this research corpus

- Notes 3, 4, and 25 establish metric balls, frontier transactions, and the
  least-fixed-point view.
- Notes 6, 16, 17, 20, 22, and 34 define graph/state/action/version variants.
- Notes 8, 9, 18, and 30 establish stopping, fairness, quiescence, and recovery
  obligations.
- Notes 11, 13, and 19 define parent, count, source-label, and canonical-order
  outputs.
- Notes 7, 14, 15, 29, 32, and 36 connect representations and hardware work to
  the unchanged or explicitly changed semantics.
- The evidence map and research protocol define how facts, observations,
  failures, and performance claims are recorded.

## Current conclusion

Exact BFS is a chain of equalities and completion claims from the requested
state graph to a physical run. No single queue discipline, bitmap, parent path,
frontier count, or throughput number proves the chain. The contract map makes
each link explicit so that algorithmic variants can be studied without silently
inheriting guarantees they no longer satisfy.
