# What explicit GPU BFS papers transfer to implicit Cayley search

Classic GPU BFS papers contain durable ideas about irregular parallel work,
frontier regimes, duplicate convergence, and communication. Their numerical
results and data-structure conclusions are not representation-neutral.

This note reads four influential lines of work through the contract map. It is
a source synthesis, not an implementation proposal.

## Merrill, Garland, and Grimshaw: work-efficient explicit traversal

[Scalable GPU Graph Traversal](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf)
represents a sparse graph as CSR row offsets and column indices. Its reference
contract addresses vertices by indices, stores `dist[0..n-1]`, and expands
pre-materialized adjacency lists. Undirected graphs are represented as two
directed edge records, and traversal rates use directed-edge counts.

The paper's important transferable observations include:

- useful work should follow logical frontiers rather than rescan every graph
  vertex or edge each round;
- adjacency-list lengths create fine-grained load imbalance;
- separating or fusing expansion, visited lookup, and compaction trades extra
  memory movement against concurrency and launch latency;
- concurrent discovery can cause redundant expansion even when final distances
  remain correct;
- small and large frontiers favor different physical scheduling regimes;
- multi-GPU striping, duplicate culling, routing, and barriers interact.

But its primitive objects are compact vertex identifiers and stored neighbor
IDs. Its status lookup includes label arrays and bitmap-assisted filtering; the
multi-GPU design can provision each GPU with a full `n`-bit best-effort mask.
Those assumptions do not hold automatically for a 128-byte implicit state with
no proved dense rank.

The paper's “edge expansion” loads an already stored column index. An implicit
Cayley transition may instead perform a state permutation, legality rule,
canonicalization, hash/rank construction, full-state equality, and wide-state
write before an endpoint even exists. Equal TEPS numerators would therefore
hide different algorithms and byte traffic.

## Beamer, Asanovic, and Patterson: direction optimization

[Direction-Optimizing Breadth-First Search](https://www.scottbeamer.net/pubs/beamer-sc2012.pdf)
targets low-diameter, often scale-free explicit graphs. Its bottom-up step
iterates unvisited vertices, scans their neighbors until finding a member of a
frontier bitmap, and stops the scan after the first parent.

The conceptual result transfers cleanly:

> Push and pull are two evaluations of the same exact next-frontier predicate,
> and their work changes with frontier geometry.

The operational preconditions are much narrower:

- the vertex universe is known and cheaply enumerable;
- unvisited membership is available for every vertex;
- predecessor/neighbor lists are accessible without generating the universe;
- frontier membership fits an efficient bitmap-like query;
- finding any parent is sufficient for the output contract;
- scanning many non-frontier predecessors is affordable in the target regime.

An implicit puzzle graph typically exposes `successors(current_state)` rather
than an array of every possible state. Inverse generators provide predecessor
transitions for a *known* state, but do not provide a cheap enumeration of all
unvisited states. Therefore “inverse moves exist” is not enough to inherit the
paper's bottom-up algorithm or switching thresholds.

The paper explicitly motivates low-diameter scale-free workloads whose large
frontiers cover a substantial fraction of all vertices. A bounded-degree
Cayley graph can have entirely different diameter, sphere growth, and
enumerability even though both traversals are called BFS.

## Enterprise: CSR, resident arrays, and TEPS semantics

[Enterprise](https://personal.stevens.edu/~hliu77/docs/sc15.pdf) evaluates CSR
graphs resident in GPU global memory. Its paper states that data are represented
with `uint64`, runs 64 source searches, and computes TEPS as directed input
edges in the traversed search divided by elapsed BFS time, counting parallel
edges and self-loops.

This clarifies two comparison traps:

1. `uint64` refers to explicit graph data records, not a 64-byte or 128-byte
   semantic state with collision-resolving identity.
2. The TEPS numerator is graph-volume based; it is not necessarily the number
   of physical neighbor checks, especially under direction optimization.

Enterprise's frontier construction, hub handling, scheduling, and direction
switching remain useful examples of regime-dependent GPU behavior. Their
reported throughput and crossover points are bound to CSR bytes, resident
status arrays, graph degree distribution, hardware generation, and output
contract.

## Bisson, Bernaschi, and Mastrostefano: distributed explicit BFS

[Parallel Distributed Breadth First Search on the Kepler Architecture](https://arxiv.org/abs/1408.1605)
uses a 2-D partition of an explicit sparse adjacency matrix. Local partitions
are compressed, global vertices map to local indices, predecessor and level
arrays have known sizes, visited is a bitmap, and local graph work uses 32-bit
indices. Evaluation uses R-MAT/Graph500-style graphs symmetrized by adding the
opposite edge and reports TEPS.

Its durable multi-GPU lessons are structural:

- partition choice determines both load distribution and communication;
- local compression and duplicate removal can reduce exchanged records;
- frontier expansion and owner/update phases need global coordination;
- communication can dominate a distributed BFS despite abundant GPU compute;
- local IDs can shrink traffic when an exact global-to-local mapping exists.

For wide implicit states, the mapping itself is a major unsolved assumption.
Routing a 32-bit local vertex ID is unlike routing a full state plus exact
collision evidence. A state generated on one rank may need to move to its owner
before equality is authoritative, and the owner may require full bytes rather
than only a hash. Consequently the paper's communication bytes per frontier
item and capacity limits cannot be copied directly.

## Assumption matrix

| Field | Typical explicit-paper contract | Implicit Cayley question |
|---|---|---|
| vertex identity | dense integer index | full state, injective rank, or collision-resolved key? |
| graph storage | CSR/CSC already resident or partitioned | transition generated on demand? |
| visited | array or bitmap indexed by vertex | exact wide-key set or proved rank? |
| unvisited universe | known `0..n-1` | enumerable at acceptable cost? |
| degree work | load adjacency entries | apply generator, legality, hash, equality, write state |
| direction | often undirected/symmetrized | positive directed generators or symmetric set? |
| duplicate identity | equal integer ID | equal semantic state after quotient/context? |
| output | distance/parent tree or traversal | path replay, canonical labels, all parents, beam solution? |
| capacity | graph and arrays sized from `n,m` | frontier, state payload, visited, parents, scratch unknown a priori? |
| communication | compact IDs/local indices | hash plus state, owner proof, or regeneration? |
| metric | input edge | one generator action or macro/suffix step? |
| completion | exact component/level traversal | radius, target, exact BFS, lookup, or pruned beam? |

Every row need not match literally. It must be translated explicitly before a
performance conclusion is compared.

## What can transfer safely

These are broad mechanisms or questions, not numerical promises:

- frontiers can be too narrow to saturate a GPU and too wide for temporary
  buffers;
- degree or transition cost can be imbalanced even when vertex count is not;
- duplicate location determines whether warp-, block-, rank-, or owner-local
  removal can see it;
- extra data movement can outweigh saved arithmetic;
- representation conversion has a frontier-dependent break-even point;
- global barriers and communication limit strong scaling;
- partitioning changes skew, locality, and exchanged bytes;
- end-to-end behavior cannot be inferred from one isolated kernel.

These principles survive because they describe dependencies and resource
flows. Their magnitude remains workload-specific.

## What does not transfer without new evidence

- absolute or relative TEPS;
- bitmap visited throughput without a dense exact rank;
- push/pull switching thresholds without an enumerable state universe;
- CSR adjacency load balance as a proxy for generator cost;
- bytes per candidate when candidate payload changes from ID to state;
- duplicate-culling effectiveness when equal keys have different physical
  locality;
- graph partition balance under a different owner function;
- parent races when the requested output is deterministic or all-parent;
- strong-scaling curves across different interconnects and state widths;
- exact-BFS guarantees for an outer beam or approximate visited scheme.

## A comparison passport

Before placing an explicit-paper result beside implicit Cayley BFS, record:

```text
semantic vertex and exact equality:
state/key/frontier record bytes:
directedness and generator set:
stored adjacency versus generated transition:
cost and validity of one transition:
enumerability of all/unvisited states:
visited and collision-resolution mechanism:
frontier completeness versus pruning:
output parent/path/count contract:
termination and completed-radius contract:
single-/multi-GPU ownership and routing:
actual generated, inspected, unique, and accepted counts:
state-transform, identity, dedup, communication, barrier times:
peak graph/frontier/visited/parent/scratch bytes:
throughput numerator and denominator:
hardware, topology, software, and graph instance:
```

Minimum useful implicit metrics are not one replacement TEPS number. They form
a funnel:

```text
generated transitions
-> valid transitions
-> distinct current candidates
-> previously unseen exact states
-> accepted frontier states.
```

Report rates and byte costs at the stages relevant to the claim. A high
generated-transition rate with near-zero acceptance may mean excellent kernel
occupancy and poor search progress simultaneously.

## Relation to the current CayleyPy outer search

The inspected `D:\100XH100` runner uses wide implicit states, generated moves,
Zobrist-key deduplication, learned scores, and a bounded global beam. It does not
have the same semantic output as exact CSR BFS in these papers.

Therefore its throughput cannot be used as an exact-BFS comparison until at
least two separations are made:

1. compare physical primitives only under explicitly different algorithm
   labels; or
2. construct an exact complete-frontier workload with an exact identity oracle
   and matched output/termination contract.

This is a comparison boundary, not a request to build the second system.

## Expert recommendation and independent check

The `multigpu_beam` expert independently highlighted dense IDs, resident
adjacency, enumerable unvisited vertices, cheap equality, graph direction, and
wide payload/output memory as the main transfer barriers. Those points were
accepted only where the primary papers above explicitly expose the corresponding
CSR/CSC, bitmap, ID, pull, TEPS, or parent-array assumptions.

The expert also recommended separating generated, valid, unique, accepted,
dedup, identity, communication, synchronization, and memory metrics. This is
consistent with the project's existing accounting in notes 4, 7, and 29; it is
not treated as measured evidence.

## Current conclusions

1. Classic GPU BFS papers predominantly optimize traversal of explicit indexed
   graphs, not construction and equality of wide implicit states.
2. Their scheduling, frontier, duplicate, partition, and communication insights
   transfer as questions and mechanisms.
3. Their numeric throughput, thresholds, bitmaps, payload sizes, and scaling
   curves do not transfer without matched representation and output contracts.
4. Pull requires an enumerable unvisited universe in addition to inverse edges.
5. TEPS must retain its exact graph-volume numerator; implicit search needs
   separate generated/unique/accepted and byte-stage metrics.
6. Comparing CayleyPy beam results with exact explicit BFS without naming both
   semantic and physical differences would compare different problems.
