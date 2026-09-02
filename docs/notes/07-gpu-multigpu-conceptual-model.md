# BFS on one and many GPUs: a conceptual cost model

This note does not propose an optimized backend.  It asks a narrower question:
when exact BFS obligations are executed on GPUs, what physical work appears,
where can it move, and what evidence would distinguish competing explanations?

## Start from obligations, not kernels

At a level boundary with `visited = B_d` and `frontier = F_d`, an exact
implementation must eventually perform the logical transformation

```text
occurrences = expand(F_d)
candidates  = exact_unique_vertex_identities(occurrences)
new         = candidates minus B_d
F_(d+1)     = materialize_every(new)
B_(d+1)     = B_d union new
```

The phases need not be separate kernels or arrays.  A fused atomic claim can
combine deduplication, visited subtraction, insertion, and output reservation.
A sort pipeline can combine identity grouping with routing.  Fusion changes
traffic and synchronization; it does not remove the logical duties.

This gives a durable way to read GPU code: for every duty, identify where it is
performed, what exactness assumption it uses, and how failure or overflow is
reported.

## One count cannot describe BFS work

For each level, record a work vector rather than a single TEPS number:

```text
W_d = (
  |F_d|,                       frontier vertices
  generated_d,                edge or move occurrences attempted
  unique_candidates_d,
  already_visited_d,
  accepted_d = |F_(d+1)|,
  state_bytes, key_bytes, parent_bytes,
  bytes read/written/sorted/communicated,
  synchronization events
)
```

The same `|F_d|` can imply very different work under skewed explicit degrees.
The same `generated_d` can imply different work when move application is
expensive.  The same duplicate ratio can imply uncontended random keys or one
hot bitmap word.  The same kernel time can imply different traversal time when
level control or communication is included.

Graph500's traversed-edges-per-second metric is meaningful for its explicit
graph contract, but it does not charge implicit successor generation, wide
state equality, or unmaterialized candidate occurrences in a portable way.

## Where GPU parallelism comes from

At a high level, a GPU may parallelize over:

- frontier vertices;
- outgoing edges or generator applications;
- candidate records;
- visited words or hash buckets;
- owner/routing bins;
- vertices in the unvisited universe for pull traversal.

These decompositions expose different amounts and shapes of parallel work.

### Explicit irregular graphs

Frontier vertices can have sharply different degrees.  One thread per vertex
may be balanced in vertex count yet imbalanced in edge work.  Edge-centric or
prefix-sum decompositions redistribute adjacency intervals, but introduce
offset construction and additional memory traffic.  Push accesses adjacency
from the current frontier; pull scans unvisited vertices and incoming edges.

Merrill, Garland, and Grimshaw's prefix-sum task management and Gunrock's
frontier operators are ways to represent and schedule this irregular explicit
work.  They do not imply that prefix sums or materialized edge frontiers are
always appropriate for an implicit graph.

### Regular implicit/Cayley graphs

If every state applies `k` generators, `generated_d = k|F_d|` is regular in
count.  That removes degree imbalance but not necessarily cost imbalance:

- generators may have different transformation costs;
- legality tests may reject different fractions;
- state-dependent canonicalization can vary;
- ranking/equality may cost more than the move;
- different parents and generators can converge on highly correlated keys.

The natural parallel rectangle is `frontier x generators`.  Whether it is laid
out parent-major or generator-major changes which candidate identities meet in
a warp or block, even though the mathematical multiset is identical.  REF-016
and REF-017 directly demonstrate this distinction for one symmetric-group
family.

## The single-GPU cost layers

### 1. Expansion

Explicit BFS reads adjacency; implicit BFS computes a transformation.  The
useful metrics differ:

- explicit: adjacency bytes, degree distribution, cache/coalescing behavior;
- implicit: state bytes loaded, instructions per move, legality/reduction cost,
  and produced key/state bytes.

Reporting both as edges/s hides whether an "edge" was loaded or constructed.

### 2. Exact identity and visited

Common physical choices include a dense bitmap, dense distance array, exact
hash table, or global sort followed by exact comparison.  They make different
assumptions:

- bitmap: a proved dense injective rank and known capacity;
- dense distance: same universe assumption with more metadata;
- hash table: collision resolution and a proved non-lossy full condition;
- sort: a sortable exact key, or post-sort equality on full states.

An atomic bitmap claim is exact only if the bit index is exact identity.  The
atomic operation resolves concurrent claims; it cannot repair an aliased key.

### 3. Duplicate convergence

Duplicates can meet at multiple spatial and temporal scopes:

```text
same parent -> same warp -> same block -> same GPU batch
-> same owner after routing -> visited from an earlier level
```

A mechanism only removes duplicates that meet within its scope.  Warp
aggregation cannot remove equal keys placed in different warps.  Local sort
cannot remove duplicates generated on another GPU unless they later share an
owner or a global step.  This explains why "duplicate ratio" alone is not a
dispatch rule.

### 4. Compaction and capacity

After filtering, accepted states need addresses in the next frontier.  Possible
physical mechanisms include per-item atomics, block reservation, scan/select,
or fixed partitions.  Their semantic requirement is the same:

- every accepted state obtains exactly one retained slot;
- no rejected state is presented as accepted;
- capacity failure is explicit and does not masquerade as a smaller frontier.

Peak candidate storage and peak accepted frontier storage are different
capacity constraints.  Fusion can avoid a candidate array but may increase
register use or make exact overflow recovery harder.  These are hypotheses to
measure, not reasons to fuse automatically.

### 5. Parent and path metadata

Distance-only BFS can retain less data than path-producing BFS.  A parent claim
is often a benign race for distance correctness: any parent from the previous
level gives a shortest tree edge.  It is not benign when deterministic parents,
all shortest paths, move labels, or stable replay are required.

The [Graph500 specification](https://graph500.org/?page_id=12) explicitly
permits nondeterministic same-level parent races but validates the final tree:
tree edges must be real, levels must be consistent, graph edges cannot skip a
level, and the reachable component must be spanned.  That output contract is
narrower than all-shortest-path enumeration.

### 6. Level control

Even when all data stays on device, exact BFS has a loop-carried dependency:
the next level's size and termination depend on the current level.  Control can
be expressed through host launches, graphs, persistent device work, or another
protocol, but the semantic questions remain:

- when is the current level complete?
- is `F_(d+1)` empty?
- did any capacity error occur?
- has a target been found under a safe stopping contract?

Kernel duration, summed kernel duration, and wall-clock traversal duration are
therefore different measurements.  REF-017 found this distinction larger than
the measured kernel-policy differences on its small `S_9` traversal.

## Benign and non-benign races

Parallel BFS often tolerates several threads discovering the same child.  The
race is benign only relative to a declared output:

- **benign for distances:** one exact atomic visited winner retains the child at
  the correct next depth;
- **possibly benign for one path:** any valid previous-layer parent wins;
- **not benign for deterministic output:** the winner depends on scheduling;
- **not benign for capacity:** every loser that reserved space first may consume
  bounded output capacity;
- **not benign for all shortest paths:** losing parents are required output;
- **not benign without exact identity:** atomics serialize the wrong key.

Thus "benign race" is not a property of a code line alone.  It is a relation
between the race, capacity protocol, and promised result.

## Multi-GPU adds authority and transport

Multiple GPUs do not change the distance recurrence.  They split its data and
introduce work in flight.

### Ownership has several meanings

For each object, ask which rank/GPU is authoritative:

| Object | Possible placement question |
|---|---|
| Explicit adjacency | Which rank stores the outgoing/incoming edges? |
| Full implicit state | Which rank retains bytes needed for the next expansion? |
| Exact visited record | Which rank decides whether identity is new? |
| Current/next frontier | Is it source-local, owner-local, or replicated? |
| Parent metadata | Stored with child, source parent, or deferred? |
| Scratch and routing buffers | Fixed per peer, pooled, or staged? |
| Termination state | Which collective or protocol proves global completion? |

These owners may differ.  "Vertex partitioning" is incomplete unless it states
which of these objects follows the vertex.

### Explicit 1D and 2D partitioning

In a 1D vertex partition, a rank commonly owns vertices and their adjacency;
discovered remote vertices are routed to their owners.  A 2D partition treats
the adjacency matrix as blocks and uses row/column communication phases.
Buluç and Madduri analyze these as different communication structures for
stored sparse graphs, including cases where 2D partitioning limits collective
participants.

For an implicit Cayley graph there is no stored adjacency matrix to partition.
A 2D sparse-matrix result therefore does not transfer literally.  One can still
borrow the questions—who has input frontier information, who computes
transitions, who receives candidate identities—but the resulting protocol must
be derived from state generation and representation.

### Source computes versus owner computes

A common implicit model is:

```text
source GPU expands its local frontier
-> locally deduplicates optionally
-> routes candidate/state records by owner(identity)
-> owner performs authoritative visited claim
-> owner stores its accepted next-frontier states
```

Local dedup reduces transport but cannot decide global novelty.  Two sources
may emit the same child; only their common owner sees convergence.  Conversely,
aggressive pre-dedup can be counterproductive when local sorting/packing costs
more than the bytes saved.  REF-005, REF-006, REF-010, and REF-011 model these
trade-offs but do not measure a real interconnect.

### The communication work vector

For each level and rank, useful quantities include:

```text
generated_local
unique_before_route
local_owner_candidates
remote_candidates_by_peer
bytes_by_record_field
unique_after_owner_merge
accepted_by_owner
max / mean work and bytes across ranks
```

Total bytes alone misses skew and message fragmentation.  Perfect byte balance
can still perform poorly if it destroys locality, creates many tiny messages,
or serializes one visited owner.  Conversely, extra bytes may be tolerable when
communication is overlapped with independent local work.  Claims about overlap
require a timeline, not just asynchronous API calls.

### Global level completion

A local empty frontier does not end distributed BFS.  Other ranks may have
frontiers or messages in flight.  A level-synchronous protocol needs an
equivalent of:

```text
all expansion complete
and all routed candidates delivered
and all owner visited decisions complete
and global sum(|F_(d+1)|) == 0
```

Target search adds another condition: no outstanding shallower connection can
beat the best known target path.  Broadcasting "target seen" is notification,
not by itself a shortest-path proof.

## Three meanings of multi-GPU scaling

Speedup is ambiguous unless the experiment names its regime:

1. **Strong scaling:** fixed graph/problem, more GPUs, lower time.
2. **Weak scaling:** problem size grows with GPU count while time or work rate is
   tracked.
3. **Capacity scaling:** the larger aggregate memory makes a previously
   impossible BFS fit, even if it is not faster.

A system can succeed at capacity scaling while losing strong-scaling efficiency
to communication and barriers.  That is not a failed result; it answers a
different question.

## Counterexamples to tempting hardware intuitions

### More parallel candidates means more useful work

A wide occurrence bag may consist mostly of duplicates or old states.  It can
fully occupy the GPU while discovering almost nothing and exhausting temporary
capacity.

### Fewer atomics means faster

Replacing per-item atomics with block scans adds instructions and barriers.  If
atomics were not the bottleneck, the replacement can be slower.  REF-014 is a
local counterexample.

### Sorting removes duplicates, therefore it wins

Sorting globally converges duplicates but moves every record and requires
scratch capacity.  Direct exact bitmap claims can be cheaper for dense ranks;
REF-015 rejects the universal claim for its tested profiles.

### Regular Cayley degree means regular GPU behavior

The number of moves per parent may be constant while candidate identities,
bitmap words, legality, ranking, and duplicate locality remain highly
structured.  Parent-major and generator-major layouts in REF-016/017 produce
the same transitions but different warp convergence.

### Twice the GPUs gives twice the speed

Every level can introduce routing, skew, collective latency, and a global
dependency.  Small or narrow frontiers may expose less local work than the fixed
coordination cost.  More memory capacity and more throughput are distinct.

### An asynchronous collective is overlapped

Issuing an operation asynchronously proves only that the call can return before
completion.  Useful overlap requires independent ready work and a timeline
showing simultaneous progress without hidden stream or host serialization.

## A disciplined measurement ladder

Without committing to a new backend, evidence can progress in layers:

1. **Semantic accounting:** exact per-level frontier/candidate/visited counts.
2. **Representation accounting:** state, key, parent, scratch, and capacity
   bytes derived and checked.
3. **Isolated probe:** one primitive with exact fixtures, warmups, repeats, and
   explicit scope.
4. **Traversal timeline:** end-to-end level loop separating kernels, transfers,
   host gaps, and synchronization.
5. **Multi-rank simulation:** exact ownership, duplicate convergence, byte/skew
   accounting without claiming network speed.
6. **Real multi-GPU measurement:** topology, peer paths, collectives, per-rank
   timelines, and correctness parity across rank counts.

Evidence at one rung cannot establish conclusions from a later rung.  In
particular, isolated throughput does not prove traversal speed, and simulated
wire bytes do not prove multi-GPU scaling.

## Sources

- Duane Merrill, Michael Garland, and Andrew Grimshaw,
  *Scalable GPU Graph Traversal*, PPoPP 2012,
  [paper](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf),
  for work-efficient explicit GPU frontier traversal and prefix-sum task
  management.
- Yangzihao Wang et al., *Gunrock: GPU Graph Analytics*, ACM TACO 2017,
  [arXiv preprint](https://arxiv.org/abs/1701.01170), for bulk-synchronous
  data-centric frontier operators.
- Scott Beamer, Krste Asanovic, and David Patterson,
  *Direction-Optimizing Breadth-First Search*, SC 2012,
  [paper](https://www.scottbeamer.net/pubs/beamer-sc2012.pdf), for top-down and
  bottom-up explicit-graph work trade-offs.
- Aydin Buluç and Kamesh Madduri,
  *Parallel Breadth-First Search on Distributed Memory Systems*, SC 2011,
  [arXiv preprint](https://arxiv.org/abs/1104.4518), for 1D/2D partitioning and
  distributed communication structure.
- [Graph500 benchmark specification](https://graph500.org/?page_id=12), for an
  explicit BFS output/validation contract and TEPS measurement scope.
- Local evidence REF-005/006/010/011 for owner-routing simulations and
  REF-012 through REF-017 for bounded single-GPU probes.

## Current synthesis

GPU BFS performance is the physical cost of maintaining an exact evolving
metric boundary.  A GPU can parallelize expansion, identity checks, duplicate
convergence, and compaction, but it cannot eliminate their semantic roles.
Multiple GPUs additionally distribute authority and introduce work in flight;
correct global completion becomes as important as local throughput.

The useful transferable knowledge is therefore a map from BFS obligations to
work, bytes, contention, capacity, and synchronization.  A particular kernel,
layout, partition, or collective remains a scoped hypothesis until measurements
on the intended graph representation and execution regime support it.
