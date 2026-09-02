# External-memory BFS: the level as a set transaction

External memory does not change graph distance. It changes which physical
operations can realize the exact layer recurrence without random access to all
of `visited`.

At the end of level `d`, let

```text
C_(d+1) = multiset union of successors of every u in F_d
F_(d+1) = unique(C_(d+1)) \ B_d
B_(d+1) = B_d union F_(d+1).
```

An external-memory algorithm is ordinary BFS if its files, partitions, sorts,
and merges implement these set equations exactly and no member of `F_(d+1)` is
expanded before the logical transition to the next level.

## A level is a transaction, not necessarily one in-memory queue pass

The physical schedule may stream chunks of `F_d`, generate candidate runs,
locally deduplicate or partition them, merge duplicates across all runs, remove
states already in authoritative `B_d`, and commit `F_(d+1)`. Only then may that
frontier be expanded. These physical operations may be interleaved or reordered
when the final set is identical. The logical barrier is before expansion, not
between every file operation.

## Two kinds of duplicate

- **same-layer duplicates:** several edges or batches produced the same state
  in `C_(d+1)`;
- **old duplicates:** a candidate is already in `B_d` and therefore has a
  shorter or equal previously established distance.

Candidate deduplication cannot replace visited subtraction. Local
deduplication cannot replace cross-run, cross-partition, or cross-GPU
deduplication. A correct owner/merge protocol must eventually make one exact
set decision for every state identity.

As set algebra,

```text
unique(C) \ B = unique(C \ B).
```

Therefore checking old visited before or after same-layer deduplication can
produce the same frontier. The orders differ in I/O and intermediate volume,
not necessarily in BFS semantics. A claim that only one order is correct is too
strong unless concurrency, parent metadata, or an approximate filter adds
another constraint.

## Early membership commit is not early expansion

An in-memory BFS may atomically insert a newly generated state into `visited`
immediately. During the level, physical `visited` then represents `B_d` plus a
subset of `F_(d+1)`, not immutable `B_d`. This is still exact for scalar
distances: the first successful claim emits the state once, later claims are
same-layer duplicates, and no claimed state is expanded until the next level.

The unsafe step is to expose a claimed `F_(d+1)` state as work for the current
level. That can cascade several graph hops in one round. Richer outputs also
need care: first claim gives an arbitrary shortest parent, but does not preserve
all shortest parents, path counts, or a canonical tie rule by itself.

```text
early exact membership commit  may be ordinary BFS
early next-state expansion      is not level-synchronous BFS by default.
```

## Why spilling the frontier is only one case

- **frontier spill:** frontier files exceed fast memory, while graph and exact
  visited membership remain cheaply accessible;
- **semi-external BFS:** vertex-indexed metadata may fit, while edge adjacency
  lives externally;
- **fully external search:** frontier, candidates, adjacency/state records, and
  duplicate-detection data all require external organization.

Spilling only `F_d` does not solve random membership probes into an oversized
`B_d`. Fully external BFS must also arrange identity partitioning, sequential
runs, exact cross-run duplicate removal, and durable level state. Its natural
cost vocabulary is transferred blocks, scans, sorts/merges, and passes—not just
generated edges.

## Delayed duplicate detection

Delayed duplicate detection writes candidates without a random closed-set
lookup for each occurrence, then removes duplicates in bulk by sorting, merging,
or hash partitioning. Delay is compatible with BFS because candidate occurrence
is not yet a distance commitment; the exact layer is settled before expansion.

Approximate filters have asymmetric roles:

- a false positive used as a final rejection can delete a reachable state and
  destroy completeness;
- a false positive used only to request an exact secondary check changes work,
  not the frontier set;
- a false negative is harmless only if a later exact stage catches the
  duplicate before expansion/commit.

Hash equality alone is not state equality. Exact search needs collision-free
ranking or collision resolution against canonical state records.

## Explicit graphs versus implicit state spaces

Dense explicit vertex IDs are easy to sort and merge conceptually. An implicit
state instead needs deterministic canonical serialization, exact equality,
stable partitioning, enough payload for later successor generation, and any
required parent/move metadata.

A perfect dense rank makes identities compact but is not required. Conversely,
a hash makes partitioning possible but does not make identity exact. Quotienting
by puzzle symmetry is a separate graph transformation, valid only after proving
that it preserves the requested distance and target semantics.

## Cayley-graph observations

For a finite generated group, canonical element encodings and deterministic
generators make delayed candidate runs natural in principle. Relations create
same-layer and old duplicates, so exact global identity remains essential.

For a non-rankable or expensive-to-rank representation, disk partitions can be
keyed by a hash prefix while retaining the full canonical element for collision
resolution and later expansion. This preserves semantics but may increase
record and I/O volume.

The full-universe scan problem that limits pull BFS is different: push-based
external BFS streams only the reached frontier and successors. It need not
enumerate every unvisited group element.

For infinite locally finite Cayley graphs, every finite-depth layer is finite,
so layer-at-a-time external enumeration is meaningful. It remains only
solution-complete for finite-depth targets; exhaustion cannot decide that an
unreachable target does not exist in an infinite component.

## Multi-GPU and storage ownership

Hash ownership appears in both distributed-memory and external algorithms, but
the resources are not interchangeable. GPU owners may support random membership
and synchronized all-to-all exchange; disk partitions favor sequential transfer
and bulk merge passes; host spill introduces a third tier shared by GPUs.

In every case, correctness needs an authoritative exact identity decision and a
global statement that every contribution to `F_(d+1)` has arrived before it is
expanded. A local empty queue is not a global level barrier.

Crash recovery adds another semantic issue. Replaying a partially committed
level can preserve an idempotent frontier set, but non-idempotent path counts or
parent logs need their own commit protocol.

## Counterexamples and failed intuitions

- **`visited` must remain immutable throughout the level.** Too strong: exact
  first claims may be committed early if they are not expanded early.
- **Local run deduplication is enough.** False across runs, partitions, or GPU
  send buffers; a later authoritative union is required.
- **A Bloom filter can be the exact disk visited set.** False if final false
  positives reject states; it can only prefilter an exact decision.
- **Frontier spill solves out-of-core BFS.** Only when graph and exact visited
  operations fit the remaining memory regime.
- **External BFS requires a dense rank.** False: canonical records plus exact
  collision handling suffice conceptually.

## Audit questions

1. What physical object represents `B_d` at each level boundary?
2. Where are same-layer duplicates merged across every producer?
3. Can approximate membership permanently reject a state?
4. When may a committed next-layer state first be expanded?
5. Is hash collision resolution exact?
6. Which data fit in device memory, host memory, and external storage?
7. Are costs edges, bytes, I/O blocks, passes, or wall time?
8. Does recovery replay a level without changing parents or path counts?
9. Does a Cayley quotient preserve the particular target and distance?
10. Who proves globally that the next frontier is complete?

## Sources

- Kurt Mehlhorn and Ulrich Meyer, *External-Memory Breadth-First Search with
  Sublinear I/O*, ESA 2002,
  [author PDF](https://www.mpi-inf.mpg.de/~mehlhorn/ftp/ExternalBFS.pdf), for
  the block-I/O model and BFS beyond naive random graph access.
- Richard E. Korf, *Best-First Frontier Search with Delayed Duplicate
  Detection*, AAAI 2004,
  [AAAI paper](https://aaai.org/Papers/AAAI/2004/AAAI04-103.pdf), for bulk
  sequential duplicate detection in implicit problem spaces.
- The `multigpu_beam` expert suggested complete expansion, exact set difference,
  and next-frontier commit as the core logical sequence. The stronger suggestion
  that physical visited must stay exactly `B_d` during the whole level was
  narrowed above: exact early claims are compatible with BFS when expansion
  remains level-separated.

## Current synthesis

External-memory BFS is a bulk realization of the same set recurrence under a
memory hierarchy. Sorting and delayed duplicate detection preserve BFS when
their final effect is exact `unique(expand(F_d)) \ B_d` and the completed next
layer is the first unit allowed to advance. Representation choices decide I/O
and recoverability; they do not relax identity, level separation, or global
completion.
