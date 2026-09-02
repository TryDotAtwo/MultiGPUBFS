# Distributed bottom-up BFS: systolic early exit and exact snapshots

## Question

How can a 2D distributed bottom-up BFS preserve the work-saving “stop after one
frontier parent” rule when one unvisited vertex's incoming adjacency is split
across several processes?

Note 14 gives the general pull semantics and implicit-state boundary. Note 189
gives top-down 2D expand/fold obligations. This note studies the specific
systolic bottom-up protocol of Beamer et al. as a semantic case study, not as a
recommended implementation.

## 1. Sequential bottom-up predicate

For a fixed exact frontier snapshot `F_d`, an unvisited vertex `u` belongs to
the next layer exactly when

```text
Pred(u) intersect F_d is nonempty.
```

For distance plus one arbitrary parent, scanning may stop at the first matching
predecessor. The optimization is existential: after one true witness, no other
incoming edge can improve scalar depth `d+1`.

The predicate must use incoming edges for a directed graph. On an undirected
graph the same stored neighbor relation serves both directions. The source
report's Graph500 setting is undirected; transpose/orientation must be declared
before transferring the protocol to directed data.

## 2. Why naive parallel shards lose early exit

In a 2D checkerboard, incoming neighbors of one candidate can be stored on
several processor-column shards. If every shard scans in parallel:

- several shards may find parents redundantly;
- shards with no early hit may continue long scans after another shard succeeds;
- a stop notification crosses the network too late to prevent current work;
- arbitrary parent becomes race/order dependent;
- richer outputs may need all matches anyway.

The result can remain distance-correct under exact merging, but the intended
bottom-up work reduction may disappear. Correctness and early-exit efficiency
are separate claims.

## 3. Frontier gather

The Beamer et al. 2D algorithm stores current frontier membership as a dense
bitmap during bottom-up levels. At each level, vector redistribution and a
column all-gather give every process the frontier segment corresponding to the
sources of its local incoming edges.

This is partial replication, not a full frontier copy on every process. Its
correctness contract is exact local membership for every predecessor that the
process may inspect:

```text
local frontier bit = 1 iff that predecessor is in the completed global F_d.
```

A false negative can miss the only parent. A stale or false-positive bit can
accept a vertex at the wrong level. Approximate frontier membership is therefore
not a safe final predicate.

## 4. `p_c` systolic substeps

Let the processor grid have `p_c` columns. One bottom-up BFS level is divided
into `p_c` substeps. At one substep, only one processor-column shard is
responsible for checking a given candidate vertex's local incoming edges.

After the substep:

- discovered `(child,parent)` updates are sent toward the authoritative parent
  vector segment;
- a dense `completed` bitmap records candidates that already found a parent;
- responsibility and the relevant completed segment rotate to the next process
  along the processor row;
- the next shard skips candidates whose completed bit is set.

After all `p_c` rotations, every still-uncompleted unvisited candidate has had
all relevant shards considered for that level. This temporal partition restores
early exit across shards at substep granularity.

## 5. Exact invariants of the rotation

For each candidate `u`, the protocol needs:

1. **Snapshot invariant:** every shard tests the same closed `F_d`, never the
   growing `F_(d+1)`.
2. **Coverage invariant:** unless `u` is validly completed, responsibility visits
   every shard that can contain an incoming edge of `u`.
3. **Witness invariant:** `completed(u)` changes from zero to one only after an
   exact local edge `(v,u)` with `v in F_d` is found.
4. **Persistence invariant:** once set in the level/visited epoch, the completed
   state reaches later responsible shards before they decide to scan `u`.
5. **Publication invariant:** a valid witness produces a durable parent update
   and next-frontier membership; the stop bit cannot outrun recoverable output.
6. **Exhaustion invariant:** after all rotations, zero completed means all
   eligible shards were checked without a witness.
7. **Epoch invariant:** frontier and completion bits from another level cannot
   be interpreted in the current predicate.

These are semantic requirements. The exact send/receive schedule can differ if
it proves the same properties.

## 6. Completion bitmap is not merely a cache

The report's `completed` bitmap denotes vertices that have already found
parents and no longer need parent search. It participates in visited/no-search
state as well as intra-level early termination.

Error directions differ:

- false completed for a genuinely unvisited candidate can suppress its only
  discovery;
- false uncompleted causes extra scans and possibly duplicate proposals;
- losing a newly set completed bit can re-open later shards;
- publishing completed before the parent/next-frontier record is durable can
  create the blind-drop state of note 178.

Therefore it requires exact epoch semantics or an exact authoritative fallback.
Calling it an advisory optimization would be incorrect if shards use it to skip
semantic work definitively.

## 7. Early exit is delayed, not instantaneous

Systolic ordering avoids scanning later shards after a hit, but it cannot undo:

- neighbor checks already performed inside the successful shard;
- work concurrently performed for other candidates;
- communication and synchronization at every substep;
- bitmap rotation and parent-update traffic.

Its critical path has `O(p_c)` sequential substeps per bottom-up level in the
source model. Increasing shard count can reduce adjacency per shard while
increasing latency and rotation overhead. This is a work/parallelism tradeoff,
not a universal scaling result.

Candidate order and shard order affect checks-before-hit and chosen arbitrary
parent. They do not affect scalar distance if every accepted parent is in the
same exact `F_d` and every candidate is exhaustively covered when no hit occurs.

## 8. Output contracts that remove early exit

The systolic protocol in the source report returns one BFS parent. For richer
outputs:

- canonical parent requires proving no later shard contains a smaller valid
  predecessor;
- complete predecessor DAG requires every frontier predecessor;
- exact path count requires every semantic predecessor contribution exactly
  once;
- multi-source canonical labeling may require a better source label on a later
  shard.

In these contracts, `completed after first hit` is not a valid skip predicate.
The scan may still avoid irrelevant nonfrontier edges through indexing, but the
central existential early-exit theorem no longer applies.

## 9. Top-down versus bottom-up 2D traffic

Top-down note 189 communicates active source identity and folds generated
candidate destinations. The systolic bottom-up protocol instead communicates:

- relevant frontier bitmap segments once per level;
- rotating completed bitmap segments across substeps;
- discovered child-parent updates;
- mode-conversion/vector-transpose state when switching direction.

It can reduce candidate-edge traffic by stopping after a parent, but adds dense
bitmap and `p_c`-substep costs. Comparing only “edges examined” omits the
communication that makes those avoided examinations possible.

## 10. GPU and multi-GPU interpretation

On GPUs, the same tension appears at several scales:

- a thread can sequentially scan one candidate's predecessors and break;
- a warp/block can cooperatively scan but needs a hit/ballot protocol;
- another GPU needs an explicit completion notification or staged ownership;
- wide frontier/completed bitmaps consume bandwidth even when compact;
- a global `p_c`-stage schedule can leave some devices with little useful work.

A multi-GPU probe should record at least:

```text
unvisited candidates tested
candidate-shards visited
predecessor checks before hit
checks after another worker already found a hit
frontier bitmap bytes
completed bitmap rotation bytes
parent update bytes
substep synchronization time
mode conversion time
scalar/DAG/count output contract
```

One parent-finding kernel timing is not a distributed bottom-up level timing.

## 11. Implicit and Cayley transfer

To imitate systolic pull on an implicit action, one would need all of:

- an enumerable exact unvisited universe;
- incoming/inverse transitions for each known candidate;
- an exact current-frontier membership representation;
- a partition of predecessor generators or transformations into shards;
- completion state that follows candidate responsibility;
- exact full-state or injective-rank identity;
- replayable parent/move metadata.

For a rankable Cayley graph with symmetric generators, generator shards could
sequentially test inverse moves of each unvisited state. But every unvisited
state must still be enumerated or represented, and full-state reconstruction,
inverse transformation, rank, and bitmap traffic remain. Regular degree alone
does not make the outer pull scan cheap.

Stabilizer aliases can let several generator shards find the same parent state
under different labels. First-hit is valid for one arbitrary parent but loses
labeled multiplicity. Infinite Cayley graphs have no finite full-universe pull
pass, so the protocol does not transfer there.

## 12. Failure and termination boundaries

A lost completed rotation can add work or duplicate parents; a falsely set bit
can lose reachability. A lost parent update after a valid stop can leave a state
claimed but unpublished. Retry therefore needs stable child/level identity and
an idempotent or compensating output merge.

Bottom-up level completion requires a consistent cut containing:

- closed frontier bitmap distribution;
- all `p_c` responsibility rotations;
- all local predecessor scans;
- all completed-state transfers;
- all parent/next-frontier publications.

Only then can the new frontier replace the current snapshot. Local completion
of one rotation or one process row is insufficient.

## Sources

- Scott Beamer, Aydın Buluç, Krste Asanović, and David Patterson,
  [*Distributed Memory Breadth-First Search Revisited: Enabling Bottom-Up
  Search*](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/Archive/EECS-2013-2.pdf),
  UCB/EECS-2013-2. Algorithm 4 supplies the frontier all-gather, `p_c` systolic
  substeps, rotating completed bitmap, and parent update protocol.
- Aydın Buluç and Kamesh Madduri,
  [*Parallel Breadth-First Search on Distributed Memory Systems*](https://arxiv.org/abs/1104.4518),
  SC 2011, for the underlying 2D top-down decomposition.
- Notes 14, 56, 174, 178, and 189 supply this corpus's pull predicate,
  distributed closure, publication, and expand/fold boundaries.

## Compact conclusion

Distributed bottom-up early exit is not a free parallel membership test. In the
studied 2D protocol, candidate responsibility moves through `p_c` adjacency
shards while exact completed state suppresses later scans. This restores
first-parent early exit at the cost of sequential substeps, bitmap movement,
and strict snapshot/publication invariants. It supports one arbitrary parent;
richer shortest-path outputs weaken or remove the early-stop advantage.
