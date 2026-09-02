# REF-010 exact distributed BFS output audit

REF-010 is the workspace's closest exact distributed BFS artifact. It is not a
multi-GPU runtime: it is a deterministic CPU simulation of bulk-synchronous
owner-computes bidirectional BFS. This note applies the output-contract matrix
from note 57 and the global stopping semantics from notes 08 and 56.

REF-023 reran it in Docker and reproduced its retained artifacts after
normalizing CRLF/LF; both focused distributed-bidirectional unit tests also
passed.

## Logical distributed machine

For one selected direction and one complete depth, the model performs:

```text
owned frontier parents
  -> complete transition generation
  -> source-rank pre-dedup
  -> route by owner(candidate)
  -> owner-side exact dedup
  -> authoritative visited lookup/claim
  -> local opposite-side intersection lookup
  -> globally visible next frontier and incumbent.
```

Both directions use the same owner function, so forward and reverse records for
one state converge at the same logical authority. This makes an intersection a
local dictionary lookup after routing.

No actual messages exist in the simulator. "Remote" means that source owner
and destination owner differ; records remain ordinary process memory. Thus the
model exercises ownership and conservation semantics without transport timing,
delivery, or failure behavior.

## Why the distance result is exact in the model

The implementation maintains exact forward and reverse distance maps. It
expands one complete current frontier, owner-deduplicates every candidate,
subtracts that direction's accumulated visited set, and checks each newly
accepted state against the persistent opposite distance map.

At a loop boundary:

```text
forward_depth = minimum unexpanded forward depth
reverse_depth = minimum unexpanded reverse depth
best_distance = shortest discovered feasible meeting path.
```

It stops when

```text
forward_depth + reverse_depth >= best_distance.
```

This is note 08's complete-level theorem, a special case of note 56. Since the
simulator has no in-flight work between completed rounds, its depth counters
really are global unfinished minima at the evaluation point.

If either direction's frontier exhausts before a meeting, the loop ends. In the
model this is an exact no-path result because the emptied frontier follows a
complete successor-closed visited ball, not a merely local empty rank.

## One shortest path, not every shortest path

For each newly discovered state, REF-010 retains one parent and move. Source
pre-dedup keeps the first `(source_owner,state)` occurrence; owner convergence
then chooses one record by a deterministic implementation tuple favoring local
production and lower source owner.

When a best meeting is fixed, the model follows:

```text
meeting -> forward parent chain -> start
meeting -> reverse next chain -> target
```

and asserts that the reconstructed move count equals the proved distance. The
experiment independently replays the returned moves.

Losing same-depth parents are discarded as frontier duplicates. Therefore:

- distance is exact;
- one returned path is shortest and replayable;
- parent choice may depend on transition/frontier/source-owner order;
- no complete predecessor DAG or all-path claim survives deduplication.

## Determinism versus canonicality

REF-023 reproduced the current artifacts exactly after newline normalization.
That establishes deterministic reproduction for the fixed Python version,
iteration order, owner functions, transition order, policies, and source.

It does not define a semantic canonical path order. The chosen parent can depend
on:

- input transition order;
- frontier insertion order;
- which producer owns the parent;
- local-versus-remote preference;
- owner function and world size.

The artifacts do not record returned paths, only mismatch counts and routing
metrics, so cross-configuration path identity was not compared. Zero replay
failures means every selected path was valid, not that all configurations chose
the same word.

## What the exhaustive validation proves

The directed corpus enumerates every subset of the 12 non-self directed edges
on four labeled vertices:

```text
2^12 = 4,096 graphs.
```

For every graph, all 12 ordered distinct `(start,target)` pairs were tested
under:

```text
world size 1, 2, 4
x smaller-frontier, alternating
= 6 configurations.
```

Total:

```text
4,096 * 12 * 6 = 294,912 distributed searches.
```

Each result distance was compared with a separate unidirectional BFS. Returned
moves were replayed, and every round checked:

```text
generated = source duplicates + source unique
source unique = local after source dedup + remote after source dedup
source unique = owner duplicates + owner unique
owner unique = already visited + newly discovered.
```

All mismatch/failure counters were zero. This is exhaustive validation for the
named finite simple-graph corpus, not a proof about arbitrary code changes or a
real distributed runtime.

The corpus does not cover:

- self-loops or labeled parallel edges;
- partial/illegal successor generation;
- hash/fingerprint identity;
- asynchronous scheduling or stale replicas;
- overflow, loss, retry, crash, or repartition;
- real transport and device concurrency.

## What S8 adds

The 40 S8 rows cover depths `2,8,14,20,28`, world sizes `1,2,4,8`, and direct
or mixed Lehmer-rank ownership. Each selected target had an independently known
exact distance from complete S8 BFS and each returned word replayed.

S8 adds a finite implicit Cayley graph with relations, convergent successors,
large frontiers, and different ownership locality. It does not expand the
output contract beyond distance plus one path. It mainly validates accounting
and shows where duplicate convergence moves as `P` changes.

## Conservation equations: strength and limit

The per-round equations prove that the simulator's counted stages partition
their input counts. They are useful because a missing category or arithmetic
mistake becomes visible.

Aggregate equality alone would not detect one lost record compensated by one
extra duplicate record. REF-010's stronger distance comparison and path replay
catch many semantic consequences on the exhaustive corpus, while direct Python
data structures make the model lossless by construction. A real runtime would
still need record/epoch identities, per-peer send/receive evidence, capacity
status, and a consistent observation cut.

Thus:

```text
model conservation + finite oracle parity
!= proof of real transport losslessness.
```

## Output-contract matrix

| Note 57 contract | REF-010 status | Evidence boundary |
|---|---|---|
| target distance | **validated exact** | exhaustive four-vertex corpus plus 40 S8 cases |
| target reachability / unreachable | **validated exact** | finite corpus; complete synchronous closure |
| one arbitrary shortest path | **validated exact** | path length agrees with distance and independent replay succeeds |
| canonical shortest path | **not provided** | deterministic implementation winner is not a declared semantic order |
| predecessor DAG | **not provided** | one parent retained; losing same-depth parents discarded |
| shortest-path count | **not provided** | no `sigma` recurrence or contribution identity |
| all shortest paths | **not provided** | no DAG or enumeration |
| uniform sampling | **not provided** | no multiplicity weights or RNG contract |
| multi-source ownership | **not applicable** | one start and one target |
| full distance map | **internal partial maps only** | search stops after target optimality; maps need not cover the component |
| distributed runtime behavior | **not measured** | one-process logical simulation |

The strongest correct summary is:

```text
REF-010 computes an exact target distance and one replayable shortest path in
its bulk-synchronous logical owner-computes model.
```

## Artifact audit

`REF-010-directed-validation.json` records corpus size and failure counters per
configuration. It does not retain:

- graph-by-graph inputs/results;
- selected path identities;
- individual round metrics;
- Python/container/source identity;
- timestamps or command line.

`REF-010-s8-routing.csv` retains per-configuration routing aggregates and target
IDs, but not paths or per-round rows. The Markdown report supplies the intended
protocol and scope. REF-023 adds the Docker image ID, current reproduction, file
hashes, and the corrected newline comparison.

Together they are enough to reproduce and audit aggregate claims, but not to
independently replay any one historical directed-case path from the artifact
alone.

## From simulation to multi-GPU proof obligations

A real implementation matching this model would additionally need evidence
that:

1. every rank expands the agreed side/depth exactly once;
2. every transition and source-local dedup result is capacity-accounted;
3. sends and receives use a consistent epoch and loss/retry protocol;
4. owner-side exact identity sees every routed candidate;
5. forward and reverse owner maps/epochs agree;
6. intersection and incumbent reductions finish before depth minima advance;
7. parent metadata remains reconstructible across owners;
8. global empty means all local queues and in-flight messages are closed;
9. failure/restart does not duplicate non-idempotent output contributions;
10. output validity remains explicitly distance-plus-one-path unless richer
    metadata is built and verified.

These are transfer obligations, not an instruction to implement the runtime.

## Preserved experiment correction

REF-023's first raw `cmp` reported a JSON difference at character two. The
cause was CRLF versus LF. A normalized diff showed identical content for both
regenerated artifacts.

This separates:

```text
byte-identical artifact
normalized textual content
parsed semantic content.
```

The original mismatch is retained because artifact comparisons need to name
which equality relation they claim.

## Current conclusions

1. REF-010 is exact bidirectional BFS inside its bulk-synchronous owner-computes
   model, not merely a distributed-looking heuristic.
2. Its validated output is target distance plus one replayable shortest path.
3. Deterministic reproduction does not make its selected parent path semantic
   or canonical.
4. DAG, counts, all paths, sampling, and full-component distance output are not
   provided.
5. Exhaustive four-vertex validation is strong finite evidence and S8 broadens
   the graph geometry, but neither executes a real transport or GPU.
6. Aggregate conservation is necessary evidence, not sufficient proof against
   compensating loss/duplication in a runtime.
7. REF-023 reproduced all semantic artifacts in Docker; the only initial
   mismatch was newline representation.
8. No implementation or optimization was added.
