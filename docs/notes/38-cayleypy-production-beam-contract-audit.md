# CayleyPy production beam: read-only BFS contract audit

This note applies the exact-BFS contract map to one concrete search pipeline.
It is an interpretation exercise, not an implementation or optimization plan.

## Snapshot and scope

The inspected checkout was `D:\100XH100` at commit
`b5fcf6b0ee3247a1f189e6da79a1641d2304bd1c`, branch `main`, on 2026-08-28.
The checkout was dirty and two commits ahead of `origin/main`; in particular,
`tools/production_runner.cu` and runtime configuration files had local changes.
Therefore every statement below describes that observed working tree, not an
immutable upstream release.

The traced path was:

```text
retained frontier at depth d
  -> generate every configured move from every retained parent
  -> hash and inspect goal / goal-neighborhood hits
  -> score threshold and within-depth Hash128 deduplication
  -> global beam selection
  -> materialize retained frontier at depth d+1
```

Primary inspected files were:

- `tools/production_runner.cu`, especially the depth loop and finalization;
- `cuda/stream2.cu`, goal and solved-neighborhood detection;
- `cuda/dispatcher.cu`, final threshold and global keep selection;
- `src/stream3.cpp` and `src/stream4.cpp`, readable CPU counterparts of
  thresholding, deduplication, routing, and tie choice;
- `src/frontier_cpu.cpp`, the small CPU depth reference;
- `src/hash.cpp`, `src/hash.hpp`, and `src/types.hpp`, state-key semantics.

## What one depth means

`production_runner.cu` initializes the frontier with one state on rank zero and
runs a loop indexed by `depth`. Before dispatch it writes
`current_solution_depth = depth + 1`. After the depth drains, finalization
materializes the selected states and makes their count the next
`frontier_size`.

Thus the runner is layered: every materialized frontier state has a retained
move history of the corresponding length. Layering alone does not establish
BFS completeness. The mathematical recurrence observed is closer to

```text
C_(d+1) = successors(K_d)
U_(d+1) = dedup_Hash128(score_eligible(C_(d+1)))
K_(d+1) = globally_selected_beam(U_(d+1)).
```

It is not the exact-BFS recurrence

```text
F_(d+1) = unique(successors(F_d)) minus B_d.
```

The difference is both the bounded selection and the absence, in the traced
pipeline, of an accumulated old ball `B_d` used for rejection.

## Candidate generation and goal inspection

`stream2_hash_goal_kernel` assigns one candidate to each retained
`(parent, move)` pair. It constructs the child state implicitly through the
generator permutation, computes its Zobrist `Hash128`, and checks one of:

1. exact byte-wise equality with the central state when no solved neighborhood
   is enabled;
2. membership of the child hash in the solved-neighborhood table;
3. optional short suffix expansion followed by either of those checks.

This happens before score thresholding and final beam selection. Consequently,
a valid target path can be reported even if that target candidate would later
have been discarded by score or width. This expands the target-inspection set
for the current retained parents; it does not restore parents pruned at earlier
depths.

For a hit, the kernel records parent, move, generated depth, hash, and optional
suffix identifier. The runner reconstructs the prefix from per-depth history,
appends suffixes, applies the complete move sequence on the CPU, and requires
the final materialized state to equal the target before accepting the artifact.
That replay is strong evidence for path validity, but a valid path is only an
upper bound on the original-graph distance.

## What deduplication means here

The readable Stream 3 reference first removes candidates above the current
score threshold, sorts by `(Hash128, score/payload)`, and retains the first
record for each `Hash128`. It then routes each surviving key to its owner rank.
The Stream 4 reference again threshold-filters, sorts by
`(Hash128, candidate_better)`, and retains one record per key.

`candidate_better` orders by:

```text
score_key, parent_idx, route_packed.
```

This is same-depth candidate convergence. A repository-wide search over the
traced source directories found no accumulated `visited`, closed set,
distance-label table, or explicit subtraction of earlier frontiers. History is
used to reconstruct retained paths, not as a membership filter. Therefore a
state reached at an earlier depth may be generated and retained again later.

The identity key is a deterministic 128-bit Zobrist XOR, not an injective state
rank and not a collision-resolving state table in the inspected code. Equal
hashes are treated as equal during candidate deduplication. The goal-centered
lookup uses a 32-bit fingerprint for bucket screening and then confirms the
full `Hash128`, but it still does not compare full `State128` bytes on device.
CPU solution replay prevents a colliding hit from becoming a silently accepted
path artifact; it does not make hash-based pruning mathematically collision-free.

## Beam selection and multi-GPU meaning

Finalization obtains a score threshold from the global score histogram. In the
multi-rank branch it counts all candidates below the threshold and at the
threshold across ranks, then defines

```text
global_keep_count = min(global_beam_width_effective, total_available).
```

It deterministically takes all globally lower-score records and only the
required prefix of equal-score records, then load-balances the selected global
set across ranks for the next frontier. This is evidence that the declared
width is global rather than an independent local top-k quota.

That distinction makes the multi-GPU beam less partition-dependent than naïve
per-rank quotas. It does not turn the selected set into a complete BFS layer:
whenever eligible unique candidates exceed `global_beam_width_effective`, some
states are intentionally absent from the next frontier.

## Stopping semantics

In ordinary mode, a hit can set `stop_flag`; after the whole dispatched depth
drains, ranks propagate the flag, select the best stored hit for that depth,
reconstruct it, validate it, and stop. Thus this is not a mid-kernel claim that
an arbitrary racing thread alone determines the returned record.

Nevertheless, the stopping proof is relative to the surviving beam history.
Earlier width/score pruning may have removed a shorter or the only solution
branch. The runner compares recorded total lengths among collected hits in the
current depth; capacity can omit other candidates' hits. For a fixed candidate,
an exact complete K1 ball with shortest suffixes and exhaustive length-ordered
K2 words including the empty word does make first hit a shortest residual
(note 40). That local theorem neither restores pruned outer prefixes nor proves
the actual hash-only tables exact, so it does not certify global shortestness.

`solve_bucket_mode` deliberately weakens immediate stopping: it records all
available hits, resets solved buffers, continues building frontiers, and stops
after a configured number of extra depths. This is a data-collection policy,
not an exact shortest-path proof for the original graph.

## Contract classification

| Contract layer | Observed semantics | Consequence |
|---|---|---|
| graph | implicit fixed-generator state graph | one move is one prefix edge |
| schedule | depth-synchronous retained frontiers | retained histories have well-defined prefix depth |
| identity | bare deterministic `Hash128` for candidate uniqueness | probabilistic semantic identity, not exact state equality |
| old-ball exclusion | none found in traced path | layers are not first-discovery spheres |
| candidate completeness | all configured moves of retained parents | complete only relative to the retained beam |
| frontier retention | global score threshold plus bounded beam | intentionally pruned when width binds |
| multi-GPU selection | global keep count, then load balancing | avoids naïve local-quota semantics |
| target witness | prefix history plus lookup suffix, CPU replayed | validates a path, not its global optimality |
| stopping | after a drained generated depth, or bucket policy | no original-graph BFS lower-bound certificate |

The most accurate short name for this observed pipeline is a
**layer-synchronous, globally selected, hash-deduplicated beam search with an
exactly replay-validated goal-neighborhood suffix**. The local neighborhood may
be built by BFS, but that component does not make the outer search BFS.

## Confirmed, inferred, and still unknown

Confirmed from the inspected source:

- expansion is breadth-like by retained prefix depth;
- goal inspection precedes score/beam pruning for the current depth;
- candidate duplicates are merged by `Hash128`;
- the next frontier is bounded by a global effective beam width;
- accepted paths are replayed against the full target state on CPU;
- no accumulated visited-set operation appeared in the traced pipeline.

Inferred with explicit limits:

- because earlier states are not subtracted, cyclic Cayley relations can
  reintroduce states at later depths;
- a larger beam can approach a complete layer for particular early depths, but
  only if no other threshold/capacity/key loss occurs and every eligible unique
  state fits;
- changing world size should preserve the intended global score-width rule,
  although equal-score tie ordering and full parity require a runtime check.

Still unknown from this read-only pass:

- whether the Zobrist map is proved injective for any specific puzzle domain;
- exact parity of single- and multi-GPU survivor sets under all score ties;
- which production configurations first bind width, score threshold, or other
  capacities for each puzzle;
- whether external wrappers use “BFS” only for the goal-neighborhood builder or
  also as an imprecise name for the outer beam;
- what shortest-path claim, if any, downstream result consumers attach to a
  replay-valid solution.

## Main conceptual lesson

The code makes the distinction unusually concrete:

> A frontier indexed by depth tells us how long the retained prefixes are. It
> does not tell us that the frontier contains every state at that graph
> distance.

Exact BFS needs complete first-discovery layers and an exact old-ball exclusion
contract. This pipeline instead spends bounded capacity on promising retained
states and uses a local exact suffix mechanism to find valid solutions. Both
ideas are coherent, but their guarantees must remain separate.
