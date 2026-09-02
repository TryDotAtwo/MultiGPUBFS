# REF-008: target-stop granularity

Date: 2026-08-27  
Status: pass

## Question

How much exact unidirectional target BFS work depends on when an implementation
is able to stop after discovering the target?

## Semantics implemented

`multigpubfs.target_search.target_breadth_first_search` supports three exact
shortest-path termination granularities:

- `candidate`: stop immediately when the target candidate is generated;
- `parent_batch`: finish a fixed-size batch of frontier parents;
- `level`: finish the complete current frontier level.

All three return the same shortest distance and a replayable path. Candidate and
batch termination deliberately do not claim that the target-depth layer is
complete. Their partial visited/frontier state therefore cannot be treated as a
complete bounded-depth BFS result.

## Correctness validation

Every directed loop-free graph on four vertices was enumerated. For every one
of 49,152 distinct ordered source/target pairs, each of the three modes was
compared with the complete BFS oracle and its returned path was replayed.

- distance/found mismatches: 0 for every mode;
- replay failures: 0 for every mode;
- unreachable pairs were included.

Aggregate generated-transition counts were 92,032 for candidate stop and
115,200 for both batch-size-2 and level stop. The equality of the last two is a
property of these tiny test frontiers, not a general result.

## S8 sweep

The adjacent-transposition Cayley graph of `S8` was searched from identity to
the same deterministic one-target-per-depth sample used by REF-007. Raw counts
are in `REF-008-s8-stop-granularity.csv`.

Selected results:

| depth | candidate | batch 32 | batch 256 | batch 1024 | level | bidirectional |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9 | 56 | 56 | 56 | 56 | 14 |
| 14 | 101,544 | 101,766 | 103,334 | 108,710 | 127,694 | 17,220 |
| 28 | 282,185 | 282,233 | 282,233 | 282,233 | 282,233 | 255,388 |

At depth 14, a 32-parent batch adds only 222 transitions (about 0.22%) over
candidate stop for this target and ordering, while level completion adds 26,150
(about 25.75%). Bidirectional BFS generates about 83.04% fewer transitions than
candidate-stop unidirectional BFS there. At depth 28, level completion adds only
48 transitions and bidirectional BFS saves about 9.50% versus candidate stop.

At shallow depths the result reverses: depth-2 bidirectional search generates
14 transitions versus 9 for candidate-stop unidirectional search. For the
chosen samples, the crossover occurs between depths 3 and 4.

## Interpretation

- Target order within a frontier can change candidate-stop work enormously.
  These targets were selected deterministically from the reference traversal,
  so this is not a target-distribution benchmark.
- Candidate stop is a useful sequential lower-work reference, but a massively
  parallel GPU cannot generally cancel every already-issued transition at that
  granularity. Kernel, workgroup, queue, or parent-batch granularity is the more
  realistic design variable.
- The previous REF-007 unidirectional baseline is confirmed as actual
  level-complete target BFS, rather than only an inferred complete-level count.
- Bidirectional and unidirectional comparisons must state termination
  granularity. Level-complete unidirectional work can materially overstate the
  advantage of bidirectional search at some depths.

## Next experiments

Sample many or all targets within each distance layer; measure distributions,
not a single order-biased target. Then compare bidirectional expansion policies
and model GPU stop latency in units of issued work rather than pretending that
candidate-level cancellation is free.
