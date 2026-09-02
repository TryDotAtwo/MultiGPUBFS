# REF-009: bidirectional expansion policies

Date: 2026-08-27  
Status: pass, with one recorded launch failure

## Question

How do complete-level bidirectional BFS side-selection policies change exact
transition work, and what do they imply for one-GPU and multi-GPU execution?

## Policies

- `smaller_frontier`: expand the side with fewer current frontier states;
- `alternating`: expand forward, reverse, forward, reverse;
- `estimated_work`: sum a supplied nonnegative outgoing-work estimate over each
  frontier and expand the lower-work side.

All policies retain the same correctness rule: expand only complete levels and
stop only when the minimum unexpanded forward and reverse depths sum to at
least the best known meeting distance. Side choice changes work, not the proof.

The result now records the expansion-side trace, permitting round count and
forward/reverse balance to be audited.

## Exhaustive directed corpus

The experiment enumerated all 4,096 directed loop-free graphs on four vertices
and all 49,152 distinct ordered source/target pairs. Exact vertex out-degree or
in-degree was used as the work estimate.

| policy | distance errors | replay errors | transitions | expanded states | rounds |
|---|---:|---:|---:|---:|---:|
| smaller frontier | 0 | 0 | 102,912 | 70,656 | 70,656 |
| alternating | 0 | 0 | 95,232 | 70,656 | 70,656 |
| estimated work | 0 | 0 | 76,416 | 64,608 | 64,512 |

At least two policies generated different work on 19,680 pairs. Estimated work
was tied for best on all 49,152 pairs and was the unique winner on 13,440.
Neither other policy was a unique winner on this corpus.

This does not establish universal dominance. The estimator here is exact and
its acquisition cost is excluded. On a real implicit graph, computing an exact
degree may be as expensive as partial generation; on distributed hardware, a
global comparison also requires communication.

## Regular Cayley graph sweep

The `S8` adjacent-transposition graph is undirected, vertex-transitive, and
7-regular. One deterministic target per depth 0 through 28 was reused from
REF-007 and REF-008. All three policies produced identical transition counts,
expanded-state counts, and side traces at every depth. Selected rows:

| depth | transitions, every policy | rounds | forward rounds | reverse rounds |
|---:|---:|---:|---:|---:|
| 2 | 14 | 2 | 1 | 1 |
| 8 | 1,554 | 8 | 4 | 4 |
| 14 | 17,220 | 14 | 7 | 7 |
| 20 | 77,644 | 20 | 10 | 10 |
| 28 | 255,388 | 28 | 14 | 14 |

Because every frontier state has the same degree, estimated edge work is
exactly proportional to frontier cardinality. Symmetry also makes the smaller
frontier policy alternate sides for these searches. A more complex policy adds
no algorithmic value on this workload.

## GPU implications

### One GPU

- For explicit CSR, degree is cheaply available from row offsets. A frontier
  edge sum still needs a reduction, but predicts edge-parallel work better than
  vertex count on irregular graphs.
- For fixed-generator implicit graphs, frontier count already predicts raw
  transitions. A selector needs some other varying downstream cost to improve
  on it.
- Strict alternation avoids selection overhead and is a credible baseline for
  symmetric regular graphs.

### Multiple GPUs

- A globally correct smaller-frontier or estimated-work decision requires a
  scalar all-reduce of local counts before choosing the next side.
- Strict alternation makes the decision locally known and removes that
  reduction, although frontier completion and distributed intersection still
  require global coordination.
- Reduction latency must be compared with saved generation and communication
  work. CPU transition counts alone cannot choose the policy.
- The initial distributed design should keep a single globally agreed side per
  superstep; independent local choices complicate global depth bounds.

## Reproduction and artifacts

Run from repository root:

```powershell
py -m experiments.run_ref009
```

- `REF-009-directed-summary.json`: aggregate exhaustive-corpus output;
- `REF-009-s8-policy-sweep.csv`: per-depth raw S8 metrics.

The first attempt, `py experiments\run_ref009.py`, failed because Python placed
the `experiments` directory rather than repository root on `sys.path`, so the
local `multigpubfs` package was not importable. No measurements were produced.
Module execution from repository root succeeded.

## Next questions

1. Does exact edge-work selection retain its advantage on larger irregular
   explicit graphs after reduction overhead is included?
2. Can cheap implicit-state features predict post-dedup accepted work better
   than raw generator count?
3. At what per-level work does an extra multi-GPU all-reduce repay itself?
