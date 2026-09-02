# REF-002: adjacent-transposition Cayley graphs of symmetric groups

Date: 2026-08-27.

## Question

What frontier growth and duplicate pressure does exact implicit BFS encounter
on small, fully enumerable Cayley graphs?

## Graph family

The vertices are permutations of `(0, ..., n-1)`. The ordered generator set is
the `n-1` adjacent transpositions:

```text
s_i swaps positions i and i+1
```

Starting from the identity, these generators produce the full symmetric group
`S_n`. The known number of vertices is `n!`. The distance from the identity is
the inversion count, so the maximum distance is `n(n-1)/2`.

These facts provide independent checks of vertex count and BFS depth.

## Implementation and semantics

- Complete deterministic level-synchronous traversal.
- Tuple permutation states, exact Python dictionary membership.
- Generator label and parent state retained for every discovered state.
- Five timed repetitions per `n`; the reported time is the median.
- The semantic validator runs after the timed repetitions and is not included in
  the reported traversal time.

`generated` counts every attempted generator application. `non_tree_ratio` is:

```text
(generated - (vertices - 1)) / generated
```

Every non-root state has exactly one tree-discovery transition, so the remaining
transitions are duplicates or transitions to already visited states.

## Environment

```text
Python 3.11.5
Windows 10.0.26200
Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
16 logical CPUs reported by Python
```

The experiment is single-threaded Python. It is not a CPU optimization
benchmark and says nothing directly about GPU throughput.

## Results

| n | vertices | generators | generated | non-tree ratio | max depth | peak frontier | median seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 6 | 2 | 12 | 0.583333333 | 3 | 2 | 0.000007500 |
| 4 | 24 | 3 | 72 | 0.680555556 | 6 | 6 | 0.000028600 |
| 5 | 120 | 4 | 480 | 0.752083333 | 10 | 22 | 0.000181000 |
| 6 | 720 | 5 | 3,600 | 0.800277778 | 15 | 101 | 0.001078700 |
| 7 | 5,040 | 6 | 30,240 | 0.833366402 | 21 | 573 | 0.013393900 |
| 8 | 40,320 | 7 | 282,240 | 0.857146400 | 28 | 3,836 | 0.109180000 |

## Correctness evidence

- Observed vertex count equals `n!` for every row.
- Observed maximum depth equals `n(n-1)/2` for every row.
- The labeled BFS validator returned no errors for every row.
- The `S3` test checks literal frontiers, a literal shortest move sequence, and
  replay to the target state.
- A negative test corrupts one parent move while keeping parent/depth metadata
  intact; the validator detects that the move cannot produce the child.

## Observations and inferences

**Observation:** The peak frontier is much smaller than the full visited set for
these small instances, but grows rapidly: `2, 6, 22, 101, 573, 3836`.

**Observation:** The non-tree ratio tends toward one as `n` grows in this family.
For a connected `d`-regular graph traversed exhaustively, exactly `V-1` of the
`dV` directed generator applications discover tree vertices, so the ratio is
exactly:

```text
1 - (V - 1) / (d V)
```

The measured values follow this identity. This is not an implementation defect;
it is an accounting consequence of exhaustive BFS.

**Inference:** Optimizing only successful visited insertions misses most of the
steady work. A high-performance implementation must make negative/duplicate
membership checks cheap and should eliminate same-level duplicates before
expensive storage or communication when that elimination costs less than the
traffic it saves.

**Unknown:** The experiment does not separate same-frontier duplicates from
edges to earlier levels. That split matters because same-frontier dedup can be
performed before authoritative global visited lookup.

## Failure and instrumentation limitation

`Get-CimInstance Win32_Processor` failed with `Access denied`, so the exact CPU
marketing name was not captured. No escalation was needed because the timing is
diagnostic only. Processor family/model and logical CPU count were captured
through environment/Python fallbacks.

## Next experiment

Instrument each depth separately:

- generated transitions;
- duplicates within the candidate batch;
- candidates already present in previous levels;
- accepted states;
- frontier width and cumulative visited size.

This decomposition will directly inform whether pre-network sort/unique is
likely to repay its cost in a multi-GPU design.
