# REF-007: exact bidirectional BFS

Date: 2026-08-27.

## Question

Can a smaller-frontier bidirectional BFS preserve exact shortest distances and
replayable generator paths, and how much transition work does it save across
different target depths?

## Algorithm contract

The forward API returns `(move, child)`. The reverse API returns
`(forward_move, predecessor)` such that:

```text
apply_forward(predecessor, forward_move) == current_reverse_state
```

Each side expands complete levels. The smaller current frontier is selected.
After an intersection gives a candidate path of length `best`, traversal stops
only when:

```text
minimum_unexpanded_forward_depth
+ minimum_unexpanded_reverse_depth
>= best
```

This is the BFS specialization of the bidirectional shortest-path lower bound.

## Path construction

- Forward records point from a state to its predecessor from start.
- Reverse records point from a predecessor to the next state toward target.
- A meeting state joins the reversed forward prefix and forward-oriented reverse
  suffix.
- Reconstructed move count must equal the best meeting distance.

## Exhaustive directed-graph validation

All `2^12 = 4,096` directed loop-free graphs on four labeled vertices were
enumerated. For every graph, all 12 ordered pairs with distinct endpoints were
checked against an independent queue BFS.

```text
graphs              4,096
ordered pairs       49,152
reachable pairs     36,864
unreachable pairs   12,288
distance mismatches 0
replay failures     0
```

This covers asymmetric reachability, dead ends, unequal frontier growth,
multiple shortest paths, and absent paths. It is strong finite evidence for the
stopping and reverse-path contracts, but not a formal proof.

## Exhaustive S4 validation

The adjacent-transposition Cayley graph `S4` has 24 states and three involutive
generators. All 576 ordered pairs, including identical endpoints, were compared
with complete single-source BFS.

```text
distance mismatches 0
replay failures     0
maximum distance    6
bidirectional generated transitions: min 0, median 15, mean 17.5, max 54
bidirectional expanded states:       min 0, median 5,  mean 5.8333, max 18
complete BFS transitions per source: 72
```

## S8 depth sweep

One deterministic target (the first reference-frontier state) was selected at
every distance 0 through 28. The comparison baseline expands all complete
unidirectional levels before the target level:

```text
unidirectional transitions = degree * cumulative frontier size at depths < d
```

The raw table is in
[`REF-007-s8-depth-sweep.csv`](REF-007-s8-depth-sweep.csv).

Selected rows:

| target depth | bidirectional generated | unidirectional level-complete | reduction |
|---:|---:|---:|---:|
| 2 | 14 | 56 | 75.00% |
| 8 | 1,554 | 15,337 | 89.87% |
| 14 | 17,220 | 127,694 | 86.51% |
| 20 | 77,644 | 256,998 | 69.79% |
| 24 | 154,784 | 280,245 | 44.77% |
| 28 | 255,388 | 282,233 | 9.51% |

## Observations and insights

1. The largest relative savings occur before and around the middle of the
   diameter, not at the farthest target.
2. For the reverse permutation at depth 28, both directions visit 22,078 states
   and together expand 36,484 states. The two balls cover most of the finite
   graph before the stopping bound closes.
3. The common intuition `O(b^(d/2))` assumes roughly tree-like growth. Relations,
   finite-group saturation, and overlapping search balls can make it a poor
   quantitative predictor.
4. Expanding the smaller frontier controls immediate work but does not remove
   the need for an exact global stopping bound.
5. Reverse generator semantics are part of correctness, not an implementation
   detail. An incorrectly labeled inverse can return a correct distance with an
   unreplayable path.

## Baseline limitation

The S8 unidirectional baseline is level-complete. A sequential implementation
could stop during the generating level when it first sees the chosen target;
parallel GPU implementations may finish a batch or level. Future comparisons
must state target-detection granularity. The reported reduction is therefore
relative to the declared level-complete semantics only.

Only one target per depth was sampled. Cayley vertex transitivity suggests
substantial regularity, but target-order effects on deterministic meeting and
work counts were not exhaustively measured for `S8`.

## Next experiment

- Add an exact target-stopping unidirectional reference with explicit
  per-candidate, per-batch, and per-level stop semantics.
- Compare bidirectional policies: strict alternation, smaller frontier, and
  estimated outgoing work.
- Model distributed bidirectional ownership and the cost of detecting frontier
  intersections across ranks.
