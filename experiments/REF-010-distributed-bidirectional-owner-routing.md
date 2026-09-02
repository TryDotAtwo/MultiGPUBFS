# REF-010: distributed bidirectional owner routing

Date: 2026-08-27  
Status: pass

## Question

Can an owner-computes distributed bidirectional BFS preserve exact paths and
the global stopping proof, and where do duplicates and communication move as
rank count and ownership change?

## Simulated superstep

For one globally selected direction and one complete BFS level:

1. Each owner expands its locally owned frontier states.
2. It deduplicates equal candidate states generated on that source rank.
3. Remaining candidates are routed to `owner(candidate)`.
4. The destination owner performs cross-source dedup and authoritative lookup
   in that direction's visited set.
5. A newly accepted state is checked against the opposite visited set, which is
   local because both directions use the same owner function.
6. The next frontier and any meeting distance become globally visible at the
   level boundary; the usual sum-of-minimum-unexpanded-depths bound decides
   termination.

The simulator retains parent/move metadata and reconstructs a replayable path.
Each round has lossless accounting equations for generation, source pre-dedup,
routing, owner dedup, visited rejection, and acceptance.

## Exhaustive correctness evidence

All 4,096 directed loop-free graphs on four vertices and all 49,152 distinct
ordered pairs were run under each of six configurations:

- world sizes 1, 2, and 4;
- smaller-frontier and alternating side policies;
- deterministic `state mod P` ownership.

This is 294,912 distributed searches. Every configuration produced:

- 0 distance/found mismatches;
- 0 path replay failures;
- 0 round-accounting failures.

This validates the finite corpus, not an asynchronous implementation. The model
is deliberately bulk-synchronous.

## S8 routing experiment

The adjacent-transposition `S8` targets at depths 2, 8, 14, 20, and 28 were
searched with `P = 1, 2, 4, 8`. Ownership used either direct Lehmer rank modulo
`P` or a SplitMix-style avalanche of that rank before modulo. Alternating sides
were used because REF-009 showed identical search work on this graph.

Selected 8-rank results:

| depth | owner | generated | removed before route | remote after pre-dedup | removed at owner | peak-round skew |
|---:|---|---:|---:|---:|---:|---:|
| 14 | direct | 17,220 | 4,031 | 8,015 | 7,553 | 1.648x |
| 14 | mixed | 17,220 | 2,481 | 12,888 | 9,103 | 1.103x |
| 28 | direct | 255,388 | 61,218 | 123,388 | 121,004 | 1.060x |
| 28 | mixed | 255,388 | 40,082 | 188,615 | 142,140 | 1.099x |

At depth 14, direct ownership sends 37.81% fewer post-pre-dedup remote
candidates than mixed ownership, at the cost of substantially worse peak-round
balance. At depth 28 it sends 34.58% fewer, while the widest round is already
well balanced for both mappings. Across rounds with at least 128 frontier
states, the maximum depth-28 skew was 2.115x for direct versus 1.287x for mixed;
the single widest round alone hides that earlier imbalance.

As `P` increases, a duplicate state is more likely to be produced on multiple
source ranks. Work therefore migrates from cheap source-rank pre-dedup to
destination-owner convergence. For depth 28/direct ownership, source pre-dedup
removed 182,222 occurrences at `P=1`, 133,656 at `P=2`, 76,962 at `P=4`, and
61,218 at `P=8`. The lost local convergence reappears as 0, 48,566, 105,260,
and 121,004 owner-side duplicate occurrences respectively.

All 40 S8 configurations found their first intersection in their final search
round. The number of newly discovered meeting states in that round grew from 1
at depth 2 to 3,836 at depth 28. The level-complete model does not attempt to
cancel within the meeting superstep.

## Architectural implications

- Both directions should initially share exactly the same owner mapping. Then
  intersection is a local lookup after routing rather than a second distributed
  join.
- Pre-routing dedup must be measured per source rank, not globally. A global CPU
  `unique` exaggerates network savings available before communication.
- More ranks do not reduce generated transitions. They change where duplicate
  convergence occurs and generally increase the fraction requiring exchange.
- A locality-preserving rank can materially reduce network traffic but should
  be judged by per-round imbalance, not final ownership or only the widest
  frontier.
- The simulated global level boundary supplies a clean shortest-path proof.
  Relaxed/asynchronous processing needs explicit epoch and termination logic
  and must not inherit this proof implicitly.
- Network cost should next be reported in bytes, including state key, side,
  depth/epoch, and parent reconstruction metadata; candidate counts alone are
  representation-independent but not a throughput prediction.

## Reproduction

From repository root:

```powershell
py -m experiments.run_ref010
```

Artifacts:

- `REF-010-directed-validation.json`;
- `REF-010-s8-routing.csv`.

## Limitations and next experiment

The simulator has no latency, bandwidth, topology, device memory, kernel, or
collective cost. It proves accounting and exposes traffic volume only. Next,
define wire records and a byte-level communication model, then compare one- and
two-phase exchange and parent-storage strategies before implementing GPU code.
