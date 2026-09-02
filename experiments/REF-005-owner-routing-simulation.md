# REF-005: owner-routing and cross-rank duplicate simulation

Date: 2026-08-27.

## Question

When an exact BFS frontier is partitioned across ranks, how much duplicate work
can each source rank remove locally, how much remains cross-rank, and how does
the owner function trade load balance against communication locality?

## Model

The exact `S8` frontiers from REF-002 are treated as immutable ground truth.
Every state is assigned to one source/owner rank. For each level and source rank:

1. expand its owned frontier states;
2. exact-deduplicate its local candidate batch;
3. route each unique state to its authoritative owner;
4. exact-deduplicate the union at each owner;
5. subtract cumulative exact visited;
6. compare the accepted union with the next reference frontier.

No communication or GPU time is modeled. Counts represent payload opportunities,
not measured performance.

## State identity and owner functions

Permutation identity uses exact Lehmer rank in `[0, 8! - 1]`.

Two owner functions are compared:

```text
rank_mod:    owner = LehmerRank(state) % P
mixed_rank: owner = SplitMix64(LehmerRank(state)) % P
```

The mixer changes partitioning only. Exact state equality remains permutation
equality.

## First simulation: direct rank modulo

| ranks | raw generated | sum local unique | locally removed | cross-rank removed at owners | owner unique | remote payload | remote fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 282,240 | 80,638 | 201,602 | 0 | 80,638 | 0 | 0.000000 |
| 2 | 282,240 | 134,342 | 147,898 | 53,704 | 80,638 | 53,760 | 0.400173 |
| 4 | 282,240 | 197,040 | 85,200 | 116,402 | 80,638 | 118,272 | 0.600244 |
| 8 | 282,240 | 214,524 | 67,716 | 133,886 | 80,638 | 136,416 | 0.635901 |

`owner unique` is invariant because owner routing reunites all exact candidates.
As source-rank count rises, candidate duplicates are split across more ranks, so
local exact dedup removes fewer occurrences and owner-side dedup removes more.

## Owner strategy comparison

Imbalance is `max_rank_count / mean_rank_count`. `max large-level imbalance`
excludes tiny frontiers and considers levels with at least `16 * P` states.

| strategy | ranks | sum local unique | cross-rank removed | remote fraction | max large-level frontier imbalance | peak-level imbalance | final visited imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank modulo | 2 | 134,342 | 53,704 | 0.400173 | 1.447368 | 1.000000 | 1.000000 |
| rank modulo | 4 | 197,040 | 116,402 | 0.600244 | 1.684211 | 1.004171 | 1.000000 |
| rank modulo | 8 | 214,524 | 133,886 | 0.635901 | 2.114943 | 1.032325 | 1.000000 |
| mixed rank | 2 | 143,988 | 63,350 | 0.500278 | 1.105263 | 1.007821 | 1.000893 |
| mixed rank | 4 | 200,742 | 120,104 | 0.750426 | 1.263158 | 1.034411 | 1.006448 |
| mixed rank | 8 | 237,940 | 157,302 | 0.875729 | 1.287356 | 1.038582 | 1.017857 |

## Observations

**Direct rank modulo:** Final visited ownership is perfectly balanced because
the complete rank interval divides evenly by 2, 4, and 8. Per-level balance is
not guaranteed: at 8 ranks, a large level reaches 2.11 times mean load.

**Mixed rank:** Large-level imbalance improves markedly, to at most 1.29 in this
experiment. Final ownership remains close to balanced rather than exact.

**Communication locality:** Mixed ownership makes source and destination owners
behave nearly independently, so remote fractions approach `1 - 1/P`. Direct
rank modulo preserves correlations between adjacent permutations and cuts the
remote fraction, especially at 8 ranks.

**Local dedup:** Better mixing distributes parents more evenly, but duplicates
of their children also arise on more source ranks. Consequently fewer duplicate
occurrences meet during source-local dedup and more survive to owner-side dedup.

## Insights

1. Exact numeric state rank and partition key have different requirements.
   Bijection proves identity; it does not prove balanced low bits at every BFS
   level.
2. An avalanche hash is not unconditionally better. It buys balance at the cost
   of locality and additional remote/cross-rank payload.
3. Total final owner balance is a weak metric. BFS must measure every level,
   especially peak bytes and slowest-rank time.
4. More GPUs can reduce the effectiveness of pre-network local dedup even before
   hardware overhead is considered.
5. Owner selection may need workload-aware partitioning or a controlled choice
   between locality and balance, rather than unconditional hash modulo.

## Methodology corrections and failure

The first command failed before execution because nested PowerShell/Python
quoting left an `f-string` parenthesis open. It produced no measurements. The
corrected command used `str.format`.

During report reconciliation, a later combined comparison printed `0.400278`
for the two-rank direct-modulo remote fraction, inconsistent with the first
table. An independent recomputation asserted cached and direct Lehmer ranks for
every state and reproduced `53,760 / 134,342 = 0.400172694`; the table now uses
that verified value. The mixed two-rank fraction is separately
`72,034 / 143,988 = 0.500277801`. This demonstrates why derived ratios must be
retained with their integer numerator and denominator.

The first imbalance summary also used the maximum across all levels. That value
was dominated by the single-state initial frontier and trivially equaled `P`.
The reported comparison adds a large-level threshold and separately reports the
peak-frontier level.

## Limitations

- No bounded send buffers, chunking, collective ordering, or topology.
- Exact Python sets approximate ideal per-level dedup.
- Lehmer ranking cost is excluded from any timing model.
- Only one graph/generator family is simulated.
- The source owner and authoritative owner use the same function; replicated or
  work-stealing frontiers are not modeled.

## Next experiment

Evaluate owner functions by a multi-objective score over every level:

- maximum candidates received by any owner;
- remote bytes;
- cross-rank duplicates;
- accepted-state balance;
- persistent visited balance.

Candidate strategies include salted/mixed rank, range partitioning, higher-bit
partitioning, and two-choice ownership with an exact deterministic rule.
