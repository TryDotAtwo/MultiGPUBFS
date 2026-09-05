# S13 capacity probe v5: search exhausted on physical 2xT4

Kaggle run 347449600, notebook `trydotatwo/mgbfs-s13-compact-capacity-probe`.
Source `0b2acdd5b9e4a6e27af13cc4f0c8c79190f23dcf`.
Native compact permutation DENSE path, owner-local buckets, batch 262144,
220,000,000 capacity/state-ring records per rank. Archive disabled.

| Metric | Observed |
|---|---:|
| Search time (maximum rank) | 3583.067481444 s |
| Sum of global layer counts | 6,227,020,800 = 13! |
| Nonempty depths | 0 through 78 |
| Largest global frontier, depth 54 | 369,741,101 |
| Largest rank-0 frontier | 184,883,067 |
| Largest rank-1 frontier | 184,858,034 |
| nvidia-smi peak per device | 13,985 MiB |
| nvidia-smi peak summed | 27,970 MiB |
| CUDA observed used bytes per device | 14,664,204,288 |

Both rank summaries report COMPLETE; launcher exit code is zero. Final
global frontier contains one state and its expansion reports no survivors.
Rank totals are 3,113,547,204 and 3,113,473,596.

This rules out capacity exhaustion for this configuration and confirms the
previous 1800-second timeout was insufficient. It is a single search-only
sample, not an archive-throughput result or an A/B speedup claim. The
durable time field is not meaningful as archive durability with archive OFF.

Equality remains probabilistic Hash128 equality. Cardinality agreement
alone is not independent full-state/layer-set validation. No state archive
was produced, so this run must NOT be published as a full HF graph dataset.
Streaming archival still needs a separate validated run after resolving its
pinned-ring throughput failure. S14 capacity remains unverified.

Local evidence: `test_results/s13-owner-local-v5/s12-capacity-probe/`:
`summary.json`, `s13-capacity-220000000.json`, rank summaries, source SHA,
depth log and external GPU sampler. Raw logs are not committed.
