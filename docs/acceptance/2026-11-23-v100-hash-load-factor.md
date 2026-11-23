# Packed resident hash load: opt-in experiment

The environment setting `MGBFS_RESIDENT_HASH_LOAD_PERCENT` selects 50–75 for the packed resident hash segment layout; unset keeps the previous 50 setting. It is an allocation target for the maximum physical rows, rounded up to a power-of-two slot count. It does not change state equality or the per-segment `2^32` configured row maximum. A denser table saves index bytes but can increase probe cost. Do not make it the default based on these single observations.

## 4×V100 evidence

Four Tesla V100-SXM2-32GB, same resident multi-CTA segmented24 executable SHA-256 `70114fdb4b93d2b07e2679e0d09f6c084408ab2d74da146b84d22c5ebc3f334c` and CUDA library SHA-256 `b51269b1d1e12b19d71cc04e288e7bad6bf44694ebaec3e6938f620835601345`. Tested source tree `41d62fa7a255a8cd1345b5082c54d1be260b30d8`, original source commit `e446cfd7a7afc44fbff3221194a0f2203ecf1f1d`. Four source changes are applied over the subsequently published baseline documentation in this branch. GPU stage metrics and compact positions were enabled; each case below is one observation.

| Graph | Load | Status | BFS s | NVML worst sampled MiB | Rank 0 segment storage bytes |
|---|---:|---|---:|---:|---:|
| S12 adjacent transpositions | 50 | COMPLETE, depth 66 | 19.428 | 5,570 | 3,299,408,384 |
| S12 adjacent transpositions | 75 | COMPLETE, depth 66 | 21.612 | 4,546 | 2,225,666,560 |
| S13 adjacent transpositions | 50 | COMPLETE, depth 78 | 349.952 | 28,242 | 27,262,982,240 |
| S13 adjacent transpositions | 75 | COMPLETE, depth 78 | 567.036 | 19,922 | 18,538,829,920 |

S12 covers 479,001,600 states; S13 covers 6,227,020,800 states. Every recorded layer count matched the Mahonian coefficients and the final reverse permutation was checked. These large runs did not compare the complete intermediate state sets. The allocation numbers cover segment states, hash index and counters, not all GPU allocations. NVML was sampled at 0.5 s and may miss short peaks.

## Equal-budget S13 pressure

With an identical segment-storage budget of 9,000,000,000 bytes per GPU/rank, load 50 stopped at depth 28 with 636,298,069 completed states; load 75 stopped at depth 35 with 2,102,968,914 completed states. Both reported `INCOMPLETE / RESIDENT_RESOURCE_CAPACITY`, and the published completed layer counts matched Mahonian values. The rank-0 segment storage receipts were 8,999,995,852 and 8,999,993,936 bytes respectively. This budget excludes transport, queues, CUDA/NCCL and other overhead. These prefixes end at different depths, so their total elapsed times are not a valid speed comparison.

The complete load and pressure archives on the compute host are `multicta-hash-load-retry-evidence.tgz` (SHA-256 `6358483dbe572f927b3126354203d01ef3185d741b76b235cb99c3528d27c17e`) and `multicta-hash-load-pressure-retry-evidence.tgz` (SHA-256 `c0ed7af31c0dca536aeb875ec41a4fe44f28b76c18db2d2a1a28a07763f57c00`).

The small 4-GPU oracle compares exact per-layer state multisets for S4 during 50 → 75 → 50 persistent jobs, including a resource stop followed by a healthy job. S6 with ordinary and deliberately colliding hashes also passed full per-layer checks. These do not replace a full large-state-set, 8-GPU or B200 check.

The compressed machine-readable comparisons of the earlier snapshot are alongside this report. The evidence snapshot containing these completed cases and test logs is `multicta-hash-load-midpoint.tgz`, SHA-256 `c7dd12932c81e29200813d3704f1482ca768204b7c34d9606b0ae9fbc782f39c`. State/index allocation receipts and sampled NVML values are different measurements and must not be combined into a claimed exact peak.

## S13 heavy-layer timing and useful payload

For the same single S13 observations, source depths 34–44 contain 3,052,643,942 parents and 36,631,727,304 generated edges, producing 3,034,361,132 states at depths 35–45. Summing the slowest recorded rank's per-layer GPU intervals gives approximately 184.098 s at load 50 and 321.331 s at load 75. This yields approximately 16.48 million versus 9.44 million newly completed states/s, and 198.98 million versus 114.00 million generated edges/s, respectively. The rank-0 receive/dedup/commit interval total was 156.229 s versus 282.754 s; its generation/partition total was 14.363 s versus 14.241 s.

These are instrumented GPU-clock intervals, not a direct global wall-clock measurement; the different GPUs' intervals cannot be summed as serial time. The whole S13 traversal reported rank-0 remote useful payload of 224,176,678,528 bytes in 14,011,042,408 16-byte records. This excludes protocol/header overhead and is not measured physical link traffic. All figures are single observations on these 4×V100 GPUs; they do not establish B200 or 8-GPU performance.
