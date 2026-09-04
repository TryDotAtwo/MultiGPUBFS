# S12 capacity boundary on 2xT4

Status: **REPRODUCIBLE CAPACITY FAILURE**

Kaggle notebook `trydotatwo/mgbfs-s12-capacity-probe`, version 1, used runtime
source `0acba04315ac41e58d286ee08052b09ecd858f04`. It ran the exact native
two-rank DENSE/CUB search with `batch=262144`, no archive payload submission,
and `MAX_PER_RANK` capacities of 14M, 15M and 16M records. The no-archive mode
uses a logical sink, so the 479,001,600-state output contract does not reserve
77+ GiB of Kaggle disk during this capacity-only probe.

| capacity per rank | result | peak VRAM per T4 |
|---:|---|---:|
| 14,000,000 | `FUTURE_FATAL_1` | 12,547 MiB |
| 15,000,000 | `FUTURE_FATAL_1` | 13,369 MiB |
| 16,000,000 | `FUTURE_FATAL_1` | 14,199 MiB |

The 16M run reached depth 37 with local frontier counts around 8.46M before its
future arena overflowed. Its 14,199 MiB peak already leaves only about 1.13 GiB
of a 15,360 MiB T4 untouched, so increasing the DENSE capacity would violate
the required 1 GiB reserve before providing enough future space.

Therefore S11 is the largest completed symmetric group for the current
two-T4 DENSE layout. S12 is not an algorithmic or NCCL failure: it is the first
measured capacity boundary. Completing S12 requires a lower-memory profile or
representation; incomplete states were not published as a graph archive.
