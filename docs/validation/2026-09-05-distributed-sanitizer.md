# Two-device archived runtime sanitizer gate

Kaggle `trydotatwo/mgbfs-distributed-sanitizer` v1: COMPLETE.
Source `420b8a7eea491f7cee4bab491e82f015323d16de`, two distinct physical T4s.

Fixture: `parent_batch_archive_preserves_full_layers_and_hashes_on_two_devices`.
Unitriangular(3,3), 27 states, generation variant 1, reverse owner map,
batch 7, archive rows 3, ring wrap, NCCL exchange and full-state/hash archive
comparison against CPU oracle. Both ranks run as device-owning threads.

| Tool | Result | Fixture seconds |
|---|---|---:|
| Plain | PASS | not a performance measurement |
| memcheck | 0 errors | 13.36 |
| racecheck | 0 errors, 0 warnings, 0 hazards | 12.89 |
| initcheck | 0 errors | 9.87 |
| synccheck | 0 errors | 5.92 |

Each instrumented run reports 1 passed, 0 failed. These timings include
instrumentation and must not enter performance comparisons.

This validates the tested distributed matrix runtime fixture, not compact
generation variant 5 used by S13, all profiles, arbitrary capacities, or
absence of all inter-stream/global-memory races. Compact runtime coverage
remains a separate required test.

Evidence: `test_results/distributed-sanitizer-v1/distributed-sanitizer/`.

## Compact path gate v2

Source `8f04c4642470c8c48954ac705dda941e06b08625`, Kaggle v2 COMPLETE.
Adds S4 compact generation variant 5, width 4, capacity 24, full archived
state sets and hash comparison to independently generated matrix CPU layers.
Both original and compact fixtures pass in each invocation (2 passed, 0 failed).

| Tool | Result | Both fixtures seconds |
|---|---|---:|
| memcheck | 0 errors | 18.56 |
| racecheck | 0 errors, warnings, hazards | 20.44 |
| initcheck | 0 errors | 13.15 |
| synccheck | 0 errors | 6.52 |

Evidence: `test_results/distributed-sanitizer-v2/distributed-sanitizer/`.
This covers the compact runtime path on S4, not an instrumented S13 run or
all frontier sizes, profiles, rank maps and failure schedules.
