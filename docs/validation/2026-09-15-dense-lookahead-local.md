# DENSE lookahead: local verification only

The reference runtime submits generation and hashing for the next bounded parent
batch after the current batch has finished packing. Current owner/NCCL work reads
packed states and sorted hashes; generation reuses children and child_hashes.
No second candidate arena is introduced. The next parent range remains live.
Generation events are sequence-guarded and retired only after packing completes.

This does not replace the admitted production dispatcher. Per-batch host waits
and the existing collective termination checks remain. HASH_FIRST does not use
this lookahead path. The submission counter is not a measurement of overlap.

Fresh checks on 2026-09-15:

- parent_batches, owner_pair and event_generation: 13 CPU tests passed.
- Linux CUDA cross-target cargo check passed for distributed_archive and
  distributed_bench, and for mgbfs-cli with its tests.
- git diff --check passed.

The new hardware fixture uses one-parent batches, checks archived full-state
layers against the oracle, and requires a nonzero lookahead submission count.
It covers compact/full state storage and CUB/BMMA owners with a reversed rank map.
It has NOT yet run on GPUs. The distributed archive suite now contains 13 tests;
the previously published 12-test gate is not evidence for this change.

Required next gate: real 2xT4 execution and all four Compute Sanitizer tools,
followed by source-pinned timing and memory comparisons. No performance or
hardware-correctness claim is made here.
