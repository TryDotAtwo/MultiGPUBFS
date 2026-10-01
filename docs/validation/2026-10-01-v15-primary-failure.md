# v15: full runtime evidence and primary failure preservation

Notebook: trydotatwo/mgbfs-native-rank-owner-t4 v15, COMPLETE.
Source: 2cdbcf62fb3f7ed466239307c45dae401b8f25ac. Actual two Tesla T4,
P2P allowed in both directions. This is not full sanitizer acceptance.

## Hardware evidence

Root summary: test_results/kaggle_native_rank_v15_summary/lsa-bfs-gate/summary.json.
Logs: test_results/kaggle_native_rank_v15_20261001/lsa-bfs-gate/.
Archives, rank results and eight SQLite traces:
test_results/kaggle_native_rank_v15_traces/lsa-bfs-gate/.

- Eight native leaf sanitizer checks pass.
- Sixteen single-rank owner capture/full-state cases pass.
- Sixteen two-process suites (faults and mapping/pre-dedup equivalence) return
  zero without timeout.
- Full racecheck and synccheck pass in all four owner/profile combinations.
- Full memcheck fails in all four combinations; no API error suppression.
- Full initcheck passes for CUB_SORT_MERGE DENSE/HASH_FIRST in this run;
  BMMA_BUCKET DENSE/HASH_FIRST fails before owner work, during NCCL activation.
  Earlier failed CUB runs remain evidence of instability, not retroactively waived.
- All four S8 timeline cases pass. Downloaded archive pairs were independently
  rechecked with verify_process_archives(n=8, world=2): 40,320 unique canonical
  states, 29 exact layers, valid checksums/config and matching per-rank counts.

Same-thread NVTX attribution of runtime APIs inside mgbfs.batch, excluding
nested mgbfs.archive_d2h, finds 178 batches/rank and zero Synchronize or
synchronous Memcpy calls in all eight traces. DENSE has no selected copies;
HASH_FIRST has 1,068 async copies/rank, 46,992 bytes, all D2D (copyKind=8,
verified against ENUM_CUDA_MEMCPY_OPER). This is not a claim of ideal overlap:
event-query credit waits and host scheduling remain. These small profiled S8
cases are not paired A/B performance measurements.

## Runtime defect and scoped correction

BMMA DENSE initcheck rank 1 reports native status 8 at LSA activation,
then status 2 in Buffer::put_u32 during setup_failure_vote. Its final JSON
incorrectly retains only CUDA_STATUS_2. HASH_FIRST similarly reports status 7
then status 2. The failing collective overwrites the primary local failure.
Zero sanitizer-reported memory errors does not imply successful execution.

vote_group_error now always attempts the same collective once, but preserves
an existing local error if that collective also fails. With no local error,
the collective failure is returned. LSA prepare/activate use this existing
helper. Constructor final agreement preserves the same precedence without
moving/dropping its successful runtime before publishing cancellation.
Prepare/activate failures also publish the startup sideband before later
setup buffers can be dropped. No collective was removed, payload padded,
buffer lease shortened, or healthy-batch readback added.

Regression failed before the fix (vote_cuda_failure instead of local_activation)
and passes after it; owner_pair has six passing tests. Complete runtime CPU
suite passed before the final sideband-before-cleanup addition; all five unit
tests and six owner_pair tests were rerun afterwards and pass. CUDA/library-owner
Rust typecheck passes, not a native link/hardware verification. This correction
does not fix NCCL's underlying initcheck/memcheck failures. Hardware verification
of the corrected source remains pending.
