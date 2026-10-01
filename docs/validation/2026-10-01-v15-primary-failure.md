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

## Replay admission

Correction source d018a6d7d2dc65747cfc72b4b9a7adfc2e70c86c was pushed to
origin/codex/library-first-bfs. The private wrapper pins this immutable source.
Its Python AST and metadata validation pass. Kaggle refused its push with
"Maximum weekly GPU quota of 30.00 hours reached." This is not a running v16
or a hardware result. A subsequent status query still reports COMPLETE for
the existing notebook. Do not repeatedly push or launch a second notebook.

Existing production NCCL wrapper test double was compiled with g++ in the
reused isolated Linux image and exited zero (nonblocking progress, grouped
failure cleanup, single abort and terminal-handle guards). This validates
wrapper behavior against stubs, not NCCL itself or actual inter-GPU transport.

## Independent local checks while Kaggle is unavailable

Two existing CUDA fixtures were freshly compiled from d018a6d for sm_86:
tests/macro_settle.cu + cuda/macro_settle.cu, and
tests/macro_future_checked.cu + cuda/future_merge.cu + cuda/macro_settle.cu.
Both passed all four unfiltered tools (memcheck, racecheck, initcheck,
synccheck), with error-exitcode 97 enabled. All process exits were zero;
racecheck reported zero errors/warnings and the other tools zero errors.
This checks bounded settle/future kernels, not distributed macro BFS.

Hardware: RTX 3070 Laptop GPU, 8,192 MiB, driver 572.70. nvcc 12.8.93;
Compute Sanitizer 2025.1.0.0 build 35583870. Reused image
multigpubfs-gpu:dev, SHA256 55f9efc3c2d82a3110e23f9fdc194026d6f55197105d10dfd6f48a4d0240bf0f.
Read-only workspace mount, no network, temporary binaries inside disposable
container, no large local dataset. This does not replace any 2xT4 gate.

Python: all 31 distributed metric tests and 22 promotion/reconciliation tests
pass. The first attempted HF test filename pattern matched zero tests and
was not treated as verification; the correct test_promo*.py discovery was
run afterwards. No old remote publication was overwritten.
