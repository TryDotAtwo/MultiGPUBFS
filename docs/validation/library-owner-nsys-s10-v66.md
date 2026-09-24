# One-T4 CUCO_RANK S10 diagnostic profile v66

Private Kaggle `trydotatwo/mgbfs-library-owner-t4`, version 66, source
`ae3dce9433508faae6730132b43b072eea84dd78`. The notebook completed
with status `PASS`. This is one diagnostic Nsight Systems run of S10 DENSE,
CUCO_RANK, pre-dedup ON, one physical T4, batch 32768, 1,000,000-record
state/ring capacities and a 96 MiB fixed cuCO pool. The archive checksums
and counts were verified. It is **not** a repeated speed or peak-VRAM
comparison, nor a two-GPU timeline. The profiler range covers the timed BFS
and archive submissions, not the later durable archive drain.

The measured run reported `search_complete_seconds=0.411638465` and
`durable_run_commit_seconds=11.339358252`; its preallocated pinned archive
ring was 973,078,528 bytes for 256 slots. The explicit runtime allocation
scope and sampled device consumption are distinct; do not substitute one
for the other. Raw report, SQLite export, Nsight statistics, manifest and
rank JSON are under
`test_results/kaggle_library_profile_v66/library-owner/screen-s10-dense-cuco_rank-w1-r0/`.

The Nsight summary gives these **aggregate durations**, not additive
end-to-end stage times (GPU operations can overlap):

| Scope | Aggregate | Calls / instances | Interpretation |
|---|---:|---:|---|
| CUDA API `cudaStreamSynchronize` | 241.3 ms | 1,135 | Host waiting is substantial in the captured range. |
| CUDA API `cudaMemcpy` | 51.1 ms | 1,420 | Host-visible transfers remain in this one-rank path. |
| GPU CUB radix onesweep | 76.3 ms | 2,096 | Largest listed kernel family; route sorting is a serious optimization candidate. |
| GPU modular materialize | 66.1 ms | 146 | Epilogue/materialization also matters. |
| GPU generation CUTLASS GEMM | 25.1 ms | 146 | Tensor Core work is not the only or largest expense. |
| GPU pack parents | 22.4 ms | 146 | Candidate for layout/fusion analysis. |
| GPU device-to-host copies | 47.7 ms | 1,711 | Includes control/archive traffic; attribution needs timeline ranges. |

These numbers support a connected audit of host control, route and
materialization. They do **not** prove that deleting a wait, bypassing sort,
or fusing GEMMs improves total BFS time. The report's `osrt_sum` also
includes archive-worker `poll` and `clock_nanosleep`; these are not GPU
stall durations and must not be added to the search critical path.

Next discriminators: export per-NVTX-range statistics from this SQLite
timeline, then profile a two-T4 full BFS with the same labels; ablate
one-rank route/pack only after preserving owner order and pre-dedup semantics.
