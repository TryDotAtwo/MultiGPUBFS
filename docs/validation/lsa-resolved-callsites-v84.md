# Two-T4 LSA source-resolved callsites, Kaggle v84

Private `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` version 84 completed on
P2P-capable 2×T4. It built BFS source
`dd5e602df8d662698f1b28b8c27ece33ad9830e2` with release debug
information, captured both rank process maps, and ran `addr2line` against
that same build before deleting the large Nsight report. The notebook code
was committed as `a203e20`. Only one Kaggle notebook was active.

The single profiled S10 DENSE/CUCO_RANK/NCCL_LSA run completed 46 layers
and 3,628,800 unique states. Both committed rank archives verified. Search
completion was 0.476537 s, durable completion 7.451358 s, and the 50 ms
external sampler saw 597 MiB per rank. These values are diagnostic, not a
five-repeat unprofiled speed or exact-VRAM comparison.

The exported CUDA API trace contains 1,462 stream-synchronize calls and
1,740 synchronous memcpy calls. The unversioned synchronize callchains
were mapped to the two rank ELFs. Their source-resolved grouping is:

| First project frame | Calls, both ranks | Aggregate CUDA API duration |
|---|---:|---:|
| `DistributedNativeBfs::all_max_ring_or_host_fatal`, `distributed_native.rs:1499` | 180 (90/rank) | 358.94 ms (165.97/192.97 ms) |
| `DistributedNativeBfs::all_max`, `distributed_native.rs:1459` | 732 | 43.74 ms |
| `finalize_rank_owner` | 184 | 1.12 ms |
| `Buffer::put`, `distributed_native.rs:278` | 92 | 0.57 ms |
| Other/first frame unresolved | 274 | about 21.10 ms |

The aggregate duration sums concurrent rank API time; it is **not** a
critical-path breakdown or a predicted speedup. Versioned and unversioned
CUDA API names refer to the same calls and must not be added together.
The synchronous memcpy rows still have no callchains, so their 1,740
invocations cannot be assigned to source sites from this trace.

The dominant source site is the post-owner group fatal vote at
`distributed_native.rs:2703`, which calls
`all_max_ring_or_host_fatal()`. It currently protects both host/API error
propagation and reuse of the single LSA receive slot. Removing its host
wait alone is unsafe: exchange-stream writes on another rank could overwrite
remote payload still being read by this rank's owner stream. A connected
replacement needs a last-owner-consumer event/slot lease, same NCCL issue
order for empty epochs, and bounded group cancellation for asymmetric
host/CUDA/archive faults. The archive-error `all_max()` before each batch
is a separate CPU dependency and must be accounted for in that protocol.

Compact evidence: `test_results/kaggle_lsa_resolved_v84/lsa-bfs-gate/`,
especially `s10-nccl_lsa-r0/rank-addresses.json`, `rank-symbols.json`,
`sync-callsites.json`, `measure.json`, and the two verify logs. The
production owner→transport→retirement path is unchanged by this run.
