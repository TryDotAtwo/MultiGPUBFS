# LSA receive-slot owner-consumed event: two-T4 gates

Private Kaggle notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` was used
sequentially, one version at a time. Both versions ran on two physical
P2P-capable Tesla T4 GPUs. The Kaggle CLI downloaded the compact output files
but returned exit 1 at final console printing because of Windows `charmap`;
the reports and test logs below were checked independently.

## v85: profiled S10, source `fb5a335`

The DENSE/CUCO_RANK/NCCL_LSA path with pre-dedup ON, batch 32,768, four
shards/rank and 96 MiB cuCO pool/rank completed S10's 46 layers and 3,628,800
states. Both rank archives returned `VERIFIED`. In this **one profiled run**,
rank-maximum search completion was 0.468267 s and durable completion was
5.857567 s. The 50 ms external sampler recorded 597 MiB per rank; it is not
an exact VRAM peak. The comparable v84 diagnostic on different source was
0.476537 s search and 7.451358 s durable; those single samples do not
establish a speedup or archive improvement.

The source-resolved v85 Nsight capture still attributes 90
`cudaStreamSynchronize` calls per rank (180 total) to
`all_max_ring_or_host_fatal`. The event protects future reuse of the single
LSA receive slot, but **does not remove** the host fatal vote or make the
owner/transport/retirement pipeline asynchronous. The v85 event record was
after that vote, leaving its own CUDA API failure outside the group vote;
this was corrected in the next source commit.

Evidence: `test_results/kaggle_lsa_event_v85/lsa-bfs-gate/summary.json`,
`s10-nccl_lsa-r0/screen-summary.json`, both `verify-rank-*.log`,
`rank-addresses.json`, and `rank-symbols.json`.

## v86: error-vote ordering, source `81158d1`

The owner-consumed event is now recorded before the existing post-owner
group vote, and a record error is included in its local failure input. The
private v86 notebook completed these physical two-GPU Rust integration tests:

- LSA full-state layers and archives versus the small independent oracle;
- host-sized native and CUCO_RANK full-state/archives;
- one-rank owner capacity failure propagated to both ranks;
- one-rank injected host owner error propagated to both ranks.

All five test logs end with `1 passed; 0 failed`. These fixtures use two GPU
workers inside one test process; they do not prove independent-process
torchrun failure handling. They also do not inject a failure of
`cudaEventRecord` itself. v86 was not profiled or sanitizer-checked.

Evidence: `test_results/kaggle_lsa_event_vote_v86/lsa-bfs-gate/summary.json`
and the five named `*.log` test outputs.

## Remaining gate

The event is only the receive-slot lifetime prerequisite. Removing the 180
observed post-owner host waits still requires bounded cross-rank handling of
asymmetric host/CUDA/archive errors and same-order zero-payload collective
epochs. The event does not solve startup cancellation, HASH_FIRST, later
peer rounds, or the per-batch archive error vote. A new Nsight timeline and
full-state/fault/sanitizer gates are required after that connected change.
