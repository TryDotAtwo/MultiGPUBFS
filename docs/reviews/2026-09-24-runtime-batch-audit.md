# Runtime batch audit (2026-09-24)

Scope: static inspection of `distributed_native.rs`, `reference_bench.rs`,
`bootstrap.rs`, `control_connection.rs`, `group_commit.rs`, and the archive
path at `1035c38` plus the pending boundary-connection fix. This is not a GPU
profile, a correctness proof, or a claim that every defect has been found.

## Findings, ordered by implementation dependency

1. **The new run-boundary gate did not build on Linux.** Kaggle two-T4 v62
   stopped at `reference_bench.rs:254` (`let extent = ... .map(Some)` without a
   returned expression), before any NCCL/BFS check. Correct that compile error
   and rerun the same pinned-source gate; no v62 runtime result exists.
2. **Boundary traffic could consume the control connection's one-frame initial
   outbox before dispatcher admission.** `BootstrapGroup::agree_boundary()`
   called ordinary `ControlConnection::enqueue/poll_*`; those mark the connection
   started, while `ControlPump::new` needs to resize its outbox first. A CPU
   regression test now exercises admission after ArchiveAdmission. The pending
   dedicated boundary methods preserve pre-dispatcher ownership, but a later
   dispatcher handoff of all three phases is not implemented.
3. **The current LSA + CUCO_RANK DENSE path is not a fully device-driven BFS.**
   Device counts avoid the route-size D2H in the LSA exchange and the rank-owner
   transaction avoids per-shard host loops. However `advance` still calls
   `all_max` around archive work and owner/fatal handling; it reads final extent
   descriptors and layer counts on CPU. The legacy host-sized transport still
   synchronizes for per-peer counts, and indexed CUCO/native/HASH_FIRST owners
   retain per-batch or per-shard host reads. Claim overlap only for measured
   segments; do not infer end-to-end GPU saturation from the owner microbench.
4. **Cross-rank fail-fast remains a protocol problem, not one removable wait.**
   The per-round failure votes preserve NCCL order after asymmetric errors.
   Removing them must provide bounded cancellation without stranding a peer in
   a different collective. The previous GPU-only vote attempt and its timeout
   are documented in `docs/plans/device-driven-library-owner.md`; keep the
   current safe votes until a tested replacement exists.
5. **Completion evidence is narrower than run durability.** File archive
   `sync_all`, FIFO `flush`, per-rank JSON `sync_all`, and the group marker are
   different boundaries. FIFO handoff does not prove remote HF persistence.
   Benchmark aggregation must reject missing/tampered group markers for new
   rank records. The group marker currently hashes rank summaries, not every
   archive byte; archive integrity must be checked independently.
6. **Setup errors are only partly coordinated.** Archive admission is agreed
   before `DistributedNativeBfs::new_*`, but constructor allocation and NCCL
   setup can still fail asymmetrically. `env_u32` also panics on malformed
   values before rendezvous. A peer can then spend its bootstrap/NCCL timeout
   waiting rather than receive a precise group failure. Separate pre-NCCL
   validation/allocation from collective communicator setup, agree on the
   former, and add a two-rank asymmetric constructor-failure fixture.
7. **Archive reservation is not derived from the rank capacity.**
   `reference_bench.rs` reserves `expected_states * (archive_width + 16)
   + 64 MiB` independently on every rank. That can over-reserve disk by up to
   the world size under equal partitioning, yet can under-reserve when an
   explicit rank capacity exceeds the graph-order estimate. It also omits
   `Archive::frame()` overhead: 112 bytes per record batch, per layer, and
   final commit, so a tiny `archive_rows` can exceed the fixed 64 MiB slack.
   Derive a checked per-rank bound from accepted-state capacity plus an
   explicit record-frame/layer-frame budget (or a conservative worst case),
   with separate file and stream preflight tests. Exact format accounting is
   `48 + N*(width+16) + 112*(record_frames+layer_frames+1)` bytes; `N` alone
   does not determine the frame counts. Expose those budgets in config rather
   than silently relying on the current 64 MiB constant.
8. **FIFO admission can block before the bounded agreement.**
   `create_archive_extent()` uses a blocking write-only `OpenOptions::open` on
   a FIFO. If its consumer never opens the read end, the rank never reaches
   `ArchiveAdmission`, so the peer's 600-second timeout cannot make the stuck
   rank exit. Use a deadline-bounded nonblocking FIFO-open protocol and test a
   missing consumer. This is distinct from later FIFO write/flush backpressure.
9. **The new boundary notebook's stated oracle gate is narrower than its
   assertions.** In `boundary_gate` it checks total S4 rows (`24`), archive
   structural verification, and marker hashes, but not per-depth state-set
   equality. The separate `library_multi_gpu` ignored oracle test exercises
   that stronger property; run it on the same pinned source before calling this
   notebook a full-state correctness gate, or narrow the notebook scope label.

## Batch implementation plan

- **Gate A: launch/commit correctness.** Fix the Linux compile error; retain
  boundary socket usability; run CPU protocol tests, two-T4 HostSized and LSA
  BFS with layer/archive comparison, asymmetric archive-admission fault, and
  marker validation. Run the independent full-state oracle on the same pinned
  source. Bound missing-consumer FIFO open and add asymmetric
  constructor failure after the archive gate
  before treating launch as fully fail-fast. Publish raw failure and success
  logs. Do not expand the
  hot-path refactor until this gate is green.
- **Gate B: one coherent hot-path transaction.** Replace host-derived owner
  decisions, route sizing, extent publication, and retirement with bounded
  device-resident descriptors and explicit stream-event ownership. Keep the
  same fixed NCCL issue order and a bounded group-fatal protocol. Treat DENSE,
  HASH_FIRST, legacy indexed CUCO, and host-sized transport as separate paths;
  no backend name alone proves that a path satisfies this gate.
- **Gate C: target validation and tuning.** Full layer/state and archive
  parity across 1/2 ranks, empty/asymmetric/capacity faults, four Compute
  Sanitizer modes, then Nsight Systems timeline and repeated time/VRAM panels.
  Sweep shards and batch sizes only after correctness and memory ownership
  pass. Report search completion separately from archive/group commit.
- **Capacity accounting before a larger archive run.** Correct the rank-local
  archive bound and test both over-reservation and explicit-capacity overflow;
  do not infer stream/HF persistence from local FIFO flush.

The batch boundary is deliberate: individual wait removal is not a completion
milestone. Gate B is complete only when the owner-to-transport-to-retirement
DAG and its failure path work together on target hardware.

## Performance hypotheses to measure, not assumed fixes

- The LSA peer path launches `lsa_publish_count` and `lsa_copy_exact` per XOR
  round, with a fixed 16 copy CTAs (`cuda/nccl_transport.cpp`). At larger
  payloads or 8 ranks this may underfill the device or expose collective
  launch latency. Sweep CTA count and payload shape only with a stage-resolved
  timeline and unchanged correctness/fatal gates.
- Even one rank still passes through route radix sort and pack. A specialized
  one-rank path could avoid work, but must preserve the exact pre-dedup and
  owner order. Compare an ablation before adding a second implementation.
- Archive hash computation and D2H are asynchronous, but parent retirement
  waits on archive completion. A slow consumer can exhaust pinned slots and
  fail the run by contract. Measure archive slots, D2H throughput, and disk
  worker occupancy alongside search time; do not call this hidden overlap.
