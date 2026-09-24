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

## Batch implementation plan

- **Gate A: launch/commit correctness.** Fix the Linux compile error; retain
  boundary socket usability; run CPU protocol tests, two-T4 HostSized and LSA
  BFS with layer/archive comparison, asymmetric archive-admission fault, and
  marker validation. Publish raw failure and success logs. Do not expand the
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

The batch boundary is deliberate: individual wait removal is not a completion
milestone. Gate B is complete only when the owner-to-transport-to-retirement
DAG and its failure path work together on target hardware.
