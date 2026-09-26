# Connected runtime audit (2026-09-26)

Source: originally `d3acc1f`; rechecked against `feea0d9` plus the scoped
`reference_launch` working change. Three unrelated dirty paths were excluded.
This is a static audit, not a new GPU test or performance measurement. It
updates the older `2026-09-24-runtime-batch-audit.md` and
`owner-transport-retirement-batch-audit.md`; those documents retain the raw
T4 evidence. The v77 two-T4 gate covers post-NCCL rank-local constructor
failure, not all startup failures.

## Connected findings

1. **Startup admission is partial, not end-to-end.**
   The depth-one reference path now bootstraps by launch identity before
   parsing the graph/config and votes on the config digest and
   `cudaSetDevice` outcome. The scoped `reference_launch` change routes
   unsupported multi-rank `MGBFS_MACRO_DEPTH` through this vote too.
   `DistributedNativeBfs::new_profile`
   validates config, sets the CUDA device, queries memory and creates streams
   before `mgbfs_nccl_create`. A rank that returns in any of these stages can
   leave a peer waiting in NCCL. The v77 group vote starts only
   after communicator creation and LSA setup. The post-NCCL result must not
   be described as a pre-NCCL resource-admission guarantee. The remaining
   constructor boundary needs a sideband/abort contract, not another
   config-only digest.

2. **The post-NCCL admission controls can themselves fail asymmetrically.**
   `admit_device_group` allocates its two control buffers before its first
   vote; `new_profile` allocates `setup_send/setup_recv` before the LSA vote.
   An OOM or stream error here can return on one rank while its peer is in a
   collective. Provisional communicator abort alone did not suffice in the
   earlier v75 fault experiment. `mgbfs_nccl_lsa_prepare` performs local
   capability query, `ncclMemAlloc` and memset; its returned error can be
   voted if control buffers survived. `mgbfs_nccl_lsa_activate` contains
   `ncclCommWindowRegister` followed by `ncclDevCommCreate`; a one-rank
   failure between these collective steps can strand its peer before the
   post-return vote. Each point needs a bounded sideband/abort contract and
   asymmetric target-hardware fixture; a GPU vote cannot be the sole failure
   channel when CUDA allocation itself has failed.

3. **The healthy DENSE+LSA+CUCO_RANK batch remains host-coupled.**
   `advance_inner` still calls `all_max` for archive error per scheduled
   parent batch, ring/transport fatal on later peer rounds, and post-owner
   host/API fatal. `all_max` synchronizes and reads a GPU word. The rank-owner
   device transaction and LSA device counts do not remove these enclosing
   dependencies. Indexed CUCO/native owners, HASH_FIRST, and HostSizedNccl
   have additional host count/directory/readback chains. At `FinalizeDepth`
   the extent readback is semantically distinct and can remain on CPU.

4. **Removing the post-owner wait would reopen receive-slot reuse.**
   `exchange_done` publishes the remote payload to the owner. It does not
   prove the owner finished reading the single LSA receive slot before the
   next exchange writes it. The current host vote/drain can incidentally
   cover this lifetime. A K-slot protocol needs an `owner_consumed` event or
   equivalent lease for each slot, and a bounded fatal tail; merely deleting
   the vote is incorrect.

5. **The timer named `durable_run_commit_seconds` stops too early.**
   `run_pass` samples it after `PinnedArchive::finish` and the
   `ArchiveCommitted` group vote, before rank JSON `sync_all` and rank-0
   `group-complete.json` publication. It is a local archive-commit timer,
   not a group-run durability timer. Existing consumers require preserving
   its schema with explicit scope; add a separate group-publication timer
   where the rank-0 marker is actually durable. A FIFO flush remains only a
   handoff, not remote HF durability.

6. **Group completion previously was not rank-symmetric.**
   `run_pass` voted after every rank-result fsync, then rank zero alone wrote
   `group-complete.json`. The scoped follow-up adds a `GroupPublished` boundary
   after that write, so a rank-zero publication error reaches peers. A CPU
   asymmetric-failure test and Linux/CUDA Rust typecheck pass; actual two-rank
   process and late filesystem-failure gates are still pending. This was a
   reporting/termination contract gap, not evidence of incorrect BFS states.

7. **The actual CLI can still bypass configuration admission.**
   `mgbfs-cli/src/main.rs` checks `MGBFS_MACRO_DEPTH` and the archive contract
   before calling `reference_bench::run`; an asymmetric error can therefore
   exit one process while its peer enters bootstrap. `reference_bench::run`
   also reads `MGBFS_BENCH_WARMUP` before rendezvous. Different warmup values
   select different bootstrap paths. The `reference_launch` fix covers only
   the inner `run_pass` path. Move all rank-varying launch validation into a
   common pre-NCCL admission, then test the real CLI with two processes.

8. **A connected FIFO reader can stall the archive indefinitely.**
   `archive.rs` removes `O_NONBLOCK` after opening the FIFO and `StreamExtent`
   writes with blocking `write_all`. The open deadline does not bound writes
   to a reader that connects but stops draining. `PinnedArchive::finish` joins
   the worker, so the failure path needs a cancellable, bounded write/shutdown
   contract. This is source evidence of a possible stall, not a measured one.

9. **The HF promotion validator previously accepted incomplete inventories.**
   `promote_hf_stream.py::combine_rank_commits` checked layer totals but
   accepted missing or empty `files` even with positive state count. The
   scoped follow-up writes per-file row counts, requires their sum to equal
   each rank's state total, resolves staging branches to immutable Git SHA,
   checks staged object size/LFS SHA, and reconciles the destination at the
   returned commit OID. All local Python tests pass. This is not a live Hub
   publication gate, and the remote Parquet footer row counts are not yet
   independently read. Existing staged V1 manifests without `rows` require
   regeneration or an explicitly verified legacy import; published datasets
   are unchanged. The old gap alone did not prove corruption of any dataset.

## One implementation batch, in dependency order

1. Freeze a versioned launch/epoch protocol: common rank identity before
   parsing, per-rank validation result and config digest agreement, local
   resource admission, communicator creation, and post-NCCL admission.
   Define who times out/aborts at each boundary; keep the working synchronous
   backend intact.
2. Specify one preallocated device epoch record with counts, ranges, sticky
   fatal, owner results, extent descriptors and generation. For every slot,
   specify producer event, consumer event, publication and reuse. Budget
   `K * (receive + metadata + scratch)` plus archive leases before coding.
3. Implement transport, owner commit, failure propagation and parent
   retirement as one bounded epoch state machine. Preserve identical NCCL
   issue order and zero-payload participation. Keep a finite outstanding
   epoch count and abort on capacity/fatal; never submit an unbounded depth
   to hide host waits. Apply separately to DENSE+LSA+CUCO_RANK first, then
   HASH_FIRST, indexed owner and HostSized if they remain supported.
4. Correct metric names/scopes in the same change set that revises group
   completion. Validate marker absence/tampering and an injected late write
   failure. Do not mix search time, archive file fsync, FIFO handoff and
   group marker publication in one number.
5. Gate the batch with CPU protocol/model tests; 2×T4 asymmetric config,
   pre-NCCL CUDA, LSA-setup, owner, archive and late-output failures;
   full-state/archives with empty and asymmetric traffic; four sanitizers;
   then a real BFS Nsight Systems timeline and repeated time/VRAM comparison.
6. Gate the actual CLI before launch, and fix FIFO liveness and HF inventory
   verification as end-to-end output contracts; isolated inner-runtime tests
   are insufficient for these boundaries.

## Separate performance hypotheses after the protocol gate

- One-rank route/pack may be redundant, but preserve pre-dedup and owner
  order; compare a correct ablation.
- CUCO_RANK selection over fixed capacity may waste work on sparse tail
  batches; measure valid/capacity ratio and full-run effect.
- LSA's fixed copy CTA count, rank shard count, and archive drain may limit
  throughput; profile on the target workload before changing them.

No new correctness or speed claim follows from this audit.

## Implementation progress after the static audit

The depth-one reference launch now rendezvous by launch identity, performs
rank-local config parsing and `cudaSetDevice`, and exchanges all 256 config
digest bits plus a local-failure flag before archive admission or NCCL
communicator creation. The two-rank CPU control test covers one-rank parse
failure on either rank and unequal digests; the available runtime CPU suite passes. A
Linux/CUDA Rust `cargo check` passes without linking a CUDA library. This is
not yet a physical GPU gate. The constructor's other pre-NCCL operations
and LSA internal setup still need asymmetric fault coverage. The scoped
`reference_launch` change rejects multi-rank macro requests during config
agreement and preserves the supported single-rank macro path; it has only a
CPU control test and Linux/CUDA typecheck, not a two-T4 run. The current
post-NCCL constructor fixture uses two threads in one process and now asserts
the exact remote-fatal code; a separate two-process fault gate is still needed
before claiming process-level termination.
