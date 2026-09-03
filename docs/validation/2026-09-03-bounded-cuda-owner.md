# Bounded CUDA owner: verified sorted-input leaf

Source: `0c27d0a491cc733bc6d87d688de8adc396db880c`.
Private Kaggle kernel: `trydotatwo/mgbfs-bounded-owner-t4`, version 1.
Result: **VERIFIED_BOUNDED_OWNER_GATE 10/10**.

## Implemented

`cuda/bounded_owner.cu` implements separate asynchronous Compare and Commit
entry points behind `cuda/bounded_owner.h`:

1. Validate device job descriptors, bucket counts, generation/lane/shard and
   arena bounds before reading candidate or persistent hash data.
2. Mark repeated incoming hashes (first incoming row is the representative).
3. Compare against prev, curr and already accepted hashes, in that priority.
   Merge-path tiles cooperatively load consecutive ranges into shared memory;
   lane-local searches use shared tiles rather than binary-searching whole
   resident layers for every candidate. Only tile boundaries search globally.
4. Block scans compact stable source indices. Per-bucket category counters and
   offsets describe survivors; capacity is checked before persistent writes.
5. Commit verifies device reservation credits, merges accepted + survivors
   into separate bounded output, copies back and publishes counts. A subsequent
   kernel publishes stage 2; consumers must wait for that completion event.

Create allocates `I` flag bytes, `4*I` index bytes and `16*J*K` merge bytes.
These are requested payload bytes, not measured total VRAM or an allocator
overhead estimate. Descriptors, input/old/accepted hashes, counts, control,
reservation result and final source-index output are caller-owned allocations.
There is no device allocation or host synchronization in Compare/Commit.
The old global-owner prototype is unchanged.

## Verification

- Initial GPU test failed against the stub (`FAIL: create`).
- Local RTX 3070 Laptop: literal category fixture, repeated-job rejection,
  12 seeded sweeps including 1-row and 257-identical-key bucket tails and
  up to 8000 input rows across four buckets; all passed.
- Hash fixtures exercise all four 32-bit words and compare with independent
  host sets and stable input indices.
- Four failure fixtures check insufficient commit grant, invalid old range,
  invalid accepted count, and actual post-dedup bucket overflow. Failed jobs
  leave both persistent hashes and published counts unchanged.
- Sweep compare and commit are enqueued without an intervening host wait or
  readback (the pre-provided test grant is a fixture, not a StateRing allocator).
- Local Compute Sanitizer failed to launch the Windows target. No local
  sanitizer success is claimed.
- Kaggle CUDA 12.8.93, compiled for sm75, actual Tesla T4 UUIDs:
  `GPU-23b33a4b-48c9-87c4-6b44-a959d1da44b2` and
  `GPU-d2a94c02-4b99-71ab-432f-6f9e8ed5396f`.
- Both cards independently ran the complete test executable in plain,
  memcheck, racecheck, initcheck and synccheck modes. All eight sanitizer
  executions reported zero errors; both racechecks also reported zero warnings.
- Downloaded logs independently checked with:

  `python scripts/verify_bounded_owner_gate.py test_results/bounded-owner-v1/bounded-owner-gate --source 0c27d0a491cc733bc6d87d688de8adc396db880c`

Local raw logs: `test_results/bounded-owner-v1/` (ignored, not public).
CPU regression: 62 tests passed on Windows; evidence-verifier/launcher suite:
13 tests passed. Source GitHub CPU CI run `33756659249` succeeded.

## Explicit remaining boundary

This is a **sorted-input owner leaf**, not an assembled production BFS and
not a multi-rank NCCL gate. Caller must supply sorted per-bucket incoming rows
with deterministic provenance order and immutable validated old directories.
The bounded multi-source merge and job splitter are still needed upstream.
The same jobs, input, persistent views and control must remain unchanged
between Compare and Commit, under an exclusive shard lease. Caller owns the
fatal rank-group policy and StateRing/materialization/archive reservations;
the supplied grant is not a substitute for implementing those reservations.
Rust runtime integration, full-state materialization, complete allocation
query provider, multi-rank scheduler, BMMA backend and end-to-end A/B remain
pending. No throughput or end-to-end memory win is claimed here.
