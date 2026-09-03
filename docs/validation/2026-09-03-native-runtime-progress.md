# Assembled DENSE executor: current evidence and limits

The new `mgbfs-runtime::native::NativeBfs` assembles GPU generation, GEMM hash,
CUB routing, GPU bucket directories, bounded owner jobs, irreversible commit,
and StateRing feedback. The CPU dispatch sees compact directory/commit metadata,
not candidate state/hash payloads. It does not call `DenseDeviceStepper`.

`native_bench` adds mandatory preallocated pinned slots and a dedicated disk
worker. Exhaustion is fatal. Search completion and durable RunCommit have separate
timers. The archive uses the existing checksummed **MGBFSAR1** codec, not the
planned schema2 wire protocol. Canonical states and associated hashes are checked
against the CPU oracle in the archive integration test. Archive hashes are
recomputed on GPU in state order; this cost is included in measured search time.

## Local checks

On the local RTX 3070 Laptop GPU, CUDA 12.5, sm86:

- Complete full-state layer sets U4(2..8), pre-dedup OFF/ON, match the CPU oracle.
- Padded n=2/n=3 states, nonidentity start, batches 1/2/7 and seeds 0/1/20260828
  exercise ping-pong reuse and state-stride handling.
- Layer capacity fails before owner hash mutation; the executor stays failed.
- Pinned archive roundtrip validates framing/checksums, every state/hash pair,
  and complete exact layer sets.
- The standalone bounded-owner fixture passes with parallel merge-path tiles.

A pageable H2D control upload originally used the default stream while consumers
used a nonblocking stream. Uploads now use the consumer stream and preserve the
host source lifetime through DMA completion. Twenty consecutive small full-layer
runs passed after this change, before the later pipeline optimization.

## Local optimization probe, not target-hardware A/B

U4(16), batch 65536, 256 buckets, 16 shards, J=16, K=131328, pre-dedup ON;
unarchived local development probe, two runs per variant, no controlled clocks:

| Variant | Search seconds | Requested device bytes |
|---|---|---:|
| One CTA per bucket, serial producer/owner | 11.934, 11.036 | 1757466523 |
| Parallel disjoint merge-path tiles | 3.852, 3.831 | 1757466523 |
| Two producer banks, producer/owner streams | 2.827, 2.710 | 1779490723 |

These are diagnostic samples, not five-run medians or a comparison with CayleyPy.
The two banks add 22024200 bytes; owner scratch remains I/J/K-bounded. A previous
pipeline probe omitted the alternate bank from its printed accounting; the table
uses the corrected counter and a fresh rerun.

## Hardware run boundaries

Kaggle `trydotatwo/mgbfs-native-runtime-t4` v1 compiled source
`a37050c4565de200697654a478d4e4ca6b63ec9b`, then failed in notebook orchestration:
the pinned bootstrap helper did not expose `run_gpu_suites`. No GPU test or A/B
result was produced. v2 fixes only that notebook error and tests the same source.
It does **not** include the later parallel-tile or ping-pong changes.

## Not complete

This is a single-rank DENSE reference, not the whole requested runtime. Native
NCCL multi-rank execution, HASH_FIRST, BMMA owner, full production preflight and
schema2 manifests/archives are not implemented here. StateRing/slot allocations
are fixed, but host directory dispatch still synchronizes each owner job and
archive D2H currently runs at the depth boundary. No complete architecture or
performance/Pareto claim follows from these checks.
