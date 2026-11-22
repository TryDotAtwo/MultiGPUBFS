# Resident segmented hash: owner primitive checkpoint

This checkpoint is an internal acceptance primitive, **not** a resident hash
BFS implementation or a release. It does not add a public execution mode.
The existing host-driven segmented hash remains the production path.

The primitive uses the existing MgbfsSegment slot and SoA layout and the
same full-row comparison as the host-driven owner. Equal hash prefixes only
select possible matches. Each logical shard needs one exclusive owner; the
caller must prove that extent allocations and counts do not overlap, keep
them alive through retries and use a stable full-state hash and owner. An
unused shared-pool lease is returned after a hard-cap rejection. A failed
extent allocation is retryable until other owners have drained; treating it
immediately as a final global memory limit is not justified.

## Physical checks

Vast instance 61935705: V100-SXM2 32 GB, SM70, driver 580.178.04, CUDA
12.8.93, NCCL 2.30.7. The owner fixture ran separately on two physical GPUs;
it does **not** exercise inter-GPU hash transport or a resident scheduler.

- Independent CPU multisets of stored u8 and i64 states matched after two
  participating shard owners, exact full-hash collisions, six one-row
  segments, a rank cap, duplicate-at-cap and replay after raising the cap.
- Distinct states with one complete hash in the same segment, a full bank,
  a full index, existing library rehash and a tombstone in a collision chain
  were exercised. A failed insert did not consume row or rank capacity.
- A VMM boundary fixture wrote row 2^32-1, then checked the duplicate and
  full result. Only a 2 MiB page was physically committed; this is an address
  test, not a 2^32-row allocation or a measured capacity result.
- Compute Sanitizer memcheck reported zero errors on this focused fixture.
- Since the full-row comparator is shared with the existing owner, the
  existing host-driven segmented24 BFS was run on two physical GPUs via
  persistent workers. All exact per-layer state lists and duplicate checks
  matched a CPU oracle for S4, directed C6 and a repeated-symbol permutation,
  for atomic and warp: 42 completed layer comparisons in total.
- Focused CTest checks for the primitive and existing segmented history passed.

The tested executable SHA-256 was
425829af00d60a817ed8f318e8e359d725584146bc453939234d1b7dbccd0f18;
the tested CUDA shared library SHA-256 was
4803558852a4f3189fa920f68e17f2971f279e8ff7f6b74aea1979cea36db0d0.

Still required: connect this primitive to a GPU-resident scheduler and public
API, prove layer closure and bank retirement, run per-layer multi-GPU oracles
through that route, handle directed history, and compare performance and VRAM.
The existing bitmap engine and this primitive do not constitute that result.
