# Resident hash: experimental independent shard owners

The device-resident hash remains opt-in and is not performance-ready. This
checkpoint adds an optional `MGBFS_RESIDENT_HASH_SHARD_OWNERS=1` mode in which
separate shards have separate nonblocking owner leases. Unset or `0` retains
the previous serial owner. When an allocation reports no free extent, the
receiver releases its own lease, nonblockingly obtains all shard leases and
rechecks; a conflicting lease preserves the live packet and retries. The
aggregate extent, row and segment limits remain 64-bit as before. Optional
`MGBFS_RESIDENT_HASH_OWNER_STATS=1` records concurrent owners and the
full-pool recheck path; it is disabled in the timed comparisons.

The mode and optional instrumentation are agreed across ranks before the GPU
run. Old and new incompatible hash-config ABIs fail explicitly. This is not an
automatic replacement for the existing host implementation or the default
resident owner.

## Coverage

Machine: up to four V100-SXM2-32GB, SM70, CUDA 12.8.93, NCCL 2.30.7. The
eight-V100 host does not provide the required all-pairs peer atomics. The tested
executable SHA-256 was
`836eb2592b5c3fed959e4b6fc41fbaf3574a282b3d5f41459eb7adca79fceef1` and the
loaded CUDA library SHA-256 was
`9b1053c7fe9e27756a0175f9c9757a9f4e0ebd353fed40b3888eeba2913f1c32`.
Source parent: `bf9892b28e7d70a84bd15d87aec07d62bd0dc0bc`.

- On 2 and 4 GPUs, the exact layer-multiset regression passed on S4,
  repeated-symbol and signed-wide-symbol graphs, including tiny segments,
  sufficient and insufficient caps, duplicate-at-full and repeat jobs in the
  same persistent worker processes. At least two shard owners were observed
  active at once. The 2-GPU all-hashes-zero run also matched the independent
  oracle on every layer; it concentrates keys in one owner/shard and therefore
  does not combine that collision with independent owners.
- Actual CayleyPy persistent `graph.bfs()` passed on four GPUs for S4,
  repeated-symbol, duplicate-at-full, a resource-stopped prefix and a healthy
  next job. Killing one rank in a 2-GPU job invalidated that generation and an
  explicit new group completed a healthy job.
- With the pool limited to one segment, 2- and 4-GPU runs confirmed a
  full-pool recheck, exact last completed states and a subsequent healthy job.
  A bounded four-GPU contention search on S6 observed peak two simultaneous
  shard owners, five initial full-pool observations, four lock-contention
  retries, one recheck and one confirmed full pool. The last completed layer
  matched the CPU multiset; every completed size matched the oracle. Whether
  this scheduling occurs on another run is not guaranteed; the test reports
  lack of an observed retry instead of calling it a correctness failure.
- Two-GPU Compute Sanitizer memcheck with CUDA API-error reporting disabled
  reported zero memory errors for the pressure/healthy-next fixture. The
  initial run with default API-error reporting emitted 46 API errors during
  library initialization, including NCCL `cudaFuncGetAttributes`, `cuMemCreate`
  and already-enabled peer access. The fixture still completed; the suppressed
  API errors are not counted as a clean default sanitizer run. Logs are saved.

Reproduce with the matching resident-capable build:
`RESIDENT_TEST_WORLD=4 MGBFS_RESIDENT_HASH_SHARD_OWNERS=1 MGBFS_RESIDENT_HASH_OWNER_STATS=1 MGBFS_SEGMENT_TEST_BATCH=1 MGBFS_RESIDENT_LANES=8 MGBFS_RESIDENT_PACKET_RECORDS=8 python tests/device_resident_hash_layers_gpu.py`.
The pool and bounded observational stress fixtures are
`tests/device_resident_hash_pressure_gpu.py` and
`tests/device_resident_hash_contention_gpu.py`; the former additionally expects
`MGBFS_SEGMENT_TEST_LIMIT=1`.

## Comparison, BFS core

Same executable/library and four GPUs; adjacent-transposition graph, two
shards, requested and actual 8192 rows/segment, initial batch 128, eight device
lanes and eight records/packet. S8 used six extents and 5,505,168 allocated
segment-storage bytes per rank in each device run. Those bytes are not
device-wide peak VRAM. Ordering was serial, parallel, parallel, serial.
All 29 Mahonian layer counts and the complete final state matched.

| S8 resident owner | BFS seconds |
|---|---:|
| serial | 2.948887 |
| parallel | 1.593999 |
| parallel | 1.591431 |
| serial | 2.943028 |

The per-shard lease decreased this small-graph time by approximately 46% in
these two pairs. This is a comparison of the two device variants, not evidence
that the device path beats the normal backend.

On S9 (362,880 states; 37 Mahonian layers and exact final state), one
observation gave 0.428312 s for host hash and 26.252630 s for the parallel
device path. Device admission allocated 46 extents and 43,336,784 storage bytes
per rank; host used two extents and 1,884,208 bytes per rank. Thus the actual
allocated geometry differs, and neither this observation nor the S8 pairs
establish a result for large layers, eight GPUs, B200 or production capacity.
The single-CTA scheduler currently uses eight producer threads and one active
receiver thread per lane; candidates also scan extent descriptors. These are
observed properties of the implementation, not a separately measured
bottleneck attribution.

[Machine-readable comparisons and pressure observations](artifacts/2026-11-22-resident-hash-parallel.json)
include per-rank receipts. Direct native memory receipts exclude CUDA/NCCL
runtime overhead and are not observed whole-device peaks. The previous
[public API checkpoint](2026-11-22-resident-hash-public.md) records the
supported graphs and remaining limitations. Directed hash, matrices, resident
hash warp/batch, max_capacity, autotune, Explore, and a compatible eight-GPU
public device-path run remain unverified or unimplemented.
