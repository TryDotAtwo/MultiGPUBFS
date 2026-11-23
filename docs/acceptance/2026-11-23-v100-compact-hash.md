# Compact hash slot positions on four V100s

This measures the existing opt-in `MGBFS_SEGMENT_COMPACT_POS=1` path for
the experimental resident segmented hash. It stores a slot position in 32 bits
only when all positions for that segment fit; there is no sentinel in a live
row's position. Larger tables retain 64-bit positions. The limit is local to
one segment's slot table, not to all rows on a GPU.

All four cells used the same binary and four V100-SXM2-32GB GPUs, adjacent
transpositions from identity, four virtual buckets per shard, rehash, and a
single-slot LSA transport. S12 used 32 shards, a requested maximum of 262,144 rows per segment bank
and an 8M next-layer cap per rank; its 128 admitted extents each held 249,983
rows. S13 used the same shards, a requested maximum of 1,048,576 rows and an
80M cap. On rank 0, the ordinary case had 251 full-sized extents and two of
699,050 and 272,148 rows; all 260 compact-case extents had 1,048,576 rows. Send and receive tiles were each 262,144 records. The S12
ordinary run belongs to the preceding rehash experiment; both S13 runs were
consecutive. These are single observations.

| Graph | Position | Complete BFS, s | Max 0.5 s NVML sample, MiB per GPU | Allocated state + index bytes on rank 0 | Extents per rank |
|---|---|---:|---:|---:|---:|
| S12 | 64-bit | 19.467 | 5826 | 3,683,382,272 | 128 |
| S12 | 32-bit | 19.599 | 5570 | 3,299,408,384 | 128 |
| S13 | 64-bit | 352.977 | 30450 | 29,574,523,736 | 253 |
| S13 | 32-bit | 354.260 | 28242 | 27,262,982,240 | 260 |

Allocated state + index includes the segment counters in the reported total;
it excludes reserve, mailbox, scratch and other CUDA/NCCL allocations. Rank 0
index allocations in S13 were respectively 23,234,587,088 and
20,719,861,760 bytes; state allocations were 6,339,930,576 and 6,543,114,240.
The number of allocated extents changed because the admissible allocation fit
changed. The NVML sample is not a guaranteed transient peak.

Both cases completed, with all 67 or 79 layer counts checked against
independent Mahonian coefficients and the exact reverse terminal state checked.
This does not compare every intermediate state set of the large graphs. On
four physical GPUs, S6 was independently checked as full per-layer state
multisets with both normal and forced equal full hashes, small segments,
resource limits and a following job in the same persistent workers.

The compared executable SHA-256 is
`137412dc2f0a4e2e186a38dc2e5911b0df02d00e1b38020e726cbe9cf52ea433`,
the CUDA library SHA-256 is
`9d264b877dfb89c50e0e35732c5dcee632d1df28ccb6de516d36268035e88022`.
The source code matches tree `c01f4bf26d128a0222b8cb7f5c5e333df7231086`,
published as [commit ceb7760](https://github.com/TryDotAtwo/MultiGPUBFS/commit/ceb7760fc49fad7cbbf0263a07edbccf48d7f820).
The full source, logs and checks for the compact run are archived as
`multicta-compact-position-evidence.tgz`, SHA-256
`3994300fe262a26cc0ecf6d86a389a2a675b4999d4383457405dee1029bb56ea`.
[Machine-readable receipts](2026-11-23-v100-compact-hash-evidence.json)
include layer arrays, each rank's allocation breakdown and sampling metadata.

This comparison does not measure maximum reachable graph size under equal
VRAM pressure, dynamic growth, eight-GPU behavior or B200 performance. The
resident path and this memory setting remain experimental.
