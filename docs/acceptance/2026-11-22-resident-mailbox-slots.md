# Experimental GPU-resident mailbox capacities

This change enables 1, 2 or 4 slots per resident device ring via
MGBFS_RESIDENT_MAILBOX_SLOTS; unset remains 1. It is opt-in on the existing
experimental device engine, not a new production default. The selected capacity
is agreed by all ranks before mailbox allocation and is recorded per rank.
Older CUDA libraries continue to work with one slot and explicitly reject 2 or
4. Startup validates payload, wraparound and returned credits in the actual
LSA window before the resident job starts.

## Correctness coverage

On four physical V100-SXM2-32GB GPUs (SM70, CUDA 12.8.93, NCCL 2.30.7),
independent CPU BFS was compared against the full multiset of states at every
finished layer on 2 and 4 GPUs, using four slots. Cases included S4, repeated
symbols and wide symbols for the hash; S4, directed C6 and repeated symbols
for bitmap. Both included tight per-rank caps and a duplicate at a full cap.
A separate 2-GPU hash run forced all hash bits to zero. On four GPUs the public
CayleyPy persistent path also switched slots 1→2→4→1 without changing worker
PIDs for both backends. The 2-GPU LSA ring test covered capacities 1, 2 and 4,
delayed receiving, two epochs and wraparound. These are small-graph checks, not
a complete 8-GPU or B200 acceptance.

With four slots on two GPUs, terminating one hash rank produced a group error,
invalidated the persistent generation and allowed a clean explicit new
generation. This checked fault path is not an exhaustive device-failure proof.
Compatibility used the earlier actual CUDA library with SHA-256
c936024ea15ac6b3cd969dc4a5c8d3d5775be35da05ee456aa81d3417d46c94b.

## S9 timing

Same executable and library for all runs: four V100 GPUs, S9 adjacent
transpositions, 362,880 states, 37 layers, 2,903,040 generated edges, two
shards, 8192 rows/segment, batch 128, eight device lanes and eight records per
packet. Slots were tested in the order 1, 2, 4, 4, 2, 1 in one persistent worker
group per backend. No NVML sampler ran during these measurements.

| Backend | Slots | BFS seconds (two runs) |
| --- | ---: | --- |
| segmented24 | 1 | 13.583, 12.436 |
| segmented24 | 2 | 11.879, 11.952 |
| segmented24 | 4 | 11.763, 11.749 |
| permutation_bitmap | 1 | 11.326, 10.838 |
| permutation_bitmap | 2 | 8.686, 8.665 |
| permutation_bitmap | 4 | 8.227, 7.569 |

All 37 layer counts were checked against independent Mahonian coefficients and
the final state against the reversed permutation. The full sets of intermediate
S9 states were not saved or compared; full multisets were checked on the smaller
cases above. The first run includes a cold worker, so these two observations do
not justify choosing a universal default. The two-run mean decreased by about
9.6% for hash and 28.7% for bitmap between 1 and 4 slots on this machine.

## Larger single pair and normal backend

On S10 adjacent transpositions (3,628,800 states and 46 layers) the hash
resident engine used the same geometry and binary, with 4 slots followed by
1 slot. BFS times were 318.213 s and 351.318 s respectively: about 9.4% lower
for four slots in this single pair. All 46 Mahonian counts and the final
reversed state were checked; full intermediate state sets were not compared.
The storage allocation receipt was 429,205,920 bytes per rank for both; this
is not a total VRAM peak. The result does not establish a default or a speedup
on other GPUs or graphs.

For context, the existing host-driven backend on the same four GPUs and native
binary, S9, requested two shards and 8192 rows per segment, produced the same
Mahonian counts and final state. Its hash BFS times were 0.362 and 0.333 s; its
bitmap BFS times were 0.361 and 0.339 s. The resident four-slot times in the
table above are therefore substantially slower on this workload. Actual
admitted storage differed: host hash and bitmap receipts were 990,170 and
246,666 bytes per rank respectively, versus the resident storage receipts
listed above. These are end-to-end backend comparisons with different actual
geometry, not an isolated estimate of transport cost. First-run launch wall
time also includes cold setup; the table reports BFS core, not setup.

In the S9 runs, the reported mailbox allocation per rank was 8,320 / 14,464 / 26,752 bytes for
1 / 2 / 4 slots. The storage allocation receipt per rank was 40,101,885 bytes
for hash and 2,302,272 bytes for bitmap; these are components, not total peak
VRAM. No sampled or exact process-wide VRAM peak was measured. Blocked-send
attempts are counts of retries, not elapsed time. Network interruptions were observed before the S9 runs; their cause is unknown.

Native identity for the comparison: executable
e304407d31c719f266b17726f63ad1ec036a9201099cd4a6aa60f266f73dd732,
CUDA library 8f242e79d5de55a115efb3ba1062e938db507fba2c2a13da4ceedea5ac57750c.
The source base before this patch was
634de0a2337a2bafe86a7964e183dc10ee3dc908.
Raw logs, exact outputs and binaries were saved as the prebench archive
0bc83982b8c343a7c8f473a650f9fc45cf99b93c982892a0cfc924ef987d22bf
and S9 evidence archive
1e832008214889ee07678cbb6a0bc5710c9a3316791e1aad4d58f8954f0db46a.

To reproduce on a compatible physical 4-GPU LSA topology, point the binary and
CUDA library variables to the built artifacts and ensure the library directory
is first in LD_LIBRARY_PATH. Then run:

~~~sh
export PYTHONPATH="$PWD"
export MGBFS_EXECUTABLE=/path/to/mgbfs
export MGBFS_CUDA_LIB_FILE=/path/to/libmgbfs_cuda.so
export LD_LIBRARY_PATH=/path/to/native/lib:/path/to/nccl/lib:/usr/local/cuda/lib64
export MGBFS_TEST_TMP_ROOT=/path/to/evidence
export MGBFS_RESIDENT_LANES=8 MGBFS_RESIDENT_PACKET_RECORDS=8
export MGBFS_RESIDENT_HASH_SHARD_OWNERS=1 MGBFS_RESIDENT_HASH_INDEXED_EXTENTS=1
export RESIDENT_AB_WORLD=4 RESIDENT_AB_N=9 RESIDENT_AB_SLOTS_ORDER=1,2,4,4,2,1
RESIDENT_AB_BACKEND=segmented24 python tests/device_resident_mailbox_bench_gpu.py
RESIDENT_AB_BACKEND=permutation_bitmap python tests/device_resident_mailbox_bench_gpu.py
~~~

The benchmark records all rank receipts and checks the loaded library SHA.
Full CPU-oracle tests are separate:
tests/device_resident_layers_gpu.py and
tests/device_resident_hash_layers_gpu.py.

[Machine-readable timing and correctness receipts](artifacts/2026-11-22-resident-mailbox-slots.json)
include per-rank allocation components; the full logs are retained in the
evidence archives above.
