# Experimental resident BFS on four V100s — 2026-11-23

This is an experimental branch, not a production release or a B200 acceptance.
The checked source for the new hash run is commit
[ceb7760](https://github.com/TryDotAtwo/MultiGPUBFS/commit/ceb7760fc49fad7cbbf0263a07edbccf48d7f820)
(tree `c01f4bf26d128a0222b8cb7f5c5e333df7231086`).
The complete layer-size arrays, geometry, binary hashes, per-rank allocations
and measurement scopes are in [the machine-readable receipts](2026-11-23-v100-resident-evidence.json).

The machine had four Tesla V100-SXM2-32GB GPUs, driver 580.178.04,
CUDA 12.8.93 and NCCL 2.30.7. GPU pairs were connected by NVLink
(NV1 or NV2). Each cell below is **one observation**. The resident and host
hash runs use different binary revisions, so their ratio measures the combined
change, not the isolated effect of any one optimization. Time is BFS time;
VRAM is the maximum of 0.5-second NVML samples per GPU, not a guaranteed
instantaneous allocation peak.

| Backend | Graph | Host BFS, s / sampled MiB per GPU | Resident BFS, s / sampled MiB per GPU | Status |
|---|---|---:|---:|---|
| permutation bitmap | S12 | 165.940 / 934 | 5.122 / 5052 | Both COMPLETE |
| permutation bitmap | S13 | 2143.999 / 2640 | 80.933 / 9350 | Both COMPLETE |
| segmented hash | S12 | 130.350 / 1750 | 19.467 / 5826 | Both COMPLETE |
| segmented hash | S13 | 1877.504 / 13014 | 349.291 / 30450 | Both COMPLETE |

All cases use adjacent transpositions from identity. S12 completed 67
Mahonian layers and 479,001,600 states; S13 completed 79 layers and
6,227,020,800 states. All counts were checked against independent Mahonian
coefficients. On COMPLETE the exact last state was the reverse permutation.
These large runs establish layer **counts** and exact terminal state, not full
state sets at every intermediate layer. Different backends have different
internal geometry; compare like backends and graph parameters.

The S13 hash benchmark used 32 shards, 1,048,576 rows per segment bank, a
100,000,000-state per-rank next-layer cap, four virtual buckets per shard,
262,144 records for each send/receive tile, packed 8-byte permutation rows,
and opt-in rehash. The four rank allocations contained 253 segments each.
For rank 0, requested state storage was 6,339,930,576 bytes, hash index
storage 23,234,587,088 bytes and total segment storage 29,574,523,736 bytes;
the separately reported reserve was 2,147,483,648 bytes. Scratch and the
mailbox were about 14.7 MB and 4.2 MB respectively. The measured run reserved
most GPU memory ahead of use: this is not a demonstration of demand-grown
storage or of maximum usable capacity. Peak simultaneous live rows were not
instrumented in this run.

On a separate diagnostic S12 run, the virtual-bucket version recorded up to
80 active owner CTAs across the 80 SMs and up to four active CTAs on one
physical shard; instrumentation was disabled for the reported speed run.
For S12 on the same rehash binary and geometry, one virtual bucket without
rehash took 26.157 s, one with rehash 25.153 s, and four with rehash 19.467 s.
These single observations do not justify a default change.

## Correctness checks

* Through the public API and persistent workers on four physical GPUs, S6
  adjacent transpositions were compared layer-by-layer to exact CPU state
  multisets, including normal hashes, forced `hash_bits=0`, small segments,
  resource limits and a healthy subsequent job. The forced-collision run
  reached 156 rehashes on one rank.
* A separate test-only projected-hash build checked all 11 S5 layers, two
  active shards on one rank, 41 distinct states with an identical full
  projected hash, resource caps of 9 and 8, and reuse of the same persistent
  worker group. There were 44 children with local and remote parent ranks,
  **derived from the exact saved rank-layer states and graph edges**, not
  counted by a direct transport trace. The test-only projection is not part of
  the published production binary.
* The archived small-graph checks also cover repeated and wide symbols.
  A 32-bit row is local to one segment; aggregate counters are wider. No test
  here physically allocated 2^32 rows, and this result does not establish S14
  capacity on B200.

For the rehash layer oracle, after building the pinned source with SM70 and
the LSA-enabled native library, the essential invocation is:

```sh
export PYTHONPATH=.:tests
export MGBFS_EXECUTABLE=/absolute/path/to/mgbfs
export MGBFS_CUDA_LIB_FILE=/absolute/path/to/libmgbfs_cuda.so
export MGBFS_TEST_TMP_ROOT=/absolute/path/to/evidence
export RESIDENT_TEST_WORLD=4 RESIDENT_TEST_SCHEDULER=batch
export MGBFS_SEGMENT_TEST_BATCH=1 MGBFS_RESIDENT_HASH_REHASH=1 MGBFS_RESIDENT_HASH_VBUCKETS=4
python3 tests/device_resident_hash_rehash_gpu.py
MGBFS_SEGMENT_TEST_HASH_BITS=0 python3 tests/device_resident_hash_rehash_gpu.py
```

The current resident transport uses one bounded LSA window, not an A/B
pipeline that overlaps production and owner acceptance. Automatic pool
growth and replay after physical extent exhaustion are still open. Explore
is a separate unimplemented mode. Four V100s do not constitute an eight-GPU
acceptance, nor do they predict Blackwell/B200 performance.

The locally retained full rehash evidence archive is
`multicta-rehash-evidence.tgz`, SHA-256
`449eb8710888ed6fbe1870b7bde13a2cd40435270ff9daadb35cf98fc89ca21f`.
Its compact receipts archive has SHA-256
`66ab30925d1ff2e25d17a9ba1cdecca659500658daa290b16081bbc2d921b0ee`.
