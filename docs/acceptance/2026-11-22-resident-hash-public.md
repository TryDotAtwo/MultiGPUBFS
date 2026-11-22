# Experimental resident segmented hash: public API gate

The new resident hash path is opt-in and **not a performance-ready default**.
It shares the resident device mailbox/epoch scheduler with pack4 bitmap. It
stores original u8/i64 states in the ordinary segmented hash layout and
compares complete states after hash collisions. Its first version serializes
owner transactions within each rank. It supports atomic, three-bank,
inverse-closed permutation actions of at most 16 elements, balanced admission,
without autotuning or a production deadline. Unsupported actions report an
error; no mid-job backend switch is performed. Existing host and bitmap paths
retain their existing defaults.

## Actual coverage

- Machine: four V100-SXM2-32GB (SM70), from an eight-V100 host; the full eight
  do not have all-pairs native P2P atomic support. CUDA 12.8.93, NCCL 2.30.7.
  This is not an eight-GPU acceptance or a B200 measurement.
- Parent of this checkpoint: `6499038d1a585139cabc55cafed5744ea62de3c7`.
  Tested `mgbfs` SHA-256
  `34173d5d674ce4abcd7050a0aa2d10017051c795e767152e8803ec29b106ba6e`;
  loaded `libmgbfs_cuda.so` SHA-256
  `cfd13b59be50bfc2e1657207d07ee889be14f3347ffdb043a825592d1e3e588d`.
- `tests/device_resident_hash_layers_gpu.py`: on both 2 and 4 GPUs, all
  complete layer **multisets** for S4 (7 layers), a repeated-symbol
  permutation graph (10), and a signed/wide-symbol graph (4) matched the
  independent CPU BFS. Tiny segments, sufficient/insufficient hard caps,
  a duplicate at full cap, and successive jobs in the same persistent worker
  processes were checked. A separate 2-GPU run forced **every full hash to
  zero** and checked those same layer multisets and capacity cases. In that
  collision run all keys share one owner and one shard; it is not a proof of a
  collision arriving from multiple ranks with two active shards.
- `tests/device_resident_hash_cayleypy_gpu.py`: actual `graph.bfs()` on 2 and
  4 GPUs matched all layer sizes and the complete final layer for S4 and a
  repeated-symbol graph; duplicate at full, a correct incomplete prefix at a
  tight rank cap, and a healthy next job in the same persistent processes
  passed.
- `tests/device_resident_hash_fault_gpu.py`: killing rank 1 during a 2-GPU
  run caused group failure with retained evidence in about 10.7 seconds,
  prevented reuse of the failed generation, and an explicit new group
  completed S4. `tests/device_resident_hash_guard_gpu.py` confirmed that a
  directed case is rejected and a new group can subsequently run.
- The existing resident bitmap full-layer oracle on 2 GPUs, the isolated hash
  owner CTest, four Rust extent-validation tests, and old/new native ABI
  capability checks passed on this build. The pre-existing 2^32 VMM probe is
  an addressability check, not a physical 2^32-row allocation.

Structured [oracle receipts](artifacts/2026-11-22-resident-hash-oracles.json)
and [S8 comparison](artifacts/2026-11-22-resident-hash-s8-4gpu.json) contain
the tested geometries and per-rank allocation receipts. Detailed per-rank
state snapshots and logs were also preserved outside the repository.

## Preliminary performance, one observation per variant

S8, adjacent transpositions, four V100 GPUs, two shards, configured segment
maximum 8192, initial batch 128, same native executable and CUDA library,
complete graph (40,320 vertices; 29 layer sizes matched the Mahonian formula;
final state independently checked). Device variants used eight lanes and eight
records per packet. Times are BFS core only; the first host launch also had
cold-start/preflight costs. No claim is made from a single observation about
repeatability or a different GPU architecture.

| Backend | Engine | BFS seconds | Allocated segment storage per rank |
|---|---|---:|---:|
| hash | host | 0.097 | 123,184 B |
| hash | device | 6.191 | 4,249,848 B |
| bitmap | host | 0.115 | 28,848 B + 1,264 B bitmap |
| bitmap | device | 1.149 | 259,632 B + 1,264 B bitmap |

These segment figures are allocation receipts, not total VRAM high-water.
Device hash preallocated 69 extents per rank versus two used by host hash in
this run. Each incoming candidate scans the extent list and the current
implementation has a rank-wide owner lease; both are potential contributors,
not separately isolated causes. Queue bytes were 344,160 per rank in each
variant. Native CUDA/NCCL allocations and an observed VRAM peak are not
included in the storage column.

## Reproduce

Use a matching CUDA/NCCL bundle supporting the resident ABI and peer topology;
set `MGBFS_EXECUTABLE`, `LD_LIBRARY_PATH`, and `PYTHONPATH` for the build.

```sh
export MGBFS_SEGMENT_TEST_BATCH=1 MGBFS_RESIDENT_LANES=4
export MGBFS_RESIDENT_PACKET_RECORDS=1 RESIDENT_TEST_WORLD=2
python tests/device_resident_hash_layers_gpu.py
MGBFS_SEGMENT_TEST_HASH_BITS=0 python tests/device_resident_hash_layers_gpu.py
python tests/device_resident_hash_cayleypy_gpu.py
python tests/device_resident_hash_guard_gpu.py
python tests/device_resident_hash_fault_gpu.py
```

`RESIDENT_TEST_WORLD=4` exercises four compatible GPUs. The comparison
`tests/device_resident_hash_ab_gpu.py` additionally expects
`MGBFS_CUDA_LIB_FILE`; its default is a single S8 pass, not a profiling matrix.

Full directed hash history, matrices, hash warp/batch, `max_capacity`,
autotune, Explore, a physical larger-layer comparison, and the compatible
eight-GPU device path remain outside this checkpoint.
