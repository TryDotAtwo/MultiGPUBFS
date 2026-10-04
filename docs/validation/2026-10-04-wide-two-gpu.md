# Two cheap two-GPU expansion gates, 2026-10-04

Source: `3fd7730811f6674c6f35211b3c2d1694bf1c6cda`, branch
`codex/bfs-tail-archive`. Both native binaries have SHA-256
`f278f2c2061f3fd54f02380e00da0beac90f33f6df7bab2763a5ccd10dd5a952`.
CUDA12.9, NCCL2.30.7, SM86, CUCO_RANK, pre-dedup ON, two ranks.
One compiled runtime handles all n/r cases; no per-pair compilation.
Detailed measured results and immutable HF revisions are in the adjacent JSON.

## Correctness and publication

Both 2xRTX3060/12GB (driver595.71.05, machine137174) and
2xRTXA4000/16GB (driver595.91.07, machine151169) passed:

- Full independent CPU-state/layer oracle for `(33,32)`, `(65,64)`,
  `(65,63)`, `(128,127)`.
- Independent oracle for 12 completed layers of `(17,1)`, `(27,12)`,
  `(128,1)`. These intentionally stopped prefixes are INCOMPLETE.
- Native warmed memory admission and production runs for `(17,1)`,
  `(33,32)`, `(65,64)`, `(128,127)`, again checked against CPU layers.
- Three repeated COMPLETE `(12,4)` runs, 55 layers and19,958,400 states
  each; identical full-state fingerprints within each model. This parity
  check is distinct from an independent CPU enumeration of this large case.
- Actual resource stop with64 rows/rank: `(16,1)` stopped after depth7
  with capacity16_16, `(17,1)` was skipped at fixedr1; `(16,15)`,
  `(17,16)`, `(65,64)`, `(128,127)` still completed. All attempted cases
  passed independent CPU-state checks. Fixed-r pruning is explicitly heuristic.

Seven wide manifests/model and five resource-stop/control manifests/model
were uploaded with shared Parquet payloads, then read back at immutable HF
revisions with all SHA-256 values and sweep ledgers verified. One representative
full `(12,4)` archive/model was also delivered as one123,024,404-byte Parquet
file with full readback. Repeat runs have identical states; their timings and
configuration evidence are retained in HF reports. States stayed on GPU-host
RAM/SSD and HF; only small metadata reports were retrieved to the workstation.

HF evidence reports are under `evidence/20261004-wide-3060/` and
`evidence/20261004-wide-a4000/` in
[the dataset](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results).
The adjacent JSON pins every delivered payload/manifest revision.

## Timing

Seconds; paired rows use the median of three complete `(12,4)` runs.
Native search uses the slower of two rank reports, not process wall time.
It includes the production BFS path with archiving enabled.

| Measurement | 2x3060 | 2xA4000 |
|---|---:|---:|
| `(12,4)` native BFS | 2.581687 | 2.648676 |
| `(12,4)` runner wall | 15.926821 | 15.400281 |
| `(12,4)` wall minus native BFS | 13.352177 | 12.743299 |
| Seven wide cases, native BFS sum | 1.895297 | 2.310310 |
| Seven wide cases, runner sum | 21.880969 | 19.322908 |
| Seven wide cases, between-pair gaps sum | 0.018434 | 0.026862 |
| Four automatic cases, native BFS sum | 1.099695 | 1.332178 |
| Four automatic cases, runner sum | 97.353462 | 83.825512 |
| Four automatic cases, admission sum | 84.107692 | 72.348598 |
| Four automatic cases, between-pair gaps sum | 0.009126 | 0.012845 |

Cold `(17,1)` admission alone took76.24/65.93 seconds, selecting
51,431,408/69,793,428 rows/rank respectively. Other automatic cases were
small orbits and used256 rows/rank, rather than wasting all VRAM. Subsequent
`(12,4)` admission took about8/7 seconds. Preparation remains substantial for
short graphs; millisecond sweep bookkeeping must not conceal this overhead.
Search timings here do not establish archive-on/off overhead or Graph32 speed.

VRAM was sampled independently with requested50ms cadence; short layers often
contain no observation and correctly retain null peaks. One A4000 automatic
prefix layer observed16,451,108,864 bytes on each GPU. These are sampled peaks,
not an exact high-water mark or proof of maximum hardware capacity.

## Failures and boundaries

The first wide run exposed `ARCHIVE_EXTENT_OVERFLOW` before the first layer
for enormous orbits. Fixed3fd7730: FIFO streams use a checked u64 logical
offset limit; their consumer bounds physical staging. Nonstream full archives
still reject overflowing reservation sizes. All13 local archive tests pass,
including sequential-write/overflow protection and a stalled FIFO consumer.
The fixed large-orbit prefixes passed on both GPUs.

Standalone CUDA Graph smoke passed on both GPUs. Forced two-rank LSA tests
failed: 3060 reports `LSA_PREPARE_GROUP: CUDA_STATUS_4` and its P2P matrix is
unsupported; A4000 has P2P access but reports `LIBRARY_RANK_DEPTH_FATAL_22_22`.
Normal startup selects HOST_SIZED_NCCL after these failures, and every successful
gate above used that transport with graph batches0. No Graph32 throughput
claim, LSA correctness claim, or B300 readiness claim follows from these runs.
These diagnostics refer to this pinned branch, not newer concurrent main code.

Default grid generation contains8255 pairs with2<=n<=128,1<=r<=n.
This session tested selected wide cases and real resource pruning; it did not
execute all8255 graph traversals or exhaust the three intentionally huge graphs.
The small delivered graphs do not repeat the earlier10GB retention stress test.

## Rental lifecycle

Owned instances54127273 and54127274 were deleted after all jobs were terminal,
HF readback finished and transferred token files were removed. Follow-up API
GETs confirmed both absent. Quoted compute-plus-disk estimate: $0.06418 and
$0.09091, about$0.155 total, excluding traffic. Actual billing is unknown because
the key lacks billing_read. No other owner's rentals/checkouts were modified.
