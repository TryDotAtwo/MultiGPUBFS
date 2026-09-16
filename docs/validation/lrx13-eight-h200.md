# LRX S13 on eight H200: complete search-only run

Source: `2893d6d54d8b6ba53a2327ce6b6ebf8ca2202518`.
Eight physical H200, all-pairs NV18 topology; native CUDA/NCCL data plane,
eight processes launched with `torchrun --no-python`. NCCL 2.28.9 pinned to
`dbc86fd06e8b0c4517b95d8958a09ccacf9520c9`, `NCCL_CUMEM_ENABLE=0`.

## Correctness

- Full-state/archived-state CPU-oracle comparison on S4 and U3(mod 3): all 32
  eight-device combinations passed (CUB/cuCollections, DENSE/HASH_FIRST,
  pre-dedup OFF/ON, two owner maps), 322.96 s. This was uninstrumented;
  it is not an eight-device sanitizer claim.
- S13: identity, L/R/X, **6,227,020,800 states**, 79 layers, diameter 78.
- Every layer count in the initial run and all five clean repeated runs matches
  `data/reference/lrx13-layers.json` from fedmug/lx-growth-distribution.
- Large-run verification is **layer counts only**, not full-state equality.
  Hash-based dedup retains its probabilistic contract.
- Archive explicitly disabled by user request: no state dataset, zero archive
  pinned/disk allocation, `durable_run_commit_seconds=null`. Layer statistics,
  rank reports, allocation plans, logs and external VRAM samples are preserved.

## Fixed configuration and measurements

DENSE, pre-dedup ON, compact permutation states (13 meaningful bytes, 16-byte
device stride), batch 262144, 8 shards, 256 buckets, 4 job buckets. Per-rank
layer capacity 60M, state ring 120M, bucket capacity 2M; 1 GiB untouched reserve.
cuCollections uses a **16 GiB fixed pool per rank**, included in full VRAM.
These are fixed-capacity runs, not minimum-memory tuning.

| Run | Backend | BFS seconds | Total peak VRAM, GiB |
|---|---|---:|---:|
| S11, one calibration sample | CUB | 1.471159 | 76.07 |
| S11, one calibration sample | cuCollections | 1.311137 | 195.52 |
| S13, initial verified run | cuCollections | 11.700875 | 195.55 |
| S13, five clean runs, median | cuCollections | **11.716894** | **195.55** |
| S13, diagnostic attempt | CUB | **TIMEOUT at 180 s wall time** | 76.10 |

Five clean times (seconds): 11.647830896, 11.737633199, 11.640318268,
12.196390344, 11.716893623. MAD: **0.069062727 s**.
Per-rank peak MiB: `[25019,25067,25067,25067,25067,25067,25067,24827]`.
Full VRAM is externally sampled with nvidia-smi at 50 ms; brief peaks between
samples can be missed. Explicit allocation planes and setup/final cudaMemGetInfo
readings are also preserved. The latter are not continuous peak measurements.

Timer excludes process/build/setup time but includes first-use work during BFS;
`warmup_completed=false`. Do not call these fully warmed-kernel timings.
CUB timeout has no completed layer histogram or valid BFS time; it is not a
speedup denominator. Its cause remains unresolved. cuCollections is neither
universally fastest nor the lower-memory choice, as the S11 results demonstrate.

## Excluded measurements / runner defect

The CUB timeout killed the torchrun launcher but left eight rank processes in
independent process groups. The next repeat series briefly overlapped them.
That entire series (`s13-cuco-repeats`) is excluded. Exact observed rank PIDs
and its orchestrator were killed; process inventory then showed no GPU jobs
and all eight GPUs reported 0 MiB. Only afterward was `s13-cuco-clean` launched.
The initial complete S13 precedes this incident and is not contaminated.

The runner now launches commands through a dedicated Linux subreaper in
`scripts/process_scope.py`. Two real Linux CPU tests reproduced leaking ranks
on both timeout and launcher exit, then passed after the fix. The gate's four
tests and the metrics runner's 25 tests also passed in Linux. This runner fix
was made after the measurements; the measured native binary was unchanged.

## Evidence

Ignored local directory: `test_results/vast-51251891/evidence/`.

- `eight-rank-oracle-pass.log`, SHA256
  `3880a7bbcbaff7fce28f41a566526d9e81e5b46129ff8f6e1e777503249f0db0`.
- `search-results-retry/`: original sequence, all rank reports, S13 verification,
  device inventory/topology and per-run external memory samples.
- `s13-cuco-clean/comparison.json`, SHA256
  `cd83abe0c6225b09421fbbfd3afc4637a2c135b18ca092948bae311f6c018af2`:
  all five rank-report sets, verified histograms, timings and statistics.
- `s13-comparison/`: CUB timeout evidence; `s13-cuco-repeats/`: excluded overlap.

The eight-H200 instance was deleted after results were saved. Absence was
verified by the provider instance list and independently by the watchdog.
Elapsed rental: approximately 2184.98 s; reported hourly rate: USD 37.50548246;
estimated base rental: **USD 22.7635**. The three project rentals total roughly
**USD 43.39 base cost**, before invoice/traffic reconciliation. These are not
verified final invoice amounts. No instance is intentionally retained for the
subsequent multiset implementation work.
