# Integrated segmented and bitmap follow-up, 21 November 2026

This is an experimental source result, not a new pip release. `legacy` remains the default. The new source needs a matching native executable and CUDA library; the existing dev40 wheel does not contain these changes. B200/Blackwell and full S14 are not accepted by these tests.

## Artifact and scope

The [published source commit](https://github.com/TryDotAtwo/MultiGPUBFS/commit/f5d3da23c6d9be6b81f5724c1692fc6c33355878) was verified file-by-file against the saved source snapshot. The production CUDA/Rust executable was built from base `d268c179f0cfda4c331ff5b39afad256443c3b84` plus the patch and untracked source hashes in the evidence. Native executable SHA-256 `607f0746f237e0dba3855da5805e6a6b2cd01e9bece2b82327781f9b4231646b`; loaded CUDA library SHA-256 `9a717837a925614f5eef5fd6faddfdfc9db2dd775973f3eaf80986c117679fb8`. The receipt describes the exact native build; final Python control-path fixes and tests were layered on top. The evidence includes observed `/proc` mappings for eight workers during the corrected benchmark. None of the collision-control test-only binaries is a release artifact.

The host had eight RTX 3060 cards, each with approximately 12 GiB, using exact exchange and eight shards. In the corrected S12 cases: 8 devices, 4 owner lanes, `segment_rows=2^32` maximum per segment/bank, actual allocation quantum 1,048,576 rows, batch 8192, `balanced`, autotune off, archive settings in the invocation receipts. Throughput columns refer to the selected depth window and count newly accepted states and generated edges. All layers were checked by count against the Mahonian oracle; the final two complete layers were also compared by full state sets and checked for duplicates. Intermediate S12 states were not fully enumerated for comparison. Small examples were checked by full per-layer multisets.

| S12 case | Result / last complete depth | BFS seconds | depths 28–38 seconds | new states/s M | generated edges/s M | worst 1-s sample MiB | max live rows on one rank |
|---|---|---:|---:|---:|---:|---:|---:|
| bitmap / atomic, coordinated build | COMPLETE / 66 | 119.004 | 61.444 | 4.217 | 46.054 | 1497 | 9640704 |
| hash / warp, coordinated build | COMPLETE / 66 | 171.716 | 92.616 | 2.798 | 30.553 | 1497 | 9549905 |
| bitmap / atomic, earlier build | COMPLETE / 66 | 119.157 | 61.701 | 4.200 | 45.861 | 1497 | 9640704 |
| bitmap / warp, earlier build | COMPLETE / 66 | 117.487 | 62.111 | 4.172 | 45.559 | 1497 | 9640704 |
| bitmap / batch, earlier build | COMPLETE / 66 | 163.236 | 83.909 | 3.088 | 33.724 | 1497 | 9640704 |
| hash / atomic, earlier build | COMPLETE / 66 | 174.316 | 93.540 | 2.770 | 30.251 | 1497 | 9549019 |
| hash / warp, earlier build, unsafe race | COMPLETE / 66 | 162.371 | 86.984 | 2.979 | 32.531 | 1497 | 9549908 |
| hash / batch, earlier build | COMPLETE / 66 | 203.692 | 109.818 | 2.360 | 25.767 | 1497 | 9548040 |

### Recovery A/B repeats on the coordinated artifact

Each run uses the same S12 graph, eight GPUs, native executable/library, cyclic
bitmap layout, transport, archive and geometry. The only intended difference
within a backend is whether bounded batch recovery is enabled. It remains
opt-in via `MGBFS_SEGMENT_BATCH_RECOVERY=1`; no default change is claimed.
Execution order per backend was off/on/on/off/off/on; each row is a separate run. A sampled
VRAM maximum is not a measured instantaneous peak.

| Case | Result / last complete depth | BFS seconds | depths 28–38 seconds | new states/s M | generated edges/s M | worst 1-s sample MiB | max live rows on one rank |
|---|---|---:|---:|---:|---:|---:|---:|
| s12-ab-bitmap-off1 | COMPLETE / 66 | 128.779 | 68.979 | 3.757 | 41.023 | 1497 | 9640704 |
| s12-ab-bitmap-off2 | COMPLETE / 66 | 117.235 | 62.116 | 4.172 | 45.555 | 1497 | 9640704 |
| s12-ab-bitmap-off3 | COMPLETE / 66 | 123.835 | 65.002 | 3.986 | 43.533 | 1497 | 9640704 |
| s12-ab-bitmap-on1 | COMPLETE / 66 | 102.631 | 53.915 | 4.806 | 52.485 | 1497 | 9640704 |
| s12-ab-bitmap-on2 | COMPLETE / 66 | 95.172 | 50.267 | 5.155 | 56.294 | 1497 | 9640704 |
| s12-ab-bitmap-on3 | COMPLETE / 66 | 95.962 | 51.701 | 5.012 | 54.732 | 1497 | 9640704 |
| s12-ab-hash-off1 | COMPLETE / 66 | 178.031 | 97.337 | 2.662 | 29.071 | 1497 | 9548345 |
| s12-ab-hash-off2 | COMPLETE / 66 | 167.878 | 90.697 | 2.857 | 31.199 | 1497 | 9549908 |
| s12-ab-hash-off3 | COMPLETE / 66 | 168.179 | 91.052 | 2.846 | 31.078 | 1497 | 9547571 |
| s12-ab-hash-on1 | COMPLETE / 66 | 176.186 | 94.900 | 2.731 | 29.818 | 1497 | 9551292 |
| s12-ab-hash-on2 | COMPLETE / 66 | 173.486 | 91.933 | 2.819 | 30.780 | 1497 | 9551292 |
| s12-ab-hash-on3 | COMPLETE / 66 | 171.049 | 91.339 | 2.837 | 30.980 | 1497 | 9550772 |

For bitmap, median off/on was 123.835/95.962 s (the observed median difference was -22.5%; causation is not established). Off range 117.235–128.779 s; on range 95.172–102.631 s.

For hash, median off/on was 168.179/173.486 s (the observed median difference was +3.2%; causation is not established). Off range 167.878–178.031 s; on range 171.049–176.186 s.

**Later investigation:** In all six bitmap runs above, each rank reported 32,646 prefetched chunks and two route retries, regardless of the recovery setting. Those aggregate counters do not establish identical work per layer, but they do not support attributing the timing difference to fewer prefetched chunks. The measured advantage also did not reproduce on the subsequently packaged dev41 binary; see the [packaged follow-up](2026-11-21-dev41-package-sm86.md). The cause of the earlier timing difference remains unknown.

### S13 common-prefix check on the coordinated artifact

These are deliberately truncated prefix runs, not completed S13 calculations.
All completed layer sizes were checked against Mahonian counts; intermediate
full states were not compared on this large graph. Samples can miss VRAM peaks.
Evidence archive preparations and small transfers (39,850 and 13,523 bytes)
overlapped the bitmap and hash prefixes at about 17:35 and 17:43 UTC respectively,
so whole-run timings are not fully isolated.

| Case | Result / last complete depth | BFS seconds | selected window seconds | new states/s M | generated edges/s M | worst 1-s sample MiB | max live rows on one rank |
|---|---|---:|---:|---:|---:|---:|---:|
| s13-final-bitmap | INCOMPLETE / 35 | 455.198 | 113.792 | 4.533 | 50.781 | 2023 | 99062123 |
| s13-final-hash | INCOMPLETE / 35 | 723.064 | 187.485 | 2.751 | 30.821 | 4425 | 93453511 |

Both reached depth 35 with identical counts. Creation of depths 34–35 took 113.792 s bitmap and 187.485 s hash; a single pair does not estimate run-to-run variance.

The earlier six cells use different native artifacts, so the table is a historical comparison of complete implementations, not a controlled measurement of individual atomic instructions. The old hash/warp line is unsafe under the reproduced race. Each case is one run. The peak is the largest `nvidia-smi` sample taken once per second; short peaks can be missed. Segment allocation receipts are separately included and do not include all CUDA, NCCL, queues and scratch.

## Correctness and lifecycle

* The old hash/warp direct GPU race fixture produced two copies of the same row; the fixed version admitted one copy and replayed the deferred other state. The fixture uses a controlled CUDA schedule. Memcheck and synccheck passed for that focused fixture.
* On two GPUs, 36 S4 and directed C6 cases cover atomic, warp and batch, normal and deliberately colliding hashes, unconstrained/minimum/one-less capacity. Every published layer is compared as a full multiset with an independent CPU BFS. The zero-hash cases use one shard.
* The combined native BFS test with two active shards, same full-hash collision, duplicates arriving from different ranks, a forced temporary cap retry and same-origin replay passed for packed and wide states on two GPUs. Every completed layer and rank routing was checked against the independent CPU oracle. At the trigger capacities the jobs later stopped INCOMPLETE; separate minimum-sufficient runs completed and one-less runs stopped with exact prefixes. The test uses a deliberately forced schedule and hash; it is not a naturally reproduced production interleaving.
* Bitmap tests compare exact layers on orbit-reduced and repeated-symbol examples; persistent layout changes were checked with 2, 3, 5, 7 and 8 GPUs. Full-domain-product ranks form a closed superdomain, not necessarily the reachable set.
* On eight GPUs, injected failures at final write/publish, a missing cached communicator, post-seed and between count and payload closed the failed group; a separately started clean group completed. These injected cases do not prove all possible driver or network failure modes.
* A resident group's new graph and repeated graph reused the verified transport in the recorded small test. The initial preflight remains costly and checks a one-shard bounded prefix only, not the production geometry. The final source invalidates the preflight before a change of worker family; segmented hash and bitmap can reuse the same family.
* Rank allocations use checked wider totals, while each segment/bank retains a configurable 32-bit-local-row maximum. The `2^32` boundary was validated arithmetically and with a reduced physical boundary fixture, not by physically filling 2^32 rows on a rank.

## GitHub CI status

The [CPU contracts run for source commit f5d3da23](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/44927979200) passed the dependency-cache and CUDA library type-check jobs. In the contracts job, the CPU contract and type-check steps passed, then repository-wide `cargo fmt --check` failed. The [preceding dev40 run](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/44848945363) failed at the same formatting step; formatting of many unrelated existing files is outside this patch. The CI run is therefore not green, and it is not a GPU acceptance test.

## Memory and imbalance experiments

Under a **128 MiB segment-storage budget per rank**, not a cap on total VRAM, S12 bitmap completed in both presets; hash stopped with `RESOURCE_10` after completed depth 21. This is a controlled pressure point, not a universal maximum. On an earlier build without reduced-domain ranking, a synthetic S10 subgroup within an S14 representation gave worst-rank live rows of 745,192 for cyclic ownership and 93,591 for page-64, while BFS time increased from 1.585 to 1.744 seconds in that unconstrained pair. A separate 8 MiB segment-budget case stopped cyclic at depth 15 but page-64 completed. Cyclic remains default: remapping can increase traffic and hurt other graphs.

### Coordinated-build ownership and capacity comparison

These four cases use the same reduced-domain build and S10 subgroup inside S14;
allocation quantum and segment-storage budget differ between unconstrained and
pressure cases, so compare layouts only within each pair. The listed window is
chosen around each run's largest completed layer and is comparable only when
its depth range matches. Per-rank live-row and sample values are in the evidence.

| Case | Result / last complete depth | BFS seconds | selected window seconds | new states/s M | generated edges/s M | worst 1-s sample MiB | max live rows on one rank |
|---|---|---:|---:|---:|---:|---:|---:|
| skew-final-cyclic | COMPLETE / 45 | 1.207 | 0.455 | 5.292 | 46.601 | 633 | 95174 |
| skew-final-page64 | COMPLETE / 45 | 1.865 | 0.929 | 2.593 | 22.836 | 633 | 94043 |
| pressure-skew-final-cyclic | COMPLETE / 45 | 1.729 | 0.792 | 3.043 | 26.791 | 301 | 95174 |
| pressure-skew-final-page64 | COMPLETE / 45 | 1.885 | 1.028 | 2.345 | 20.646 | 301 | 94043 |

* skew-final-cyclic: per-rank peak live rows 92,438–95,174; counted remote records 13,063,680; padded-record diagnostic 152,053,440; bitmap bytes per rank [56704]; domain [3628800]; retries 0.

* skew-final-page64: per-rank peak live rows 92,349–94,043; counted remote records 18,833,770; padded-record diagnostic 87,296,982; bitmap bytes per rank [56704]; domain [3628800]; retries 0.

* pressure-skew-final-cyclic: per-rank peak live rows 92,438–95,174; counted remote records 13,063,680; padded-record diagnostic 206,478,720; bitmap bytes per rank [56704]; domain [3628800]; retries 3.

* pressure-skew-final-page64: per-rank peak live rows 92,349–94,043; counted remote records 18,833,770; padded-record diagnostic 87,296,982; bitmap bytes per rank [56704]; domain [3628800]; retries 0.

Remote-record counters describe the software transport path and are not hardware-measured link bytes. Padded-record counts are a diagnostic and should not be equated to transferred bytes when exact exchange is enabled.

### B200 memory arithmetic, not a hardware test

NVIDIA specifies 180 GB per B200 GPU and 1.44 TB per eight-GPU HGX node ([specification](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)). For the complete distinct-symbol Coxeter graph with adjacent transpositions and one byte per symbol, an independent Mahonian calculation gives:

| graph | group states | maximum layer | maximum sum of two adjacent layers | maximum sum of three adjacent layers |
|---|---:|---:|---:|---:|
| S14 | 87,178,291,200 | 3,727,542,188 | 7,455,084,376 | 11,141,606,753 |
| S15 | 1,307,674,368,000 | 50,626,553,988 | 101,253,107,976 | 151,421,412,675 |

For S14 the ideal bitmap lower bound is 10.897 GB of visited bits plus 14 bytes per row for the largest pair of layers: about **115.268 GB** in total. For a three-bank sorted-index hash with 12 index bytes plus 14 state bytes per row, the analogous lower bound is **289.682 GB**. These figures omit queueing, NCCL, CUDA, allocation rounding, imbalance and other reserves, and do not establish a fit on a particular node. The open-address hash geometry differs.

For S15 the corresponding two-bank bitmap lower bound is **1,682.256 GB**, greater than the advertised 1,440 GB on 8 B200 even before overhead. Under the inverse-closed three-bank segmented layout, state rows alone require **2,271.321 GB** at the maximum three-layer sum; the analogous 12-byte sorted-index model requires **4,088.378 GB** before other overhead. These limits apply to the assumed 15-byte rows and complete Coxeter S15; a different representation, a smaller reachable domain, or out-of-core execution changes the calculation. No S14/S15 B200 run was performed.

Evidence: [2026-11-21-integrated-followup-evidence.json.gz](2026-11-21-integrated-followup-evidence.json.gz), SHA-256 `1e23ff4d8e29f9f31ff644a5916b61cbbdce98ce35b16279d83f25e0876ae164`. It contains invocation and per-rank summaries, layer counts and timings, sampled VRAM maxima, native build receipt, benchmark source patch and untracked hashes, and selected test logs. Large graph state archives and full telemetry samples are not included.
