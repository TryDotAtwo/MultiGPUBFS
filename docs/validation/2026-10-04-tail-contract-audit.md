# Final tail contract audit

Branch: codex/bfs-tail-archive. Implementation through 21a6797; subsequent
changes in this audit are documentation only. No changes were made to another
agent's checkout. GPU states remained on rented hosts/HF; only metadata reports
were read on the workstation. All owned rentals have confirmed API absence.

| Requirement | Implementation and authoritative evidence | Scope |
|---|---|---|
| Startup VRAM admission, without repeated reserve work in BFS | run_auto_tail.tune_pair, vram_autotune, native constructor queries; 12 consecutive two-rank queries passed | Startup-only queries, native pool bound includes fragmentation slack; not an absolute reachable-state capacity proof |
| Use almost all available VRAM | Automatic and near-full gate, two A4000, 98.553% allocated memory independently observed per GPU | Default 256 MiB reserve; explicit smaller reserves measured separately; occupancy does not prove bandwidth saturation |
| Full 32-batch CUDA Graph windows | distributed_native graph bridge, graph/direct full-state parity; 22 actual windows/rank in n15r4 samples | NCCL_LSA DENSE rank-owner path; unsupported paths use direct batches; short tails direct |
| Measure speed, select graph only on measured benefit | Alternating six matched n15r4 prefix samples: direct 3.577442398 s, graph 3.480637629 s; smaller calibration keeps direct on regression | 2.706% improvement in measured case; no universal zero-overhead claim |
| Compare archival cost and owner choice | Earlier n15r4 three repeats/owner/mode: rank .462/.647 s, indexed .521/.742 s without/with archive | Identical 26-layer prefixes on two3090; rank selected for measured configuration |
| Current archival cost | Three n15r4 matched pairs: archive 3.500298535 s, search-only 1.261651098 s | 2.774379 ratio on two A4000; archival overhead remains, distinct from startup VRAM admission |
| Automatic n,r grid and intelligent pruning | Automatic 527 pairs; 147 attempted, 236 fixed-r pruned, 144 structural exclusions; no pending | Small-buffer diagnostic proves policy; separate native admission gates prove large-buffer selection |
| Full correctness and incomplete prefixes | 117 complete graphs and all30 stopped prefixes independently compared with CPU word sets; n15r4 layers0..23 independent CPU hashes | Remaining n15 depths24..33 cross-mode parity only |
| Parameters, packing, reproducibility | TailArchive manifest graph/start/actions/packing, commit, full launch configuration and executable SHA256; launch/admission reports | 8 bytes n<=16, 16 bytes17..32, alphabet<=16, native orbit<u64 |
| Counts, whole-layer timing, GPU peaks | completed_layer writes all completed layer metadata; existing rank advance durations; separate 50 ms nvidia-smi monitor | Null when no sample lands in a layer; sampled peak rather than continuous maximum |
| COMPLETE whole-layer tail | At least three full layers; evict only if the remaining full tail still has >=10 decimal GB | Physical synthetic 13 GB stress retained four whole layers10.4 GB, checksums verified; not a real10GB complete GPU graph |
| INCOMPLETE suffix | snapshot packs <=1 decimal GB of completed layers; preserves first ordinal/full-partial metadata | Physical1GB suffix readback and actual failed searches checked |
| Interim durability and no archive VRAM history | SSD working tail retains COMPLETE-capable window; interim snapshots after completed layers; pinned RAM queue, no retained device history | Final temporary rank spools released only when all readers stopped; unit tests passed on Linux |
| Large shared Parquet and file descriptions | tail_cohort:20-case/2GB whole-case groups,512MB uncompressed shard target,262144-row groups, Zstd; depth spans/ordinal/fullness/bytes/SHA | Nine shared physical files for147 diagnostic cases; small per-case JSON manifests remain |
| Ordered HF publication and bounded SSD backlog | Payloads before manifests/ledger; pinned readback verifies bytes/SHA before release; case-boundary backlog control | Real count/byte-boundary release and retry gates pass; byte threshold reduced for semantic test |
| Convenient analysis without workstation state download | analyze_tail uses HF range requests, bounded batches, n/r/depth filters, conservative row-group skipping | Actual remote HF reads passed for8/16-byte cases; missing statistics safely fall back to batch filtering |
| No excluded derived output | Case manifest stores required layer metadata and state files; no paths/predecessors, duplicate percentages/reasons or separate generation/dedup/exchange timings | Diagnostic native logs are distinct from the analytical state schema |

## Evidence

- [Automatic, release, entire grid and near-full measurements](2026-10-03-automatic-and-release.md).
- [Query rendezvous](2026-10-03-memory-query-rendezvous.md).
- [Reserve and wide-state gates](2026-10-03-reserve-and-wide-state.md).
- [Physical retention and earlier owner comparison](../bfs-tail-archive.md).
- [Remote analytical reader and final Linux Python gates](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/e597e44bfa50a744d60b7b1aaaed808fe9ec25cc/evidence/20261004-remote-analysis/report.json).

Unverified GPU properties: heterogeneous GPU groups, long-duration near-full
allocation stability, universal graph benefit, maximal bandwidth/SM utilization,
and a real >=10GB COMPLETE BFS on these rented cards. They are not promoted to
verified results. Hardware-specific calibration remains necessary at startup.
This audit establishes implementation and measured validation of the requested
contract, not zero archival cost or successful enumeration of every graph.
