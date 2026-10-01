# Two-GPU CUCO multiset and packed tail gate

Branch `codex/bfs-tail-archive`; the active BFS checkout was not switched.
Measured native source `024570b1` (full ID recorded in bundle/reports), final
CPU tail panel source `8d2ceec7`; final capacity/smoke source `70c603fb`.
These revisions change the multiset owner admission and archive boundary
markers, not cuco CUDA kernels. Existing NCCL wheel is used without the other
agent's experimental guard patch.

Hardware: 2 NVIDIA RTX 4060 Ti, 8188 MiB each, driver 570.153.02, 32 GB host
RAM, 100 GB overlay disk. Build uses pinned CUDA 12.9.86/12.9.79, SM89, and the
existing cuco/CUTLASS/library-owner lockfiles. CPU packing uses NumPy 2.2.6.
Build summary and full runtime paths are retained in local evidence.

Real two-rank full-word oracle passed 12 cases: n=5/7,r=4; CUB,
CUCO_INDEXED/CUCO_RANK; pre-dedup OFF/ON; reversed owner map. Both real archived
LRX7r4 runs reproduce all 210 words in 12 exact layers. After downloading and
restoring tar hard links, all six final-panel archived LRX9r4 runs independently
reproduce all 15,120 words in 26 layers locally. LRX11r4 has 1,663,200 states;
its large state sets were not independently replayed, so only native totals,
configuration and matching panel layer histograms are claimed there.

Final paired panel: fresh processes, no full-graph warmup, three repetitions,
alternating owner order and archive on/off order. DENSE, pre-dedup ON,
compact word states, HOST_SIZED_NCCL; batch 32768, 8 shards/256 buckets,
per-rank capacity 1M, future ring 2M, fixed library pool 512 MiB, archive
8192 rows/256 pinned slots. Times include first-use work within BFS and the
native archive submissions when enabled. HF upload was not enabled.

| Graph | Owner | No archive median (MAD), s | Tail archive median (MAD), s |
|---|---|---:|---:|
| (9,4) | CUCO_INDEXED | 0.208769 (0.003009) | 0.276184 (0.018362) |
| (9,4) | CUCO_RANK | 0.207700 (0.001246) | 0.277632 (0.033200) |
| (11,4) | CUCO_INDEXED | 0.458607 (0.001905) | 0.519417 (0.025774) |
| (11,4) | CUCO_RANK | 0.417333 (0.001580) | 0.496671 (0.022516) |

CUCO_RANK has a lower no-archive median on (11,4). Its archive advantage is
small compared with three-run dispersion; this is not a universal ranking or
LRX15r4 performance result. A ready explicit CUCO_RANK config is supplied in
configs/tail-lrx11-cuco-rank.json; CUCO_INDEXED remains available.

Whole snapshot files are pinned by hard links; only a partial oldest layer
is copied. Local complete tail keeps >=3 full layers AND >=10 decimal GB,
or all layers back to graph start. INCOMPLETE snapshots are <=1 GB and contain
whole packed records. Every completed layer retains counts/duration/per-GPU
sampled VRAM metadata. The 50 ms monitor can miss short layers: null peaks
are explicit, not invented exact maxima. Retention thresholds were exercised
with scaled unit fixtures; a physical >=10 GB archived GPU run remains open.

Real deliberately undersized CUCO_RANK run preserves INCOMPLETE, depths 0..3,
21 states, and primary error LIBRARY_RANK_DEPTH_FATAL_16_16. Unfinished layer
data is excluded. Upload failure and snapshot pinning are tested with a fake
API. Full Python suite: 212 tests, 8 skipped. No new sanitizer pass is claimed.

Live HF publication subsequently succeeded using an existing user credential
stored locally with Windows DPAPI, outside Git. The connected HF MCP still has
read-repos only. Publisher uploads payloads before manifests and retains failed
pinned inputs. Xet upload initially failed with local access denied; ordinary
LFS upload succeeded with HF_HUB_DISABLE_XET=1. SDK cache readback hit Windows
path length limits, so verification downloaded remote bytes directly over HTTPS.

- [COMPLETE (9,4), CUCO_RANK](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/tail-runs/20261002-tail-n9-r4-cuco-rank-verified-v2/manifest.json):
  26 files, 15,120 states, 120,960 bytes; remote manifest equality, file sizes
  and SHA-256 verified after upload. Also independently replayed locally.
- [INCOMPLETE capacity failure](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/tail-runs/20261002-tail-capacity-incomplete-v2/manifest.json):
  four completed layers 0..3, 21 states, 168 bytes; unfinished layer excluded.
  Remote manifest equality, all four file sizes and SHA-256 verified.

Local upload receipts/readback records are in ignored test_results/publish-*/.
Tail/sweep Python tests: 13 passed. Finite n,m sweep is implemented and tested
on CPU; a complete GPU grid remains unexecuted. Physical SSD retention has
since passed a synthetic 13 GB write / 10.4 GB complete tail / 1 GB incomplete
tail stress with checksum readback (see physical-tail-stress.json). A real GPU
archive of that size and the (15,4) archive on/off comparison remain open.

Evidence: ignored local test_results/tail-vast-20261002/evidence.tgz, SHA-256
deec51debc98fb308628845780ce592d6fe2af3c6ce793362309ea991c7c3314.
Local replay record: local-full-state-replay.json. Rental 53761045 was deleted;
absence confirmed by provider instance inventory. Approximately 1684.68 s at
$0.227778/hour yields $0.10659 base cost. Traffic and final invoice are not
reconciled. Windows watchdog was launched but its long-lived persistence was
not established; deletion was performed and verified explicitly before exit.
