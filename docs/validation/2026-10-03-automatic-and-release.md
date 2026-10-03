# Automatic sweep and remote cohort release

Hardware: two RTX A4000 on machine 149741, CUDA 12.9, NCCL 2.30.7.
Native build: `ab2b3abcdf99e657380bb1c8f4e95a4caf25657d`.
The automatic gate used that pinned build; release code came from `8572c2e4`.

## Automatic supported grid

The entry point received no n/r range. It generated all 527 format-domain pairs
for n=2..32, classified packing/u64 exclusions and ran until its compute deadline.
It attempted 35 cases: 34 COMPLETE and one honest empty INCOMPLETE startup
failure at the capacity-probe deadline. All 34 complete cases passed independent
CPU full-word/layer-set comparison. The final ledger retains 348 pending pairs.
No hardware resource pruning was reached in this maximum-capacity gate.

Two publication generations exercised reuse of the first immutable 20-case
group. Final HF readback verified all 35 case manifests, both shared Parquet
files (583,052 physical bytes), every SHA256 and the complete sweep ledger.
This is a successful bounded end-to-end gate, not full-grid completion.

Evidence: https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/ea37384a94437f5b73f945dda1a91cd2e1a7732e/evidence/20261003-automatic-guard/report.json

## Readback-before-release

The release gate used real already-verified GPU archives on the same host.
For the production count boundary it verified and released the first 20-case
shared group, retained the open suffix and its metadata, republished without
repacking the released group, then verified all remote files and the ledger.
A separate publication copied three real cases with hard links on the GPU host
and used a deliberately reduced 1000-byte group target. Each byte-closed group
was verified and released; a repeated publication uploaded zero payload files,
and final HF readback still passed. This proves the byte-bound branch and
receipt reuse, not a physical 2 GB stress test.

Only GPU-host packed/Parquet state files were unlinked, after pinned-revision
manifest and streamed full-checksum verification. Native reports, launch
configuration, raw and published manifests, and durable release receipts remain.
No states were downloaded to the user's computer.

Evidence: https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/0d2e67bcce4dd6039e64b1441794db91c2002276/evidence/20261003-cohort-release/report.json

CPU validation: 46 tail tests and 12 automatic tests passed. Corrupt remote
readback retains local inputs; released payloads cannot be reused against a
different repository. Production targets remain 20 cases or 2 decimal GB of
packed state data, with whole-case overshoot; Parquet targets 512 MB uncompressed
columns and 262144 rows per row group.

## Entire-grid diagnostic completed

Source fbe9da30; the same pinned native build. No range was supplied.
All 527 format-domain pairs were classified: 147 attempted, 236 pruned by
fixed-r resource boundaries, and 144 structural exclusions. No pending pairs.
Independent CPU full-word oracles verified all 117 COMPLETE graphs and the
completed-layer prefixes of all 30 INCOMPLETE searches. HF readback verified
147 manifests, nine shared Parquet files (2,220,960 bytes), SHA256 and ledger.
Diagnostic capacity was deliberately 256 rows/rank and a 64 MiB pool; this
proves traversal/pruning/publication behavior, not maximum hardware range.
(7,1) failed capacity, (8,1) was skipped, while (8,6) completed.

Evidence: https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/evidence/20261003-full-grid-small-capacity/report.json

## Near-full VRAM and 32-batch graphs

Source 2eba332b, native sources unchanged from the pinned build. n15 r4,
48,128,101 admitted rows/rank, 34 completed prefix layers, 449,989,768 packed
bytes. Six alternating direct/graph samples matched every layer state hash
and count. Each graph sample executed 22 full 32-batch windows per rank.
Median direct 3.577442398 s; graph 3.480637629 s (2.706% less), selected graph32.
50 ms monitoring observed 16,509,829,120 / 16,752,181,248 bytes on each GPU
(98.553%). This measures allocated memory, not working-set throughput saturation.
Three matched archive-on/off pairs measured 3.500298535 / 1.261651098 s median,
ratio 2.774379: archival overhead remains substantial. No zero-overhead claim.
An independent CPU oracle subsequently matched all six GPU sample hashes for
layers 0..23, totaling 1,303,953 states. Depths 24..33 have cross-mode parity only.
Large-prefix automatic calibration now permits up to 600 seconds, still bounded
by 10% of remaining compute time; small whole-graph calibrations retain 90 seconds.

Evidence: https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/7827d4610cdfdcd4e22c32b9e07191ecb82d9d33/evidence/20261003-nearfull-fast-prefix/report.json
CPU oracle: https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/evidence/20261003-large-rank-prefix-oracle/report.json

The owned rental 54052996 was deleted by the watchdog; API absence confirmed.
Quoted three-hour compute/disk cost was approximately $0.58, excluding traffic;
actual billing was unavailable because the API credential lacks billing_read.
