# Four upload policies and full automatic GPU-host run

The isolated `codex/bfs-tail-archive` branch implements independent upload timing
without changing the retained data contract. `search`, `graph`, `background` and
`end` all preserve the agreed manifests, complete-layer statistics, provenance
and bounded packed-state tails in Parquet. States stay on the GPU host or HF.
The default is `end`. No per-pair 120/300-second timeout remains; an externally
owned compute/publication/lease deadline still limits paid runs.

## Actual hardware and sources

Two RTX 3060 12GB, SM86, driver 595.71.05, CUDA 12.9, NCCL 2.30.7,
HOST_SIZED_NCCL, CUDA Graph batches 0. P2P is unavailable on this rental.
This evidence does not admit Graph32, NCCL_LSA, B300 or more than two ranks.

The 93-pair full sweep used `d2356c270dbef73d75e9e9d75957b91d1424ae52`.
The large archive-credit regression used `b9bc075fbe9e0a7ea829b5d5eab570949b477ec0`.
The final explicit native credit policy and all-mode resume gate used
`768c1056ad3dc6d44e41b6c58a22dfe21ca1f74f`, binary SHA-256
`9446c3eb608f9ab1934300680b9ae40e616dc2d8ba8e9acffd64feab7d443303`.
The companion JSON records pins, timings, immutable HF revisions and receipts.

## Storage admission and resumption

At case boundaries, every mode reserves SSD space for the next working tail,
native spools and Parquet conversion. At a storage pause or final stop, the
last short cohort is sealed; `cohort_seal_after` persists its boundary so future
cases cannot extend a released group. Publication uploads payloads before
manifests, streams checksum readback and writes a release receipt before unlink.
After verified release, the same resident rank group resumes pending pairs in
the selected upload mode. Upload/readback failure retains inputs.

The actual two-GPU gate forced a storage-admission pause after the first case
in **each** of the four modes, then performed real HF publication, readback,
release and continuation. All modes completed `(4,1)`, `(4,2)` and `(4,3)` over
two compute/upload cycles using one rank-group PID per mode. All downloaded
Parquet states and depths matched independent CPU BFS on the GPU server.
End mode used a 20-case cohort target, proving a short group can be sealed and
released before reaching the normal target. Search-mode temporary HF previews
were removed only after canonical archives were verified. This is injected
admission pressure, not a claim that the physical SSD was filled to ENOSPC.

## Bounded archive-credit policy

The large sweep exposed temporary pinned-slot exhaustion as
`ARCHIVE_PIN_RING_FATAL`. It was an archive credit failure, not GPU OOM, and did
not prune the fixed-r branch. The durable-tail launcher now explicitly selects
`MGBFS_ARCHIVE_CREDIT_MODE=wait`. Bare native runtime retains fatal exhaustion
unless waiting is selected. An explicit `fatal` setting also overrides the
durable-tail default. The selected environment is saved in launch metadata.

The ready-credit path remains immediate. Only empty credit queues wait for the
SSD worker, with 20 ms cancellation checks and native rank-failure cancellation.
Slots return after the recorded D2H event and disk write; ownership and bounded
allocation are unchanged. This does not add device batch synchronization, but
an SSD bottleneck can now stall the archive producer instead of aborting BFS.
It is not evidence of zero end-to-end archival overhead.

Five actual CUDA worker tests passed, covering immediate fatal exhaustion,
waiting for a paused writer without losing live readers, successful drain,
write failure and final-sync failure. Three CPU credit tests cover ready,
cancellable wait and disconnect paths. A first version of the fatal-policy test
omitted its empty layer commit and failed with ARCHIVE_UNCOMMITTED_LAYER; the
fixture was corrected and all five CUDA tests passed on the final source.

Additional actual two-GPU runs completed `(8,1)` and `(12,4)` with only two
archive slots. The former's full 40,320 states over 29 layers were independently
checked against CPU BFS from HF Parquet on the GPU host. With the normal bounded
queue, `(13,3)` completed 1,037,836,800 states over 72 layers, retained the whole
8,302,694,400-byte packed graph, and passed HF checksum/manifest/ledger readback.
Its native search took 118.894 s. The old full-sweep INCOMPLETE attempt remains
immutable; the corrected regression has a separate run ID and receipt.

## Full-run timing and coverage

| Measurement | Seconds |
| --- | ---: |
| Whole program | 2137.783 |
| Pair sweep including archive drain and admission | 1022.241 |
| Parquet packing, publication, release verification | 972.078 |
| Final HF readback | 139.055 |
| Native search from complete rank reports, 87 pairs | 146.698 |
| Known search including completed layers of failed attempts, lower bound | 454.349 |
| Admission, 93 pairs | 13.925 |
| Native setup from complete reports, 87 pairs | 1.092 |

The actual previous search-end to next search-start gap had median 0.126 s,
sum 561.721 s and maximum 136.519 s over 92 transitions. It includes archive
drain, cleanup, admission and cancelled/failed attempts; it is not just dispatch
or compilation. The small ledger transition metric alone was 1.632 s and must
not be substituted for those gaps. Component observations overlap and are not
additive. Unreported work in failed layers is unknown, so the search lower bound
must not be promoted to an exact full-program BFS total. Build is reused across
pairs; the final incremental native build took 11.131 s once, not per graph.

The grid is automatically constructed for all 8255 pairs in `2 <= n <= 128`.
93 pairs were attempted: 87 COMPLETE and 6 INCOMPLETE. Typed GPU resource evidence
pruned 458 larger-n pairs independently at fixed r; 7704 remained pending when
the run was explicitly cancelled to repair the archive-credit failure. This
run does not claim graph exhaustion across the whole grid. All 87 COMPLETE
layer-count sums equal n!/r!; no archive-complete/native-exhaustion mismatch was
observed. The 90 Parquet payloads total 13,896,774,792 bytes, with 93 manifests
and the sweep ledger verified from HF. VRAM sampling remains 50 ms per GPU;
unsampled short layers retain null observations rather than invented peaks.

## Checks and rental cleanup

76 local tail tests, 9 upload tests and 8 VRAM-planning tests passed. The final
source passed the actual GPU worker and all-mode CPU-oracle/resume gates above.
Instance **54135219**, machine **137174**, was deleted after every own test job
became terminal and the remote HF token was removed. Vast GET confirmed ABSENT.
Observed lifetime and RX+TX give an approximately $0.52 quote-based estimate
for this rental including its 100GB disk and traffic. Actual billing is not
verified because the API key lacks billing_read. The $2 test cap was maintained
by the accepted quote, two-hour watchdog and traffic allowance.

HF evidence lives in dataset `TryDotAtwo/multigpubfs-bfs-results` under
`evidence/20261004-full-v3`, `-v4` and `-v6`. Final delivery adds the combined
summary and collected metadata under `evidence/20261004-full-final`.
