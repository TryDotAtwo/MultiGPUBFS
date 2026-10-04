# Resident pair dispatch, two RTX 3060, 2026-10-04

Own branch `codex/bfs-tail-archive`; final tested source
`82256488e` (full commit in adjacent JSON), native binary SHA-256
`1eab1758457d09d3d96d08e0fd5c0be8bdcf04da6c6518fbadb6e36aedcc3081`.
Driver595.71.05, CUDA12.9, NCCL2.30.7, CUCO_RANK, two ranks,
HOST_SIZED_NCCL, Graph batches0. Same native binary across Python-only fixes.

Automatic production sweeps now keep one torchrun group alive across healthy
pairs and memory probes. Compatible geometry reuses zeroed device allocations,
RMM pool and drained pinned slots. HOST NCCL is parked only after unanimous
rank agreement, with cancellation/retirement contexts cleared. Any native
failure ends the rank group; the next eligible pair starts a fresh generation.
Changed geometry discards incompatible device allocations. Cached admission
requires matching configuration and a covered orbit bound; small-orbit results
cannot be treated as the maximum hardware capacity. GPU inventory, binary hash
and commit metadata and the independent50ms VRAM monitor are session-scoped.
No per-pair compilation and no new per-batch GPU synchronization.

## Measured results

Runner wall includes admission, native execution and local archive completion;
native search is the maximum of rank search_complete_seconds. Wall-minus-search
is not a pure dispatch timer. HF publication and post-run oracle checks are
outside these timings.

| Case/metric | Standalone | Resident warm |
|---|---:|---:|
| `(8,2)` runner wall, seconds | 2.846754 | 0.352437 |
| `(8,2)` native search, seconds | 0.217662 | 0.199393 |
| `(12,4)` runner wall, seconds | 15.881982 median of3 | 5.017777 / 5.036313 |
| `(12,4)` native search, seconds | 2.586754 median of3 | 2.569364 / 2.563431 |
| `(12,4)` wall minus search, seconds | 13.292934 median of3 | 2.448414 / 2.472882 |
| `(17,r)` warm admission | repeated probing | 0.000086–0.000096 seconds |

Cold admission of `(17,1)` fell from the prior standalone76.24-second gate to
2.044767 seconds. Cold process/context startup still happens once. Repeated
`(12,4)` used33 cached buffers, one NCCL communicator and one RMM pool per rank.

An additional clean run measured **end of the previous final native layer to
beginning of the next depth0**, with no fingerprint work between jobs:
**2.469484 and2.449897 seconds**. Archive processing remains substantial for the
19,958,400-state,55-layer graph. A proposed50ms polling explanation was tested
and did not explain that residual; the drain fix passed but did not materially
change the measurement. Profiling places the dominant Python cost in snapshot
file operations/fsync; this is diagnostic evidence, not an established lower
bound. No claim of zero pair overhead or maximum achievable throughput.

## Correctness and delivery

Eight fixed cases `(8,1)`, `(8,2)`, `(8,3)`, `(17,16)`, `(33,32)`, `(65,64)`,
`(128,127)` and a12-layer prefix of `(128,1)` matched both standalone packed
state fingerprints and independent CPU layer/state enumeration. The128-prefix
is intentionally INCOMPLETE. The latest native code was unchanged in these
Python-only updates; fresh `(8,1)/(8,2)` and full `(12,4)` gates were rerun.
All nine full `(12,4)` runs across standalone and resident cohorts matched full
state fingerprints; this does not assert independent CPU enumeration of19.9M
states. Three automatic `(17,r)` prefixes validated admission reuse. A real
64-row failure restarted the rank group, skipped the next larger fixed-r1 case,
and still completed `(16,15)` and `(17,16)` with CPU oracle agreement.

Parquet payloads, manifests and sweep ledgers were uploaded from the GPU host
and read back at pinned HF revisions with every checksum verified. The latest
three full runs were published as three123,024,404-byte Parquet files; repeat
runs are separate cohort groups because a cohort rejects duplicate graph keys.
Only metadata was retrieved to Ivan's computer. Intermediate harness failures
and profiling logs are also preserved on HF.

[Immutable delivery report](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/73db62098c208dcf26cf6f8c046cd862b6e5fe15/evidence/20261004-resident/delivery.json)
contains payload revisions. Adjacent JSON also retains the earlier eight-case,
automatic admission and failure-recovery gates.

Local tests passed: resident controller1, automatic admission16, VRAM admission8,
sweep13, native failure5, archive5, wire4; Rust protocol and bootstrap tests also
passed before the final Python-only changes. Linux Rust1.75/CUDA build and real
multi-rank execution passed on the rented machine.

The table's archive contract remains intact: intermediate snapshots after every
completed layer, whole-layer COMPLETE retention, bounded INCOMPLETE suffix,
per-layer counts/time and sampled GPU peaks, checksums/configuration/status.
Cold starts, geometry changes and archive I/O still cost time. LSA communicator
reuse and B300/Graph32 performance have not been validated by this gate. Actual
coverage is the cases listed above, not a full traversal of all8255 pairs.

Owned Vast instance54131688 was removed after terminal jobs and token removal;
provider GET confirmed ABSENT. No other agent rental or checkout was changed.
