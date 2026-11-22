# dev44: advisory profile cache and resource-limited tuning

This experimental development package fixes three Python-side contracts; the native
CUDA bundle is unchanged from dev41/dev43. This report does not claim B200, a
completed S14, or a physically full 2^32-row segment.

## Artifact identity

- Published wheel:
  [multigpubfs-0.2.0.dev44-py3-none-any.whl](https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/b2ed1712e6c79f7402b45f961355899cb2e80c3f/wheels/multigpubfs-0.2.0.dev44-py3-none-any.whl).
  SHA-256 `29b1ffdf6d31b194184e0d377e83e22b0f8b8c7b1355b36ff7b09eda5bdfd10b`.
- Exact Python source commit `5ed4b90e6ab3ff8ab16c78e36722e9c41fb5cd70`;
  source tree `c17ddbb8849870e0a37701c947fa5ee0f8446bd7`.
  Wheel Python files were compared with this tree. The wheel's embedded
  `_package_info.json` records the pre-build *base* `cd0b669c4e77bf76e8ab5d4991d7f8dfa8445ee2`;
  it is not the complete dev44 source identity. Use the source commit and wheel
  hash above together for provenance.
- SM86 native executable SHA-256
  `10cb2e53915b9e8936582889445eed263cd6c370ab877aae0ee628fd8f2da687`;
  CUDA library SHA-256
  `ca4fba4419120aaa4a37123e56faedbb91c8e4b58e263481e9a15b517c43d908`.
  Native asset index `native-pip-20261121-v41`.

## What changed and what was tested

The profile cache is advisory: invalid JSON, non-object JSON and read failures
produce a miss. A failed cache write returns the measured profile with its
stage/errno recorded. Non-cache errors are not converted to success. A
rank-consistent, verified `RESOURCE_10` tuning prefix can select a clearly marked
unmeasured admissible profile for production; a partial layer is not published.
The selector compares per-rank capacity within the same scope and hard cap,
then measured speed. A planned capacity is an estimate, not a physically
demonstrated maximum.

The isolated dev44 Python suite reported 1,149 passed and 259 subtests passed.
[GitHub Actions](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/44974671911)
verified the public wheel SHA, pip installation, GPU-free prefetch and offline
cache reuse. The repository-wide native CPU workflow on the dev44 wheel commit
[failed at Rust formatting](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/44973534674);
the same formatting step also failed on the pre-existing base commit. Earlier
CPU contract/type-check steps in that run passed; the workflow as a whole is
not green.
Stage-specific cache fault injection, malformed input, inconsistent rank
receipts, CUDA/NCCL errors and divergent prefixes were included. A cache
unavailable to an installed public API job did not prevent the measured
profile's single production run; the writable-cache path was tested with a
subsequent cache hit. Cache behavior is not a retry of BFS.

Installed-wheel tests used 2 and 8 physical RTX 3060 GPUs on one host. For
`segmented24` and `permutation_bitmap`, both `balanced` and `max_capacity`,
a deliberately tight per-rank cap caused all eligible tuning probes to reach
the resource path and production to return `INCOMPLETE`. Every retained state
in every completed layer of those small cases was checked as a multiset against
an independent CPU BFS; a healthy next job completed on the same persistent
worker processes. The wheel installed from the public SHA-pinned URL was
independently run on eight GPUs for all four resource/preset combinations.
An installed two-GPU `hash → bitmap → hash` sequence retuned after an
approximately 96 MB change in free memory; the capacity scores are plans and
not a pressure proof that every scored row fits.

Tuning compares complete retained snapshots exactly. Where the native receipt
contains only bounded samples for larger layers, their validation does not
establish equality of every state. Probe divergence, mismatched rank receipts,
or CUDA/NCCL errors must not become a healthy unmeasured fallback. The
[previous dev43 report](2026-11-22-dev43-8gpu.md) documents the earlier exact
small-layer matrix and the limits of the earlier S13 prefix.

## S12 comparison and queue geometry

One eight-GPU host, eight 12-GiB RTX 3060 (SM86), driver 570.153.02,
PCIe across two CPU NUMA nodes; one graph: all adjacent transpositions of S12,
479,001,600 states in 67 layers. Each run checked all layer sizes against an
independent Mahonian count, and the full states of the last two layers. Middle
layers were checked by counts, not full state multisets.

The six-mode dev43 baseline used identical topology, 8 shards, exact exchange,
balanced and effective queue 5632, four owner lanes, no autotuning. Three
repeats, BFS seconds:

| Backend | Strategy | Repeats, seconds |
|---|---|---|
| permutation_bitmap | atomic | 38.839, 38.822, 38.763 |
| permutation_bitmap | warp | 34.346, 34.453, 34.725 |
| permutation_bitmap | batch | 76.486, 77.962, 77.488 |
| segmented24 | atomic | 27.079, 27.295, 27.212 |
| segmented24 | warp | 14.427, 14.513, 14.431 |
| segmented24 | batch | 27.103, 27.647, 27.525 |

The following factorial experiment used `permutation_bitmap/atomic`,
the same native binary and CUDA library, and a pre-final dev44 Python snapshot
with tuning disabled. The later changes were in advisory tuning and cache
handling. Requested queue 8192 was admitted as **5632**. Times are seconds:

| Actual queue | Owner lanes | BFS: three runs | Sum of layer times, depths 27–38 |
|---:|---:|---|---|
| 2048 | 1 | 114.216, 113.291, 113.276 | 64.894, 64.493, 64.587 |
| 2048 | 4 | 99.044, 97.900, 98.015 | 56.255, 55.542, 55.683 |
| 5632 | 1 | 50.815, 50.656, 50.598 | 27.920, 27.805, 27.781 |
| 5632 | 4 | 38.892, 38.678, 38.611 | 21.230, 21.147, 21.108 |

Each configuration had 8 shards and a requested initial batch of 8192;
the effective queue-batch ceilings were 512 and 2048, respectively. The
27–38 window contains 277,885,890 newly accepted states and 3,013,839,301
generated edges. Layer sums are computed from reported per-layer critical
rank times; they are not an independently sampled wall interval. Rank-zero
prefetch counts correlated with the queues (112,309 versus 32,646) but do not
prove a single bottleneck. The [machine-readable receipts](artifacts/2026-11-22-dev44-s12-factorial.json)
contain the effective geometry and per-rank memory fields; the
[executed harness](artifacts/2026-11-22-dev44-s12-factorial.py) requires its
documented remote paths to be adapted.

Sampled NVML memory at one-second intervals is not an allocation high-water
proof; short peaks may be missed. Timings are specific to this topology and
graph. The older strategy comparison and this bitmap factor test have different
source snapshots; only the specified native hashes are identical. The full
pre-final Python tree for the factorial test was not captured, so its exact
Python snapshot cannot be reconstructed from these receipts alone. No default
was changed from these measurements, and no B200 inference follows.

## Batch recovery: isolated S12 A/B

Using the publicly installed dev44 wheel on the same eight GPUs, fixed
`permutation_bitmap/atomic`, 8 shards, 4 owner lanes and effective queue
5632, we ran recovery off/on/on/off/off/on. Every run completed S12 with all
Mahonian layer counts and exact full state sets for the last two layers.

| Recovery | BFS, seconds | Depths 27–38, seconds |
|---|---|---|
| off | 38.980, 38.658, 38.700 | 21.355, 21.146, 21.155 |
| on | 38.585, 38.750, 38.702 | 21.130, 21.230, 21.131 |

Every rank reported two batch recoveries when enabled and none when disabled;
its final batch ceiling was 8192 versus 2048. Rank-zero prefetch chunks
were 32,646 in all six cases. Differences are within this small sample's
variation on this graph and should not drive a default change. The first
run's wall time included cold setup (72.542 s versus 38.980 s BFS); later
wall times were approximately 38.86–38.97 s. These are not isolated
measurements of every setup component.
[Per-rank receipts](artifacts/2026-11-22-dev44-recovery-ab-s12-8gpu.json)
and the [harness](artifacts/2026-11-22-dev44-recovery-ab-s12-8gpu.py)
record the inputs, device list and environment-specific paths.

## S13: same completed prefix on the installed public wheel

The SHA-pinned public dev44 wheel ran both configurations sequentially on
the same 8×RTX 3060 host with exact exchange, 8 shards, 4 owner lanes,
requested queue 8192, effective queue **6144**, and tuning disabled. Both
stopped intentionally at the profile layer limit, with
`INCOMPLETE / PROFILE_LAYER_LIMIT`, not an exhaustion or full S13 result.
All sizes of completed layers 0–38 match the independent Mahonian oracle:
2,965,188,705 states. Full large-layer state sets were **not** compared.

| Backend / strategy | BFS prefix seconds | Depths 34–38, seconds | Max sampled memory per GPU |
|---|---:|---:|---:|
| permutation_bitmap / atomic | 219.833 | 102.774 | 2021 MiB |
| segmented24 / warp | 93.026 | 46.908 | 5447 MiB |

Each row is one observation; layer-window time is the sum of reported
per-layer times, not an independent wall-clock interval. It covers
1,378,000,276 new states and 15,780,016,836 generated edges. The effective
queue-batch ceilings differed (bitmap 2048, hash 8192); these timings do not
isolate backend cost. The one-second NVML samples are not true peak-allocation
measurements. The received `peak_live_rows` is per-rank high-water; its maximum
was 110,344,906 bitmap and 107,792,853 hash, and these maxima need not be
simultaneous. The [per-rank receipts and topology](artifacts/2026-11-22-dev44-s13-prefix-8gpu.json)
and [executed harness](artifacts/2026-11-22-dev44-s13-prefix-8gpu.py) are
included; the harness uses environment-specific remote paths.

## S13: complete run on the installed public wheel

A later, sequential full run on the same eight-RTX-3060 node completed in
both configurations using the SHA-pinned installed dev44 wheel, its public
`run_graph` entry point and the same persistent worker processes. The graph
was S13 with all adjacent transpositions, `autotune=False`, balanced, exact
exchange, 8 shards and 4 owner lanes. Both runs returned
`COMPLETE / FRONTIER_EXHAUSTED`, depth 78, with **6,227,020,800** states.
All 79 layer sizes match the independent Mahonian recurrence; the full
state sets of the final two layers were compared against their independently
constructed expected sets. Full state sets of the other large layers were
**not** compared.

| Backend / strategy | BFS seconds | End-to-end call seconds | Depths 34–44, seconds | Max sampled per GPU |
|---|---:|---:|---:|---:|
| segmented24 / warp | 218.854 | 236.709 | 108.522 | 5447 MiB |
| permutation_bitmap / atomic | 500.511 | 500.914 | 234.518 | 2021 MiB |

The depth 34–44 window has 3,052,643,942 new states and 36,412,333,584
generated edges. Window seconds are sums of reported layer times, not separate
wall intervals. There is one observation per configuration, in the listed
order. Both requested queue 8192 and admitted 6144; actual queue-batch
ceilings were 8192 for hash and 2048 for bitmap, so the result does not
isolate the costs of the two data structures. The configured segment-row cap
was 2^32; the observed allocation geometry used 1,048,576 rows per bank,
40 segments per rank and 3 banks. This is not physical validation of a single
2^32-row segment. Maximum per-rank reported live rows were 110,750,360 for
hash and 111,623,259 for bitmap; rank maxima need not be simultaneous.

The NVML values are the maximum of one-second samples, not a guaranteed
instantaneous VRAM peak. Reported segment-storage high-water bytes per rank
were 5,326,769,152 for hash and 1,635,779,520 for bitmap, excluding the
queue, scratch, CUDA and NCCL internals; bitmap additionally reported a
97,297,200-byte visited plane per rank. These values must not be summed with
NVML samples as if they were independent peaks.

[Full per-rank receipts](artifacts/2026-11-22-dev44-s13-full-8gpu.json),
[executed harness](artifacts/2026-11-22-dev44-s13-full-8gpu.py),
[run log](artifacts/2026-11-22-dev44-s13-full-8gpu.log),
[hardware](artifacts/2026-11-22-dev44-s13-full-hardware.txt),
[topology](artifacts/2026-11-22-dev44-s13-full-topology.txt), and terminal
states for [hash](artifacts/2026-11-22-dev44-s13-full-hash-terminal.json)
and [bitmap](artifacts/2026-11-22-dev44-s13-full-bitmap-terminal.json) are
provided. The complete evidence archive has SHA-256
`bf441de4e10fb581cff72196855c31c2758ca85e15c294afb54a6652398b1978`.
Paths in the harness are specific to the original environment. The observed
results do not establish S14/S15 capacity or performance on B200.

## Open coverage

Production pressure testing of the absolute largest graph, full-state
comparison of every large S12/S13 layer, packed frontier storage, physical
SM100/B200 execution and multi-host acceptance remain open. Actual memory
depends on rank skew, simultaneous banks, queues, allocator/NCCL reserves
and archive settings. Estimates are not a guarantee that S14 or S15 fits.
