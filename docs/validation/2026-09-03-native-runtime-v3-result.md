# Native DENSE: measured T4 search and memory improvement

Private Kaggle `trydotatwo/mgbfs-native-runtime-t4`, version 3, completed.
The independent raw-evidence verifier reports `VERIFIED_NATIVE_RUNTIME_GATE_10/10_AND_AB`.

Native source: `4fea92d2228c19be83ca2d16464daf5623aa21ba`.
Immutable CayleyPy baseline: `f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`.
Native generation remains variant 0 in this experiment. Later preflight,
tiny-plan launch sizing and optional generation-variant wiring are not part of
this measured source.

## Scope and verification

This is an assembled **single-GPU** DENSE BFS with GPU-resident states, bounded
bucket owner, parallel merge-path tiles, and producer/owner ping-pong. It is not
the complete multi-rank production runtime.

Both physical T4s passed independent plain/memcheck/racecheck/initcheck/synccheck
suites: 10/10, zero errors and zero race warnings. The suites cover full small
layers, state/slot reuse, padded matrices, nonidentity starts, capacity failure
and the mandatory archive. Plain tests additionally compare every full-state
layer U4(5..8), pre-dedup ON/OFF. The large plain tests are not relabeled as
large-workload sanitizer tests. Independent GPU suites do not test NCCL.

- Timed GPU0: `GPU-ace42b1a-f229-d84f-e76b-fdefe6fbcbb5`.
- GPU1 for independent validation: `GPU-e95285c9-313c-66d3-c748-bd577326b334`.
- CUDA 12.8.93, sm75, Rust 1.75.0, CUTLASS
  `ffa119a1255d78998536107466cc7097ecefa393`.
- Native/CayleyPy full-state SHA256 layer sequences agree at m8.
- Every successful large run agrees on all layer counts and m^6 cardinality.

## m16: five fresh-process measurements per backend

16,777,216 states, 25 nonempty layers. Both backends warm a complete same-workload
BFS before timing. Build, process startup and allocation are excluded. Native
search includes archive D2H submission; disk completion has its own timer.

Native batches 65536/262144/524280 and baseline batches
65536/262144/1048576 were tried independently. The fastest successful batch for
each backend was then measured in five fresh processes, alternating backend order.

| Backend | Selected batch | Search seconds median ± MAD | Full-device peak MiB |
|---|---:|---:|---:|
| Native DENSE, mandatory archive | 524280 | 1.13546 ± 0.00296 | 2713 |
| CayleyPy, no archive | 1048576 | 1.62068 ± 0.00025 | 6283 |

Native search is **1.43× faster**, with **56.8% lower** sampled device memory at
the respective fastest tested configurations. This is not a claim about every
possible configuration. For context, the smaller-batch calibration points are:

| Batch | Native seconds / MiB | CayleyPy seconds / MiB |
|---:|---:|---:|
| 65536 | 1.93467 / 1813 | 2.41526 / 1941 |
| 262144 | 1.25839 / 2195 | 1.83746 / 3563 |

Those calibration points are single trials, not five-run medians.

Native durable RunCommit median is **5.67342 s** from search start. The archive
cost is real: native uses **2,701,090,560 pinned host bytes** in this configuration.
The baseline creates no archive, so an end-to-end output-equivalent speedup is
not established by comparing these two contracts.

Other memory counters:

- Native requested CUDA allocations: 2,719,470,659 bytes.
- Native cudaMemGetInfo used peak: 2,844,655,616 bytes, including context.
- Baseline Torch allocated peak: 2,978,486,784 bytes.
- Baseline Torch reserved peak: 6,461,325,312 bytes.
- External sampling: nvidia-smi every 50 ms, whole-device usage including startup
  and warmup. The inherited JSON field `smi_process_peak_mib` is a device counter,
  not a process-only NVML counter.

## Larger graphs: one capacity probe each, not medians

The m16-selected batches were retained, without an additional per-graph sweep.
Layer capacity was fixed before each run at min(m^6, 32,000,000).

| Graph | States | Native search s | Native durable s | CayleyPy search s | Native / baseline device MiB |
|---|---:|---:|---:|---:|---:|
| m20 | 64,000,000 | 5.28627 | 21.56818 | 6.61256 | 4143 / 11435 |
| m24 | 191,102,976 | 19.60807 | 65.02221 | 22.75882 | 4149 / 11285 |

Native pinned RAM was 4,211,016,960 bytes at m20 and 8,271,041,280 bytes at m24.
This explicitly larger, preallocated backlog prevents the 512 MiB archive-ring
failure recorded in version 2. There is no in-run resize, disk wait in the GPU
producer, archive-disable option in the benchmark, or fallback.

m24 has 37 nonempty layers and a largest frontier of 17,784,471 states. Both
backends finish search in less than a minute. Native durability exceeds a minute;
it must not be presented as a minute-long BFS search. A separate m28 capacity
probe uses the same gated source with a new explicit layer capacity of 48,000,000.

## Reproduction and remaining boundary

Logs, all 22 raw worker rows and GPU test logs:
`test_results/native-runtime-v3/native-runtime/`. Full archives remain available
in the private Kaggle output; local downloads exclude the multi-GB archive files.

```
python scripts/verify_native_runtime.py test_results/native-runtime-v3/native-runtime --source 4fea92d2228c19be83ca2d16464daf5623aa21ba
```

The requested full architecture is still incomplete: native NCCL multi-rank BFS,
HASH_FIRST, BMMA owner, schema2 archives and the production CLI are not certified
or implemented by this single-rank benchmark. Do not integrate this result into
CayleyPy as a completed multi-GPU backend.
