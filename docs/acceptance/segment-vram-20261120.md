# Segmented VRAM admission and bitmap two-bank checks, 2026-11-20

These results concern the specified tests on one machine with 2 RTX 3060 12 GB,
driver 580.159.03, CUDA 12.8.1 and NCCL with P2P disabled. They are not B200,
8-GPU or greater-than-2^32-live-rows-per-rank acceptance. The tested source was
based on commit `90e9c77d5ced7e25874af50972fb54fd5b15e238`; the tested CUDA
library SHA-256 was `61b1446b58fd237470d13e317a44cc9b4cf2fbaecf9d7cc1678cb7bca6f008cb`.
The main benchmark executable SHA-256 was
`81ef62c6c2f623374f265b5a8085f9549468497f4bfe8c504b6a64ea05b0c32c`.
After adding the compatibility advertisement, the API retest used executable
`aa9051c5aeb02486b8537b9e661c256fd4771d93ae38a354bbd5d50387eb2852`.

## Fixed-budget and API checks

* `tests/segmented_variable_budget_gpu.py` checks exact S5 layers with an
  independent oracle and exact previous/final samples under deliberately
  constrained segment budgets. Variable-size tail segments and a per-segment
  limit smaller than the aggregate capacity were exercised.
* `tests/segmented_capacity_comparison_gpu.py` compares equal synthetic
  storage-byte limits. Hash completed depth 2 under balanced and 3 under
  max-capacity at 1526 bytes/rank; bitmap completed depth 1 and 3 respectively
  at 294 bytes/rank. These intentionally tiny caps demonstrate allocation
  mechanics; they do not estimate B200 capacity.
* `tests/segmented_memory_presets_gpu.py` checks both backends and both
  presets with CayleyPy's persistent API: complete S5 layers and samples,
  forced INCOMPLETE, retained samples, the same worker processes, and a
  succeeding graph. The ABI-3 executable was used in the final repeat.
* `tests/segmented_preset_autotune_gpu.py` checks full S9 on two GPUs for all
  four combinations, measured pilots, oracle layers and profile-cache reuse.
  Cache hits took approximately 17–18 ms in that test. This is not a general
  guarantee that a new graph never needs retuning.
* CUDA leaf tests checked both slot-position widths, collisions and retirement.
  CUDA compute-sanitizer memcheck found no memory errors in those leaf tests
  and a separate bitmap-ranking leaf test on this machine. The sanitizer
  test explicitly skipped the 4-GiB boundary allocation; an ordinary GPU run
  exercised that allocation. This does not prove every inter-GPU race absent.
* The focused Python/API suite passed. A full Rust workspace attempt with CUDA
  EAGER had two LSA tests fail because this build has LSA compiled off
  (`LSA_PREPARE_GROUP: CUDA_STATUS_7`). Its separate LSA-dependent CLI case
  was also skipped. These are not LSA hardware acceptance.
* The separate legacy distributed-archive integration suite had 8 passes and
  6 failures on this setup. One was the disabled LSA path, two were
  `HASH_FIRST_TC_DEVICE_UNSUPPORTED`, two were `NATIVE_PLAN_CREATE_FAILED
  status=3` (cause not established here), and the existing forced-capacity
  group-fatal test returned `EPOCH_CREDIT_TIMEOUT` for one rank instead of
  the expected group-fatal status. This suite does not exercise the new
  segmented or bitmap kernels, but the failure prevents a claim that all
  legacy paths passed. Its relation to the baseline is not established.

## S11 bitmap bank comparison

With the same 2 GPUs, 4 shards, 4 lanes, batch, queue and exact exchange,
all 39,916,800 states and 56 layers matched the independent Mahonian oracle.
Runs were interleaved:

| Banks | BFS seconds | State-bank allocation per rank |
| ---: | --- | ---: |
| 3 | 11.0183, 11.0086 | 69,206,208 bytes |
| 2 | 10.8884, 10.9137 | 46,137,536 bytes |

The state-bank allocation fell by 33.3%; total VRAM also includes bitmap,
queues, communication, context and other allocations. The ~1% timing
difference is too small to establish a speedup from two runs each. For an older S13 comparison on 8x RTX 3090 (three-bank bitmap), see
[bitmap audit](bitmap-audit-20261120.md): 393.32 s hash versus 593.55 s
bitmap, each one run. Bitmap reduced its batch ceiling from 8192 to 2048
after retries, while hash retained 8192; this is not an isolated backend cost.

## S14-S16 capacity illustration

For ordinary Coxeter graphs with distinct symbols, one-byte symbols, 8 ideal
equally loaded GPUs, no allocator fragmentation and two bank capacities
sized to the largest layer:

| n | domain | maximum layer | visited bitmap GiB/GPU | two state banks GiB/GPU |
| ---: | ---: | ---: | ---: | ---: |
| 14 | 87,178,291,200 | 3,727,542,188 | 1.27 | 12.15 |
| 15 | 1,307,674,368,000 | 50,626,553,988 | 19.03 | 176.81 |
| 16 | 20,922,789,888,000 | 738,680,521,142 | 304.47 | 2751.80 |

These are storage estimates, not measured fits. They exclude buffers, NCCL,
reserved memory, allocator fragmentation and unequal ownership. Different
generators change layer populations. S14 appears plausible on 8 B200 for
this exact graph but has not been run. Even this optimistic two-bank S15
estimate exceeds 192 GiB/GPU; for S16 the bitmap alone exceeds it. Other
representations or out-of-core methods require separate implementation and
measurement.
