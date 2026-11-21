# Post-allocation retry and memory-check evidence, 21 November 2026

## Scope

Two RTX 3060 (12 GiB each), PCIe NODE, driver 580.178.04, CUDA toolkit/Compute Sanitizer 12.8, source tree based on `d067734f3e9f012e3e394ccf712f123867c00e0c` with the post-allocation rollback/retry fix published in `98fb1fb940c1aa4f1405a3157b71e5f5f9fe2075`. CUDA library stayed identical between pre-fix and post-fix runs. The [fix report](2026-11-21-segment-postalloc.md) includes the controlled allocation fault and exact small-graph checks. None of these measurements is a B200 acceptance test.

## S13 pressure comparison

Graph is the adjacent-transposition Cayley graph of S13, identity start; batch hash, eight shards, two GPUs, no autotune. An *additional configured reserve* of 10,300,000,000 bytes per GPU limits admission; it is not a physical 10.3 GB competing allocation. No disk state archive was measured. Every completed layer size below was checked against the independent Mahonian count. Full state sets and cross-rank uniqueness were **not** checked on these large pressure runs.

| preset | version | status and completed depth | peak live rows/rank | comparable layer-time sum through depth 21 |
| --- | --- | --- | --- | --- |
| max_capacity | before fix | INCOMPLETE, RESOURCE_10, 21 | 40,984,370 / 40,964,209 | 519.447073 s |
| max_capacity | after fix | INCOMPLETE, RESOURCE_10, 22 | 52,687,114 / 52,678,233 | 520.625163 s |

The largest reported live value includes work attempted after the last completed layer; it is not a completed layer width. One run per configuration. The after-fix Python test parent hit its 900-second harness timeout, but the native workers continued and produced independent reports; their phase files, rank reports and shutdown/recovery method are recorded in the evidence. The 1,149.935285 s reported BFS time includes an incomplete layer and is **not** the time for the complete graph. The common-prefix comparison is a sum of completed-layer timing fields and is not an end-to-end wall-time comparison.

Balanced hash stopped at depth 21 with RESOURCE_10 in both versions. Elapsed incomplete-run measurements were 21.825158 s before and 21.947523 s after; those figures include unfinished work and should not be compared as a complete graph.

For bitmap batch, the same additional reserve and an intentional profile stop at depth 24 produced:

| preset | completed depth and stop | reported BFS time | 1 s sampled peak GPU memory per rank | peak requested segment-storage bytes/rank |
| --- | --- | --- | --- | --- |
| balanced | 24, PROFILE_LAYER_LIMIT | 39.360553 s | 2001 / 2001 MiB | 1,308,623,616 |
| max_capacity | 24, PROFILE_LAYER_LIMIT | 404.580935 s | 1503 / 1503 MiB | 872,416,000 |

The bitmap domain itself additionally used 389,188,800 bytes/rank. These presets selected different bank/lane/queue geometries: balanced 3 banks, 4 lanes, queue 24,576; max_capacity 2 banks, 1 lane, queue 2,048. Thus these are **not** equal-geometry timings. Both were deliberately stopped at depth 24, not at memory exhaustion, so this experiment does not establish the maximum capacity of either preset. NVML peaks are sampled and can miss transients; tracked storage bytes are not whole-process VRAM.

## Compute Sanitizer

All six backend/strategy combinations ran S3 on both physical GPUs and matched the exact CPU oracle. The unfiltered Compute Sanitizer run exited 99 and reported 360 CUDA API errors. The preserved unlimited log attributes the reported API errors to `ncclInitKernelsForDevice` in NCCL 2.30.7: missing kernel image and invalid resource handle returned during CUDA function-attribute probing, plus `cudaGetLastError` calls. NCCL's [kernel initialization code](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/enqueue.cc#L30-L50) and [error checks](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/include/checks.h#L41-L47) show the handling of failed probes; this attribution does not make the unfiltered run clean.

A separate run of the same six checks with `--report-api-errors no` exited 0 with `ERROR SUMMARY: 0`: no memory errors were reported under that configuration. It suppresses **all** CUDA API error reporting, and does not prove race freedom or provide B200 acceptance.

## Evidence

* [Pressure metadata and logs](2026-11-21-pressure-postalloc-evidence.json.gz): SHA-256 `4b50ca0e6800a4eb06e415e24daab16ad426dc7c93fd5c4a9fc1e734c3ed6f4c`.
* [Full and API-filtered sanitizer logs and receipts](2026-11-21-sanitizer-evidence.json.gz): SHA-256 `b9bea00a70abd5cddc9b4e24118e10785c3733f0bbfb51b38a93c4f68b5a8482`.

These archives contain telemetry, reports and scripts, not enumerated graph states.
