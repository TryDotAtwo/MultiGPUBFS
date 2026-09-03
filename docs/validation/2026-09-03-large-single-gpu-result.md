# Large single-GPU comparison: capacity reached before tuned minute-scale BFS

Private Kaggle comparison version 3 completed. Native source:
`57e4f6f36d030c59c91720e5179cb85e120a2794`; immutable CayleyPy baseline:
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`.
This run predates the new generation variants and does not measure their effect.
Timing GPU: Tesla T4 `GPU-fd0cc756-bd0b-a1d0-b7e9-371e663e282a`.
Neither implementation creates an archive. Native uses u8 states and Hash128;
the baseline uses int64 states and its existing hash. This is experimental
single-GPU BFS, not the complete proposed native runtime/output contract.

## Protocol and completion

Graphs U4 m16, m20, m24 completed on both implementations. At m28 all three
independent batch configurations failed for each implementation; m32 was not run.
Native capacity was fixed before allocation at min(m^6, 32000000), with a 1 GiB
untouched VRAM requirement. No in-run fallback or capacity resize was used.

Largest common completed graph: m24, 191102976 states, 37 nonempty layers
(depth 0 through 36). All successful runs agree on every layer count and total;
full state-set digests were not collected for these large graphs. Small-graph
full-state validation is documented separately.

Of 28 worker rows, 22 completed and six recorded the m28 capacity failures.
All 22 successful raw worker logs were independently checked against summary
times and layer counts; all six failure logs were inspected.

For each process, a complete same-workload BFS warmup precedes the measured BFS.
Initialization/import/build and warmup are excluded from search seconds. Selected
batches were calibrated on m24, then measured in five fresh processes each,
alternating backend order. The external nvidia-smi sampler runs every 50 ms and
includes setup/warmup in its process-lifetime peak.

## Result

| Backend | Selected parent batch | Search seconds median +/- MAD | Allocation counter MiB | External process peak MiB |
|---|---:|---:|---:|---:|
| Native legacy generation | 524280 | 41.4255 +/- 0.0093 | 11292 fixed CUDA allocation delta | 11399 |
| CayleyPy single GPU | 1048576 | 23.2677 +/- 0.0135 | 7711.003 Torch allocated / 11164 reserved | 11285 |

Native takes 1.78x as long and has approximately 1% higher external sampled memory
peak on this graph. It does not establish a time or memory Pareto win.
Allocation counters have different meanings and must not be conflated with the
external process peaks.

Calibration (single trials, not five-run medians):

| Parent batch | Native seconds | CayleyPy seconds |
|---|---:|---:|
| 65536 | 238.78 | 66.20 |
| 262144 | 70.03 | 33.36 |
| 524280 | 41.40 | not tested |
| 1048576 | unsupported legacy grid | 23.24 |

Some configurations genuinely took minutes, but the tuned configurations did
not. Report status is CAPACITY_OR_SIZE_LIMIT_BEFORE_MINUTE_SCALE, not fulfillment
of the requested >=60-second tuned workload. Do not slow down the configuration
deliberately or aggregate repeated small searches to claim minute-scale load.

## First tested capacity boundary

- Native m28: all three batches stopped with OWNER_FATAL_1, the fixed owner
  capacity guard. This is a configured 32-million-row limit, not proof that the
  graph cannot fit any possible T4 implementation or allocation plan.
- CayleyPy m28: all three batches raised torch.OutOfMemoryError in
  `torch.vstack(layer2_batches)` while allocating approximately 3.76--3.98 GiB.
  These are failures of the tested configurations, not an exhaustive allocator
  tuning result.

The batch sensitivity motivates examining repeated owner/merge work in addition
to generator tiling. It does not by itself quantify which GPU stage dominates;
a timeline/profile is still required before making that attribution.

Evidence: `test_results/kaggle_single_gpu_benchmark_v3/single-gpu-bench/`.
Summary SHA-256:
`194dd1999a762e1cfd6f263f016559c399d74ac90f1dd50979caa10f143d9973`.
