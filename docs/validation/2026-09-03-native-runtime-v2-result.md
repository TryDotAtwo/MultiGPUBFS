# Archived native DENSE reference: T4 gate and first A/B

Kaggle `trydotatwo/mgbfs-native-runtime-t4`, version 2, completed.
Native source: `a37050c4565de200697654a478d4e4ca6b63ec9b`.
CayleyPy baseline: `f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`.
This native source precedes parallel owner tiles and the producer/owner ping-pong.

Both physical T4s passed the full small-layer/StateRing/archive fixtures under
plain execution and all four Compute Sanitizer tools: 10/10, zero errors and
zero race warnings. Plain runs also verified every full-state layer U4(5..8),
pre-dedup ON/OFF. Independent GPU suites are **not** a multi-rank/NCCL run.

- GPU0 / timed device: `GPU-cf1673bd-d5e5-43e9-c132-343fda928763`.
- GPU1 / independent tests: `GPU-bbd91757-6fec-3b32-1b8c-555eab3d8c9d`.
- Rust 1.75.0; sm75; CUTLASS `ffa119a1255d78998536107466cc7097ecefa393`.
- Native and CayleyPy full-state SHA256 layer digests agree at m8.

## m16 measurements

16,777,216 states; 25 nonempty layers. Five fresh processes per backend,
alternating order; full same-workload warmup excluded. Both selected batch
262144 from the two tested choices 65536/262144. This is not the final baseline
sweep: its 1048576 batch is included in the next run.

| Backend | Search seconds median ± MAD | Full-device sampled peak MiB | Output |
|---|---:|---:|---|
| Native bounded DENSE | 2.48068 ± 0.00109 | 2109 | Mandatory state/hash archive |
| CayleyPy single GPU | 1.85318 ± 0.00674 | 3563 | No archive |

Native search is 1.339× slower; sampled device memory is 40.8% lower. This is a
memory tradeoff, not a speed win. Native durable RunCommit median is **5.84002 s**
from search start: the remaining disk-worker work is not hidden in search time.

Native requested allocations: 2,094,321,563 bytes; cudaMemGetInfo observed
2,211,315,712 used bytes, including context; pinned archive: 536,870,912 bytes;
physical disk extent: 603,979,776 bytes. Baseline Torch peak allocated:
1,333,602,304 bytes; peak reserved: 3,609,198,592 bytes. Those allocation counters
have different semantics and are not substituted for the full-device sampler.
The inherited sampler field `smi_process_peak_mib` actually comes from the
`nvidia-smi --query-gpu=memory.used` full-device counter and covers warmup/setup too.

## Capacity failure is retained

At m20, native stopped with `ARCHIVE_PIN_RING_FATAL`: its fixed 512 MiB staging
ring filled. It did not wait for disk, resize, or claim a RunCommit. This is a
**host archive-capacity failure**, not VRAM OOM or wrong BFS layers. CayleyPy m20
completed in 8.210 s at 5931 MiB sampled device memory. m24 was not attempted.

The next experiment provisions an explicitly larger pinned backlog before
depth 0, includes the GPU optimizations, and extends both batch sweeps. That is
a new configuration, not a fallback inside the failed run. More pinned RAM is
a real resource cost and must remain visible in its report.

Evidence: `test_results/native-runtime-v2/native-runtime/`. State archive files
remain in the private Kaggle outputs; the local download contains logs/JSON/CSV.
Independent audit:

```
python scripts/verify_native_runtime.py test_results/native-runtime-v2/native-runtime --source a37050c4565de200697654a478d4e4ca6b63ec9b
VERIFIED_NATIVE_RUNTIME_GATE_10/10_AND_AB
```

The full requested architecture remains incomplete: this report does not certify
native multi-rank execution, HASH_FIRST, BMMA, schema2 archival or the production CLI.
