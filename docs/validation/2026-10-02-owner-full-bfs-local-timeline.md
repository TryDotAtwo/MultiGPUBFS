# Full BFS owner timeline, local diagnostic

Source: c5553394b068f45eafb57e0a14a6122bacb8e7cb. Existing Linux debug
CLI binary from the macro-merge verification; existing NVTX native library
selected at load time. One RTX 3070 Laptop (sm86), diagnostic NCCL archguard,
not two T4s. Nsight Systems 2025.6.3, cuda/nvtx/osrt, sampling disabled.
No runtime or profiler helper source was changed for this run.

Configuration: S8, world=1, DENSE, NCCL_LSA, CUCO_RANK, pre-dedup ON,
batch=1024, layer=40320, state ring=80640, buckets=8, shards=4,
job buckets=2, bucket capacity=40320, fixed library pool=64 MiB.
Matrix-u8 states and mandatory archive, 512 rows/slot and 128 pinned slots;
warmup disabled. MGBFS_TRACE_RANGES=1; TRACE_ROUTE not enabled.

Artifacts: test_results/owner_timeline_c555339_20261002/dense.nsys-rep,
dense.sqlite, archive-rank-0.mgbfsar1, result/rank-0.json and group commit.
Independent verify_process_archives(n=8,world=1) passed: 40320 unique full
canonical matrix states, every one of the 29 distance layers matches CPU.

## Host API attribution

NVTX containment uses matching globalTid and a fully contained API interval.
61 mgbfs.batch ranges contain 100 cudaMemcpyAsync and 100 cudaMemcpy2DAsync
calls, and **zero** cudaMemcpy, cudaStreamSynchronize or cudaEventSynchronize.
The async calls include archive D2H; absence of synchronous APIs is not proof
of no device dependencies or complete overlap.

FinalizeDepth ranges contain 348 synchronous cudaMemcpy (20.473 ms aggregate
API time), 145 cudaStreamSynchronize (3.304 ms) and 532 cudaMemcpyAsync
(7.952 ms). Whole-capture totals include initialization and archive worker:
447 cudaMemcpy, 262 cudaStreamSynchronize and 230 cudaEventSynchronize.
These whole-process numbers must not be called batch hot-path waits.

Recorded GPU interval union: 19.245 ms busy in 613.586 ms recorded span;
0.833 ms multi-stream activity, 0.237 ms compute/copy overlap. This is not
occupancy, FLOPS, a critical-path estimate or the BFS timer. The small S8
diagnostic does not saturate the GPU and cannot establish peak throughput.

Profiled search=0.374622070 s, durable commit=0.388716267 s. Explicit device
plan=75813632 aligned bytes; setup/final cudaMemGetInfo observed 1750597632
bytes, **not** a continuously sampled peak. These are one instrumented run,
not an A/B or speedup claim.

## Remaining gate

One rank cannot validate peer payload traffic, asymmetric cancellation,
receive-slot reuse across processes, or healthy-next-batch behavior on
two T4s. Target two-rank trace, all asymmetric faults and NCCL registration
initcheck remain open. Distributed macro depth also remains unsupported.
