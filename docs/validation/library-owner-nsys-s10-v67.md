# Two-T4 CUCO_RANK S10 diagnostic profile v67

Private Kaggle `trydotatwo/mgbfs-library-owner-t4`, version 67, source
`ae3dce9433508faae6730132b43b072eea84dd78`. The notebook completed
with status `PASS`. One profiled S10 DENSE, CUCO_RANK, pre-dedup ON,
HostSizedNccl run used two physical Tesla T4s, batch 32768, 1,000,000
state/ring records per rank, a 96 MiB cuCO pool per rank, and 256 pinned
archive slots per rank. Both archives passed checksum/count verification.
This is a diagnostic timeline, not an unprofiled speed comparison or
evidence for the separate LSA transport.

| Rank | Search s | Durable commit s | Setup/final-sampled device bytes | Pinned archive bytes |
|---:|---:|---:|---:|---:|
| 0 | 0.5265674 | 2.790716688 | 583,991,296 | 973,078,528 |
| 1 | 0.520303064 | 10.78433529 | 583,991,296 | 973,078,528 |

The search times are Nsight-instrumented; the marked durable asymmetry
needs a repeated archive-lane investigation. It does not establish a
repeatable rank imbalance. The 50 ms external VRAM sampler and the runtime's
setup/final samples are not byte-exact peak proofs.

Nsight's captured CUDA-API aggregate across both processes includes 2,002
`cudaStreamSynchronize` calls totaling 466.3 ms and 2,460 `cudaMemcpy`
calls totaling 123.6 ms. Listed GPU kernel aggregates include 540 NCCL
send/receive kernels totaling 222.1 ms, 1,092 NCCL all-reduce kernels
totaling 62.3 ms, CUB radix onesweep totaling 81.3 ms, modular
materialization totaling 68.7 ms, and generation CUTLASS GEMM totaling
25.1 ms. These are **sums of activity durations**, which can overlap;
they are not additive critical-path times. The result supports work on
transport/control and route before GEMM-only tuning, but does not prove
which proposed change speeds the whole BFS.

The profiler capture is the timed BFS plus archive submissions. Warmup and
durable archive drain are outside it. Raw manifest, Nsight report/SQLite,
statistics and both rank JSON files are retained under
`test_results/kaggle_library_profile_v67/library-owner/`.
