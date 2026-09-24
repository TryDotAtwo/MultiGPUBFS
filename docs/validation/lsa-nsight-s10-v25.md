# Two-T4 LSA vs HostSized Nsight diagnostic (v25)

Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v25 ran exact source
`424dd966aa08c1989fe851505a4023d78f532267` on two P2P Tesla T4s.
One profiled S10 CUCO_RANK DENSE, local pre-dedup ON run per transport
completed 46 layers and 3,628,800 states and verified both rank archives.
Nsight Systems 2025.3.2 traced CUDA, NVTX, and OS runtime activity.

| Whole-session CUDA API metric, both ranks | HostSized NCCL | NCCL LSA |
|---|---:|---:|
| `cudaStreamSynchronize` calls | 6,290 | 4,534 |
| Sum of host-side sync durations | 1,506.6 ms | 952.0 ms |
| Synchronous `cudaMemcpy` calls | 4,960 | 3,528 |

| GPU kernel timeline | HostSized rank 0 | HostSized rank 1 | LSA rank 0 | LSA rank 1 |
|---|---:|---:|---:|---:|
| Communication interval union | 499.132 ms | 443.872 ms | 318.185 ms | 307.960 ms |
| Compute interval union | 335.020 ms | 326.987 ms | 439.676 ms | 464.119 ms |
| Intersection of those unions | 40.321 ms | 36.853 ms | 132.778 ms | 140.137 ms |
| Union of all kernel intervals | 793.832 ms | 734.006 ms | 625.083 ms | 631.942 ms |

Communication is classified by demangled kernel names containing `nccl` or
`lsa_`; all other kernels are compute. The interval analysis excludes DMA.
The capture covers process startup, warmup, BFS, and archive, so these are
whole-session diagnostics, not BFS-only timings. There is one profiled run
per transport, not a repeated causal comparison. The observed GPU overlap
does not establish a CPU-free pipeline. Stage-bounded timeline analysis and
the full-app sanitizer/fault gates remain open.

Raw output: `D:\MultiGPUBFS\test_results\kaggle_lsa_nsight_v25\lsa-bfs-gate`.
