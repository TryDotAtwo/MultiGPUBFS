# Final physical-T4 BFS comparison

All rows exhaust the same S10 adjacent-transposition matrix Cayley graph: 3,628,800 unique states in 46 exact layers. Timings are medians of five fresh-process repetitions after a separate calibration. CUDA/NCCL warm-up is outside the timer.

| Hardware | Runtime | Selected batch | Search median | MAD | Peak VRAM / rank | Total peak VRAM | Native speedup | Native memory reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1x Tesla T4 | Native DENSE/CUB | 262,144 | 0.465951 s | 0.005155 s | 3,195 MiB | 3,195 MiB | 3.16x | 71.1% |
| 1x Tesla T4 | CayleyPy BFS | 262,144 | 1.474593 s | 0.005595 s | 11,037 MiB | 11,037 MiB | baseline | baseline |
| 2x Tesla T4 | Native NCCL DENSE | 262,144 | 0.512600 s | 0.003527 s | 3,431 / 3,431 MiB | 6,862 MiB | 2.692x | 62.77% |
| 2x Tesla T4 | CayleyPy torchrun BFS | 1,048,576 | 1.379884 s | 0.012133 s | 9,223 / 9,207 MiB | 18,430 MiB | baseline | baseline |

Native always writes the complete lossless state + Hash128 archive. CayleyPy writes only layer counts. Native durable completion is therefore reported separately: about 4.6 s on 1xT4 and a 3.257536 s median on 2xT4.

The 2xT4 native search is slower than the 1xT4 native search on S10 (`0.512600 / 0.465951 = 1.100x`). This graph is too small for NCCL sharding to amortize its fixed per-depth collective cost; no positive two-GPU scaling claim is made. The relevant A/B result is that the native two-rank runtime beats the tuned two-rank torchrun baseline while preserving the stronger archive contract.

Detailed evidence:

- `docs/validation/2026-09-03-symmetric-single-t4-ab.md`
- `docs/validation/2026-09-04-distributed-two-t4-ab.md`
- `artifacts/kaggle/symmetric-single-v3/symmetric-single-gpu/summary.json`
- `artifacts/kaggle/distributed-bench-v8/distributed-bench/summary.json`
