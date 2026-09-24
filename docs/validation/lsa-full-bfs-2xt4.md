# LSA full-BFS gate, 2×T4, 2026-09-24

Private Kaggle notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v1 built
immutable source `76c333798ad599c5cb52c29d0afae5c3f7f35e47` with pinned
NCCL 2.29.7, CUDA 12.9, CUTLASS and cuCollections. The runtime selected
`ReferenceTransport::Lsa` and `CUCO_RANK` in the actual two-GPU DENSE BFS,
not only in a transport microbenchmark. Both physical devices were Tesla T4.

The ignored integration test
`cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle` passed. It
compared complete layer state sets and verified per-rank archives against the
CPU oracle for unitriangular(3,3) and permutation S4, owner maps `[0,1]`
and `[1,0]`, and pre-dedup `OFF` and `ON` (eight fixtures total). It includes
multiple depth and peer-exchange epochs, but is a small-graph correctness
gate, not a throughput or capacity measurement.

The previous leaf gate v9 at source `9a4b11b` ran the transport-fatal import
on each T4. Plain, `memcheck`, `racecheck`, `initcheck` and `synccheck` all
passed for that leaf. It does **not** sanitize the full LSA BFS. The LSA
transport's kernel-filtered `initcheck` still timed out in the earlier v20
probe, and unfiltered setup sanitizer results remain unresolved. The full
BFS gate ran plain only.

Version v1 still read route counts on the CPU after each pack even in LSA
mode. The follow-up source change after this run removes that D2H route-count
read for LSA, but requires its own physical gate. Host synchronization for
generation-buffer reuse, owner control, retirement and failure collectives
remains; this is **not** a CPU-free end-to-end pipeline. The LSA receive slot
is additionally allocated alongside legacy receive buffers, so no VRAM
improvement is claimed yet.

Raw evidence: `test_results/kaggle_lsa_full_bfs_v1/lsa-bfs-gate/summary.json`,
`lsa-full-bfs.log`, `native-build.log`, `library-build.log` and pinned build
logs in the same directory.
