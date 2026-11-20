# S12 segmented atomic vs warp: two RTX 3060

Experimental grouped strategy, not a released build. 2026-11-20, one host with 2 physical RTX 3060, 12 GiB each. Source used in the measurement was local commit 071f411c6c222b6982049a4e88a89dcecc367347; its code matches published e5a2adce88c0e6253f08400a1a58a7888a722d1f. Their only tree differences are the previously published S13 report and its archive. The invocation receipts record native and CUDA-library SHA-256s.

Both variants: exact exchange, adjacent-transposition S12, 479001600 states, 67 completed layers, depth 66, two GPUs, eight local shards, four owner lanes, 8192 parents per route batch, three rolling banks, configured segment ceiling 2^32 and actual 1048576 rows/bank/segment, balanced preset, autotune disabled, no source retries. No C strategy is claimed by this report.

Independent Mahonian polynomial layer counts matched every completed layer in each run. The last two layers were checked as full exact sets and duplicate-free. This is not a full-state comparison of all large layers. The CUDA small-graph gate separately compared full per-layer sets, including directed and repeated-symbol graphs, against a CPU oracle.

| Backend | Strategy | BFS seconds, two runs | Depth 28 -38 seconds, two runs | Sampled peak GPU MiB (per rank) |
|---|---|---:|---:|---:|
| bitmap | atomic | 77.808, 69.169 | 41.184, 36.485 | {'0': 925, '1': 925} |
| bitmap | warp | 28.729, 28.620 | 15.145, 15.061 | {'0': 925, '1': 925} |
| segmented hash | atomic | 79.866, 79.531 | 42.734, 42.591 | {'0': 2303, '1': 2303} |
| segmented hash | warp | 43.279, 43.183 | 23.546, 23.504 | {'0': 2303, '1': 2303} |

Depth 28 -38 includes 259128390 newly reached states and 2829700863 generated edges. These are elapsed layer times; generation, communication and acceptance overlap and are not separately attributed here.

The NVML values above are 1-second nvidia-smi samples and can miss short peaks; they are not exact allocation peaks. Per-rank segment-storage receipts were 603980160 B plus 29937600 B for the bitmap, and 2080375936 B for the hash. Extra send-buffer receipts were 11534400 B/rank. These receipts do not cover every CUDA, NCCL, or transient allocation. Peak live rows were 38179213/rank for bitmap and 38179331/38179095 for hash.

Warp groups route queue reservation and row/cap grants; bitmap also combines same-word masks, while deferred-bit rollback still uses atomics. Hash retains per-slot CAS for exact insertion. Actual effects are workload and hardware dependent and are not evidence for 8 GPUs or B200.

Small GPU checks included complete CPU-oracle layers on two GPUs for S4, directed C6, and repeated symbols; grouped 64-bit matrix and wide-state cases; resource/spill retries; mismatch and invalid strategy rejection before the layer; and failure of one rank followed by a fresh persistent job. Logs and parameter receipts are in the evidence archive. No raw graph states are included.

[Evidence archive](s12-strategy-evidence-20261120.tar.gz), SHA-256 5e4c49bfdf2e51bd44e94ecf39431c98aa7db3b247b629a7599796ff2bbfd4b3.
