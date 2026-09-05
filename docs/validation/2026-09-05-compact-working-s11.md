# Compact permutation working state: first T4 result

Source: 9f440a1ceff379f1c9df57fe97baa6ff98bd21c0. Kaggle s11-distributed-probe v6.
Two physical Tesla T4 GPUs; DENSE native NCCL, batch 262144, capacity 8M/rank.
All 56 global layer counts match the matrix ring v4 run, totaling 39,916,800.
This is count agreement, not a full-state-set or archive roundtrip verification.

| Representation | Search seconds | Durable seconds | Peak MiB/rank |
|---|---:|---:|---:|
| Matrix ring v4 | 4.070927190 | 44.650878652 | 2587 |
| Compact ring v6 | 2.010380659 | 8.201729035 | 1075 |

Single historical runs, not controlled repeated A/B. Archive enabled for both,
but matrix archive versus compact archive changes output encoding and byte count.
No tuned CayleyPy comparison claimed. Compact archive stores 11-byte states;
VRAM uses 16-byte padded states, with hashes computed over 11 bytes.

Generation gather-reference test passes n=3,12,17 and batches 1,7,65.
memcheck/racecheck/initcheck/synccheck report zero errors/hazards on both T4s.
These sanitizer results cover the generation test only, not the entire runtime.
Evidence: test_results/s11-compact-working-v6/s11-distributed-probe/.
