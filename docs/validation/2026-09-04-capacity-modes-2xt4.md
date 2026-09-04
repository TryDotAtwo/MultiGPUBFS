# Cluster capacity modes on 2xT4 (2026-09-04)

Private Kaggle notebook `trydotatwo/mgbfs-capacity-modes-t4`, version 1,
compiled source `3378317bfce34ed38c06a886d692c9f8f6a91769` and completed the
same exhaustive S8 BFS under both immutable allocation policies.  Both runs
produced the exact 40,320 states and the same 29 layer counts.

| mode | records/rank | records/cluster | search s | durable s | CUDA peak/rank | nvidia-smi peak/rank |
|---|---:|---:|---:|---:|---:|---:|
| `EQUAL_GLOBAL` | 20,160 | 40,320 | 0.104569 | 0.258624 | 273,547,264 B | 261 MiB |
| `MAX_PER_RANK` | 40,320 | 80,640 | 0.112631 | 0.393150 | 292,421,632 B | 279 MiB |

Each rank additionally used a fixed 20 MiB pinned archive ring and reserved
70,334,464 disk bytes.  Therefore this gate isolates the device-capacity
contract: `EQUAL_GLOBAL` divides one declared global record budget across the
ranks, while `MAX_PER_RANK` allocates that budget independently on every rank.
The former reduced the measured device allocation by 18,874,368 bytes per T4
for this small workload.

This is a correctness and accounting gate, not a stable performance result:
the S8 search lasts only about one tenth of a second.  The next timing comparison
must repeat both modes on S11 or larger with the same archive contract.

Raw logs, per-rank JSON and 50 ms `nvidia-smi` samples are retained under
`artifacts/kaggle/capacity-modes-v1/capacity-modes/`.
