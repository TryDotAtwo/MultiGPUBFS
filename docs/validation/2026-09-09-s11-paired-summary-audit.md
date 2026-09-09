# S11 paired panel: summary reconciliation

Source `66d82d03cb055daa08dae328978208efda7c8ede`; baseline
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`. Kaggle v13 one rank and
v5 two ranks both terminal COMPLETE. Downloaded environment and summaries
match these commits. `scripts/audit_s11_panel.py` independently reconciles
all 120 native measured runs, their rank-level layer sums, warmup flags,
archive verifier results, memory sampling completeness and five repetitions
per configuration. All report 39,916,800 states across 56 layers.
This verifies summary consistency, not independent replay of full state sets.
Raw per-rank logs still require separate reconciliation.

Seconds are medians of five fresh processes. Peak VRAM is total sampled MiB.
Native archives are mandatory; CayleyPy produces no archive.

| Configuration | 1 T4 search | 2 T4 search | 1 T4 durable | 2 T4 durable | VRAM 1 / 2 |
|---|---:|---:|---:|---:|---:|
| CayleyPy tuned | 21.8761 | 18.7506 | n/a | n/a | 14713 / 26368 |
| DENSE scalar CUB OFF | 7.4109 | 4.8759 | 12.6499 | 10.2216 | 7521 / 7586 |
| DENSE scalar CUB ON | 7.4488 | 4.9796 | 12.5684 | 10.5050 | 7521 / 7586 |
| DENSE scalar BMMA OFF | 68.6969 | 35.9203 | 69.5586 | 38.6197 | 7521 / 7586 |
| DENSE scalar BMMA ON | 68.5042 | 35.8138 | 69.3547 | 38.6447 | 7521 / 7586 |
| HASH_FIRST scalar CUB OFF | 9.9243 | 6.7305 | 15.0725 | 11.7721 | 7417 / 7378 |
| HASH_FIRST scalar CUB ON | 9.9566 | 6.6412 | 15.1109 | 11.8312 | 7417 / 7378 |
| HASH_FIRST scalar BMMA OFF | 71.2860 | 37.5935 | 72.2548 | 38.7749 | 7417 / 7378 |
| HASH_FIRST scalar BMMA ON | 71.1955 | 37.5506 | 72.3076 | 38.7387 | 7417 / 7378 |
| HASH_FIRST MMA CUB OFF | 10.6854 | 7.4840 | 15.8290 | 12.5045 | 7417 / 7378 |
| HASH_FIRST MMA CUB ON | 10.7916 | 7.4432 | 15.9064 | 12.4600 | 7417 / 7378 |
| HASH_FIRST MMA BMMA OFF | 72.2973 | 38.3087 | 73.2855 | 39.6687 | 7417 / 7378 |
| HASH_FIRST MMA BMMA ON | 71.8266 | 38.2019 | 72.9665 | 39.4991 | 7417 / 7378 |

Both baseline batch-1048576 calibration rows failed (exit 1), retained in
the audit rather than silently discarded. Failure cause requires raw logs.
DENSE CUB OFF search MAD is 0.003079 / 0.060297 seconds. Its 1-to-2-GPU
scaling is approximately 1.52x, not 2x. Archive durability remains a material
tail. BMMA and integer MMA do not improve end-to-end time on this workload.
These runs predate the separate exchange stream (b3c191a); v45 is currently
RUNNING and cannot yet validate that later change.
