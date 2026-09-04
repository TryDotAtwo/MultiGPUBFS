# Symmetric catalog Kaggle gate v6

Source commit: `52e5e58c0bc0f151b6ea8f9b0de64f2aac8eb909`

Kaggle kernel: `trydotatwo/mgbfs-hugging-face-publisher`, version 6.
Hardware observed by the gate: Tesla T4, 15,360 MiB total, 14,912 MiB free.

The exhaustive native `macro_bench ... verify` run, archive export, Parquet
conversion and independent Parquet replay verifier all completed. These timings
include the expensive per-layer verify snapshots and are correctness/catalog
timings, not the tuned CayleyPy A/B benchmark.

| Group | Unique states | Layers | Search s | Durable s | CUDA allocation | Pinned RAM | Disk extent | Verification manifest SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S8 | 40,320 | 29 | 0.077758579 | 0.287817267 | 642,646,016 B | 374,865,920 B | 70,334,464 B | `a61265f13e0518e41d966ab825c32c0bbd6bd20da81f2b11f8cef1ca180d8290` |
| S9 | 362,880 | 37 | 0.436525946 | 0.628297510 | 915,275,776 B | 454,524,928 B | 102,308,224 B | `ae1a8e04929fe09204077b76c38a5077beae5f9ad68d5b09ce6056fd2b988406` |
| S10 | 3,628,800 | 46 | 6.137391299 | 6.337196300 | 3,350,069,248 B | 543,555,584 B | 488,049,664 B | `c70481bcaa087d69997a9c2c4ccb9f2230514335b5fa2f5ca4905edc97183fdf` |

Maximum depths were 28, 36 and 45 respectively. CUDA context usage was
112,066,560 bytes in every run and is separate from the allocation column.

Every run reports `hash_state_pairs_verified=true`. The catalog uploader is
append-only and the verified Parquet package remains in Kaggle output under
`symmetric-catalog/catalog-upload/`.

Publication is the only incomplete step. The gate ended with
`COMPUTE_COMPLETE_UPLOAD_PENDING` because the Kaggle secret `HF_TOKEN` was not
available. It did not silently skip verification or discard the staged data.
