# Archive consumer profile: S11, Kaggle v23

Source `34c8c90f44f528552e693280c9d83a7cc57682d3`, physical 2xT4.
COMPLETE, 39,916,800 records, 40 Parquet shards. GPU archive oracle test PASS.
Search 2.265903158 s; native archive completion 22.56014379 s.

| Consumer stage, seconds | Rank 0 | Rank 1 |
|---|---:|---:|
| FIFO reads including waiting | 3.711 | 3.878 |
| Frame checksum | 3.106 | 2.949 |
| Arrow construction | 2.290 | 2.212 |
| Sink add_batch | 14.093 | 14.366 |
| Parquet encoding (subset of sink time) | 13.391 | 13.248 |

Each rank consumed approximately 19.958M records in 121 record frames.
Do not add encode time to sink time. Final partial-shard encoding is included
in the encode counter but not add_batch time; stage totals are diagnostic,
not a precise end-to-end wall-clock decomposition. Preuploads overlap and
their durations must not be summed as elapsed wall time.

Evidence: `test_results/hf-editor-v23/s11-hf-stream/summary.json` and both
`streamer-rank-*.stderr.log`. Peak live upload slots were 5/4 of 8. S11 did
not exhaust that queue. Encoding is the largest measured serial consumer
stage in this run; this does not prove networking can sustain S13.

The current writer uses dictionary encoding and ZSTD on every column.
Unique states, hashes and rank ordinals do not benefit from dictionaries.
Next candidate is column-selective dictionaries (metadata only), measured
against the current writer with exact Parquet roundtrip tests. Avoid changing
logical schema, archival checksum, or upload lifetime rules.

Primary API reference: https://arrow.apache.org/docs/python/parquet.html

HF publication:
https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/commit/17d2ef0337fcb00ecc381799314dcdd40bd5ea40

No new S13 launch: its backlog remains unresolved.

## Column-selective dictionaries: v24

Source `915e59bf997c8fad5b2a28ca312bca19460f40d7`, same S11 capacity,
batch, archive rows and slots. Only metadata columns retain dictionaries;
ZSTD remains unchanged. COMPLETE, 40 shards / 39,916,800 records.
All global layer counts match v23; the physical two-GPU oracle test passed.

| Metric | v23 | v24 |
|---|---:|---:|
| Rank 0 encode seconds | 13.391 | 8.242 |
| Rank 1 encode seconds | 13.248 | 8.393 |
| Search seconds | 2.266 | 2.114 |
| Native archive completion seconds | 22.560 | 16.981 |
| Peak VRAM per rank, MiB | 699 | 699 |

Encoding time fell 37–38%, native archive completion about 25%. These are
single separate Kaggle runs, not repeated paired measurements on one host.
No change in BFS throughput is attributed to this writer-only optimization.
Rank read/checksum/Arrow times remain roughly 3.6/2.8/2.2 seconds; encoding
is still the largest individual stage. No evidence yet that S13 can sustain
the consumer rate without overflowing its pinned ring.

Evidence: `test_results/hf-editor-v24/s11-hf-stream/`.
HF promotion:
https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/commit/2f77dce99de88f2d8eac20859305da5051d6187e

## Hash compression experiment: v25, no demonstrated benefit

Source d621ed4 used NONE on hash128_le only. S11 COMPLETE and GPU oracle
passed. Encoding was 8.983/8.895 seconds, native archive 18.438 seconds and
search 2.093 seconds. This separate run does not isolate CPU/network noise,
but it provides no improvement over v24 (8.242/8.393 and 16.981 seconds).
Total Parquet upload bytes were 918,146,419 versus 918,184,067 in v24:
effectively unchanged. Keep the v24 ZSTD writer; do not promote an unproven
optimization. Exact state/hash roundtrip tests remain.

Evidence: `test_results/hf-editor-v25/s11-hf-stream/`.
Publication: https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/commit/ffdc21ec0cf602617245c1c8cb645165f504f431

Next investigation should measure column-level encoding costs in a bounded
same-host replay (no GPU rerun for every codec hypothesis). Avoid repeated
full builds and HF publications merely to compare writer options. Keep
network validation as a separate gate after choosing the local writer.
