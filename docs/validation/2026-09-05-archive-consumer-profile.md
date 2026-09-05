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
