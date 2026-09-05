# Same-host codec replay: CPU Kaggle v1

Notebook `trydotatwo/mgbfs-archive-codec-replay`, source 8c85a94,
PyArrow 24.0.0. One immutable S11 shard, 1M rows, Arrow table 157MB.
Each case ran three times with alternating variant order; all 108 exact
roundtrip comparisons passed. No graph data was downloaded locally.

| All columns, median | Encode seconds | Parquet bytes |
|---|---:|---:|
| ZSTD, all dictionaries | 0.74829 | 26,864,910 |
| ZSTD, metadata dictionaries | 0.44672 | 22,844,694 |
| Snappy, metadata dictionaries | 0.36859 | 29,163,342 |
| LZ4, metadata dictionaries | 0.37075 | 29,866,034 |

Snappy saves approximately 17% encode time but produces 28% more network
payload; LZ4 is no better here. Keep the current selective-dictionary ZSTD
production writer pending a justified end-to-end tradeoff. Codec substitution
alone is unlikely to remove the S13 archive backlog.

ZSTD selective column medians: state 0.14469s, hash 0.05710s, ordinal
0.04841s; repeated run/group/config metadata together 0.15989s. Per-column
times are standalone measurements, not an additive wall-clock decomposition.
The expanded repeated metadata merits investigation before adding more
encoding workers. Any compact Arrow representation must preserve the
published logical schema and exact state/hash bytes.

Input/output evidence: `test_results/archive-codec-v1/codec-input.json`,
`codec-profile.json`. Fixed writer buffer 128MiB excludes source table,
validation table, and encoder scratch; observed Arrow allocation is not
whole-process peak RSS. CPU-only Kaggle hardware differs from GPU notebooks:
these are same-host codec comparisons, not GPU end-to-end performance.

S13 remains unpublished. No new S13 computation was launched.
