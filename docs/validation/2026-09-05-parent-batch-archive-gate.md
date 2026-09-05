# Parent-batch archive gate, Kaggle v20

Source: `d690b462d9fe6fc6cddd412853861b8ff9960c18`.
Notebook: `trydotatwo/mgbfs-s11-hf-stream`, version 20, COMPLETE.
Hardware: real 2x Tesla T4.

The `distributed_archive` test passed (1 test, 1.51 seconds). It compares
all archived matrix states per depth against the CPU oracle, verifies each
state/hash pair and archive framing, uses reversed owner mapping and archive
rows smaller than a compute batch. This is a small correctness gate, not a
sustained-throughput or whole-runtime sanitizer gate.

S8 then completed with 40,320 records and 29 nonempty global layers:

| Metric | Value |
|---|---:|
| Search seconds | 0.313459529 |
| Native archive completion seconds | 1.321621507 |
| Peak VRAM per rank | 437 MiB |

Native archive completion is not global HF promotion latency. No speed claim
is inferred from this small single run.

HF promotion succeeded for `s8-native-2xt4-20260905-134005`:
https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/commit/1534eaf9155c463a1fd6b1b0afb4f9f7bb2d00b5

Evidence remains under `test_results/hf-editor-v20/s11-hf-stream/`:
`distributed-archive-test.log`, `s11-native-hf.json`, `promote.log`.
Only small logs/metrics were downloaded locally, not graph data.

Next gate submitted as v21: S11, same source, 4,000,000 state records/rank,
262,144 parent batch and archive rows, 256 pinned slots. This exercises more
archive data but does not by itself prove S13 sustained backlog is resolved.
S13 is still not published.
