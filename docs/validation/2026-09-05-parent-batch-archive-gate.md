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

## S11 gate v21: COMPLETE

Same source; physical two-device archive test passed again (1.56 seconds).
The S11 run completed with 39,916,800 records. Search took 2.239227736 seconds;
native archive completion took 21.780343483 seconds. Peak VRAM was 699 MiB
per rank with this smaller 4M/rank capacity configuration. Do not compare
this allocation directly with historical runs using other capacities.

Publication succeeded for `s11-native-2xt4-20260905-134725`:
https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/commit/2a14618f03da7134c38c1ed975d331705ad89f66

The substantial archive tail remains visible; this does NOT establish that
archiving is free or that sustained S13 backlog fits. Evidence is in
`test_results/hf-editor-v21/s11-hf-stream/`.

S13 v22 has now been submitted with source d690b46, 220M records/rank,
262144 batch/archive rows and 256 pinned slots. It is a new implementation
test, not a rerun of the failed whole-layer archiver. Final publication and
capacity outcome remain pending.

## S13 v22: INCOMPLETE, pinned ring exhausted again

The revised implementation completed depth 35 and failed while processing
depth 36 with `ARCHIVE_PIN_RING_FATAL: receiving on an empty channel`.
Torchrun reports rank 1 exit 1 and peer termination at 13:56:26 UTC.
Both streamers subsequently report `ARCHIVE_TRUNCATED`; no complete graph
was promoted. Evidence: `test_results/hf-editor-v22/s11-hf-stream/`.

Parent-batch submission removes the whole-layer enqueue burst but is not
sufficient to keep the archive consumer up with S13. The same-size pinned
ring still accumulates backlog. This run must not be repeated unchanged or
described as fixed. Logs do not yet isolate time in FIFO/checksum, Arrow
construction, Parquet encoding, and network preupload. Measure these stages
on Kaggle before choosing a consumer optimization or capacity change.

The two-device full-state archive test passed in the preceding gates; this
failure concerns bounded throughput, not proof that the generated states
are incorrect. The incomplete archive remains unsuitable as a full dataset.
