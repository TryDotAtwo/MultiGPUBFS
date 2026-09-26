# S13 Hugging Face remote Parquet footer audit (2026-09-27)

Dataset: `TryDotAtwo/multigpubfs-bfs-results`, immutable revision
`d43c3aa640ef12935ff12f986e53d3e6fef6e92f`, run
`s13-native-2xt4-20260905-152407`. This is a read-only remote-metadata
check. No state payload or Parquet object was downloaded to the workstation.

The public run manifest still resolves as `COMPLETE`, lists 6,228 state
objects and 6,227,020,800 unique states, and contains 79 layer counts whose
sum is 6,227,020,800. The Dataset Viewer reports the older `layers`, `runs`
and `states` configs; it does not index this S13 run's `states/<run>-...`
objects as a separate viewer split. Therefore its `/size` total cannot be
used as the S13 row-count proof.

PyArrow read remote footers through `HfFileSystem` at the immutable revision,
using ranged HTTP reads. A 32-file random/end-point pilot passed: every
sample had eight columns and at most 1,000,000 rows. The four end-point
objects reported 1,000,000 and 547,204 rows for rank 0, and 1,000,000 and
473,596 for rank 1 (first and last part respectively).

A subsequent 16-worker attempt inspected all manifest paths but completed
footer reads for only 5,601/6,228 files. All 3,114 rank-0 parts were read:
their footer row total is 3,113,547,204. For rank 1, 2,487/3,114 parts were
read, totaling 2,486,473,596 rows. The other 627 requests failed with HTTP
429: Hugging Face rate-limited the anonymous client. All successfully read
files had eight columns, 1..1,000,000 rows and unique part numbers. The
scan's overall validation exited nonzero. **Do not report full remote footer
verification as passed.**

One diagnostic compared `str(ParquetSchema)` values and found many apparent
differences. That comparison is invalid because the string includes a Python
object address. Two sampled schemas had the same printed physical fields
after the address. A complete schema check must compare Arrow/Parquet field
structure, not object representations.

Next gate: repeat the missing/full footer scan with the existing Kaggle
`HF_TOKEN`, modest concurrency and retry/backoff, without downloading state
payloads. Verify every part number per rank, each footer row count and the
structural schema; require summed footer rows and layer counts both to equal
6,227,020,800. This remains metadata integrity, not independent BFS-state
equality or a proof against 128-bit hash collision.
