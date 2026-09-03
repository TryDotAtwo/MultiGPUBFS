# MGBFS Hugging Face Dataset V1

The dataset is the complete durable result of one exact BFS, not merely a
benchmark table. A publication is valid only for a checksummed `COMPLETE`
archive after all ranks reached `RunCommit`.

## `states`

One row per unique discovered state: `run_id`, `group_id`, `config_digest`,
physical `rank`, logical owner, shard, bucket, original BFS `depth`, rank-local stable ordinal, canonical
row-major `state` bytes and the exact 16-byte little-endian `Hash128` used by
the run. Files are rank-local bounded Parquet shards with Zstandard compression.

## `layers`

One row per original BFS depth: global unique-state count, state width, archive
payload bytes and a deterministic SHA256 over rank-ordered `(rank,state,hash)`
records. Frequently queried generation, duplicate decomposition, routing,
stage-timing, future-occupancy and imbalance counters have typed nullable
columns. `metrics_json` additionally retains every recorded counter without
forcing absent metrics to zero.

## `runs`

Completion status, total unique states, maximum depth and the lossless native
summary JSON including build, environment, topology, seeds, rank map, profile,
macro generator manifest, capacities, allocation ledger and checksums.

`manifest.json` freezes every Parquet shard's path, size and SHA256. Failed or
incomplete executions are never exported as complete state datasets; their
diagnostics belong to a separately versioned failures table.
