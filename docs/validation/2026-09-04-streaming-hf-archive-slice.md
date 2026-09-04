# Bounded streaming archive slice

Status: `static-green`; T4 FIFO integration and live Hugging Face promotion are
still pending.

Implemented:

- `StreamExtent` preserves the checksummed `MGBFSAR1` byte stream while writing
  strictly forward to a FIFO instead of reserving the entire graph archive.
- `HubStagingSink` owns a fixed number of fixed-size host-RAM Parquet buffers.
  A buffer is reusable only after its upload future returns successfully.
- Exhausted slots, an oversized Parquet shard, upload failure, archive
  truncation, chain mismatch, or missing `RunCommit` are fatal. No complete
  stream receipt is emitted in those cases.
- State shards are uploaded only to a run-specific staging branch. Per-rank
  receipts retain checksums, layer counts, archive-chain digests, and remote
  source paths. `combine_rank_commits` rejects missing ranks or mismatched
  configuration before any global promotion.
- The existing file-backed, physically preallocated archive remains the default;
  FIFO mode is explicit through `MGBFS_ARCHIVE_STREAM=1`.

Local evidence:

```text
cargo test --locked -p mgbfs-runtime
py -m unittest discover -s tests -p "test_stream_hf_archive.py" -v
py -m unittest discover -s tests -p "test_promote_hf_stream.py" -v
```

This slice does not yet claim live Hub writes, server-side promotion, cleanup of
failed staging branches, or sustained overlap with the native two-rank runtime.
