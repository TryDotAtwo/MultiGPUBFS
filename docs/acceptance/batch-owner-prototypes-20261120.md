# Experimental batch GPU primitives: bounded route and owner

This is an engineering checkpoint, not a working BFS strategy. The production
runtime still rejects MGBFS_SEGMENT_STRATEGY=batch. Existing Atomic and Warp
paths are unchanged. These standalone CUDA modules are not yet linked into
the production native library and have not passed full-layer BFS acceptance.

- segmented_batch_route.cu: bounded route bucket sort and dense scatter;
  overflow is reported before any route record is scattered. The existing
  owner and NCCL protocol are not replaced by this prototype.
- permutation_bitmap_batch.cu: one exclusive shard transaction, exact local
  IDs, word-level mask reduction, scan, bounded reservation, materialization
  and word publication. Same-word transactions must be ordered by the caller.
- segmented_hash_batch.cu: exact sorted-origin lookup and immutable runs in
  segment-local indexes. Tail compaction moves only key/row indexes; if merge
  workspace is too small, both runs stay searchable.

Standalone physical tests used two RTX 3060 devices and CUDA 12.8 SM86. Route
tests cover successful queue contents, overflow and retry. Bitmap tests cover
word boundaries, duplicates, spill, bounded tiles and ordered concurrent
streams. Hash tests cover forced hash collisions, full state comparison,
bounded tiles, spill and deferred/complete index compaction. Compute Sanitizer
memcheck reported no errors for the tested binaries. These tests execute on
both devices separately; logical two-rank tests are not an end-to-end
distributed run.

Still required: route/owner lifecycle integration, memory admission and byte
receipts, rank-wide cursor/drain handling, history retirement, no-leak
persistent execution, full exact BFS oracle checks, physical multi-GPU
acceptance and the six-cell A/B/C large-layer benchmark. The prototypes alone
must not be selected for production or treated as proof of full BFS accuracy.
