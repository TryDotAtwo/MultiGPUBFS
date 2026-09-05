# Archive submission is currently a whole-layer burst

Inspection after HF v19 failure:

- `distributed_bench.rs` calls `archive_current` before `advance` at every
  depth, not as bounded parent batches are consumed.
- `distributed_native.rs::archive_current` iterates every frontier extent,
  acquiring a pinned slot for each fragment, capped by archive.rows and batch.
- Every fragment occupies a whole slot even when an extent is short.
- No more parents are expanded until that submission loop returns.

Thus pinned capacity is consumed by a whole-layer burst. Its availability
depends on how quickly the worker catches up while submissions are issued,
not simply the average disk/network throughput across the BFS depth.
ARCHIVE_PIN_RING_FATAL is therefore not sufficient evidence that the
long-term upload bandwidth is too low. Earlier reports' throughput language
should be read as an observed pipeline deficit, not an isolated diagnosis.

Required next audit: count actual extents/fragments and submitted/completed
slots at failure, then move archive scheduling into bounded parent processing
while preserving original depth order and D2H-before-state-overwrite events.
Do not add producer waits or silently enlarge slots as the architectural fix.
The three existing native archive_current implementations must remain
explicitly distinguished; this finding is about the distributed path.
