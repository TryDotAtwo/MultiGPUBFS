# CUCO_INDEXED boundary

Source pin: NVIDIA/cuCollections `532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4`.
The official mapping-table example stores integer indices and compares the
original larger keys through a custom equality functor. Thus a 128-bit Hash128
need not be truncated to fit the T4-compatible atomic key representation.
The example explicitly is not a performance demonstration, and its illustrative
hasher reads only the first component. Do not copy that hash policy into BFS.

Source: https://github.com/NVIDIA/cuCollections/blob/532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4/examples/static_set/mapping_table_example.cu

Integration consequences (our inference, not an upstream performance claim):

- Equality must load and compare all four u32 words. Index equality alone is
  wrong, and fingerprint equality alone would change the collision contract.
- Every stored index must continue to address immutable keys. A candidate slot
  cannot be recycled while the table still references it.
- A bounded prototype can rebuild the index from committed storage before
  reusing transient candidate storage. Rebuild time and scratch must count in
  the comparison; this is not assumed to win against CUB or cuDF.
- Use an explicit invalid-index sentinel outside all valid rows; guard count
  bounds before insertion. Table capacity must be preallocated and checked.
- Concurrent insertion does not by itself establish minimum source-ordinal
  provenance. If the shared owner interface promises KEEP_FIRST, test and
  implement that selection rather than assuming insertion order.

Required next evidence: pinned dependency build on sm75; all-word-distinct keys,
duplicate candidates, committed overlap, transient-slot reuse and full-capacity
failure fixtures; then real owner comparison. No CUCO runtime result exists yet.

## Proposed persistent/transient split (not implemented or measured)

The first indexed design need not rebuild accepted membership after every batch.
Use two fixed-capacity sets per shard:

- Persistent set: indices into previous, current and accepted key planes. After
  StateRing credit and accepted-key copy, insert the new **accepted** indices,
  never candidate-slot indices. Accepted rows remain immutable for the depth.
- Transient set: indices into one candidate slot, cleared only after all result
  readers finish. It deduplicates the current batch independently of history.

Proposed index representation is u64: top two bits identify previous/current/
accepted/candidate storage, low 62 bits identify the row. Actual row bounds stay
at the ABI's signed-32-bit limit. UINT64_MAX is outside every valid index. All
four Hash128 words participate in equality and hashing; these index bits do not
replace the hash or change its collision contract.

Concurrent insertion may select an arbitrary representative. After insertion
has finished, find that representative and atomic-min the original candidate
**row** into an incoming-sized ordinal array. Select the minimum row and return
its supplied source index. Minimum source-index value is not a substitute for
KEEP_FIRST when callers provide non-monotonic provenance values.

Candidate probing reads the persistent set but cannot publish into it. Commit
copies the selected keys to stable accepted storage before inserting their
stable indices. Keep an explicit exclusive shard lease and stream ordering.
At depth finalization, drain and discard membership before overwriting history.
Capacity must cover previous+current+maximum accepted rows at the chosen load
factor, plus the independent incoming table, ordinal/select scratch and actual
rounded cuco storage extent. Use a fixed-RMM allocator adapter; the default
cuco CUDA allocator is not acceptable for this profile's physical-pool contract.

Pinned `static_set.cuh` exposes allocator and stream constructor parameters,
`insert_and_find_async` and device `ref` operations. These are API feasibility
observations, not evidence that the proposed adapter builds or performs well.
The earlier rebuild approach remains a simpler reference candidate; its cost
must be measured if implemented. The two-set option also has extra memory and
random-probe costs and is not presumed to beat the sort/merge baseline.
