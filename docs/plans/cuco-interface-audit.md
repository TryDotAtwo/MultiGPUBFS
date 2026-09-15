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
