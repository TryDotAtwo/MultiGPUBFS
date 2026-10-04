# Two seeded runs per graph — acceptance requirements

Requested 2026-10-04. Existing eight-GPU single-run calibration is historical evidence, not acceptance of this requirement.

- Run each eligible (n,r) twice consecutively, independently, same start/actions/packing.
- Different MGBFS_HASH_SEED_HEX values (128-bit native GEMM hash seeds); record actual native reported seeds. Existing native binary supports this environment variable.
- Compare completed-layer state counts and status/last completed layer; exclude timing and sampled VRAM from equality.
- Preserve two sets of complete timing/VRAM statistics and five complete final layers per run.
- Publish both outcomes even when mismatch, with matched boolean and comparison scope/reason; missing/unfinished second run must not be called matched.
- Keep replicas identifiable in HF manifests and Parquet, including when payload cohorts share files.
- Compare immediately after second graph run; preserve fastest end-upload/storage-pressure upload modes.
- Resource pruning must use confirmed pair outcomes, not transient mismatch or one seeded allocation outcome alone.
- State fingerprints, if enabled, must be independent of rank distribution and row ordering and carry an explicit probabilistic verification scope.
- Current single-run live process cannot be upgraded by editing its Python source; transition through verified cancellation/publication or after current graph without losing existing evidence.
- Test identical counts/different times, count mismatch, status mismatch, missing second outcome, reordered states and actual seed propagation. Target eight-GPU evidence remains required within existing total $20 cap.
