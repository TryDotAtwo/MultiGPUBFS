# Explicit search-only contract check

The upcoming Vast LRX13 run is search-only by user request. Default archived
CLI behavior is unchanged. The explicit flag allows the library owners without
an archive; it does not silently select another backend.

Search-only now constructs no `PinnedArchive`, reserves zero archive disk and
pinned bytes, and records `output_contract=search_only_layer_counts` with a null
`durable_run_commit_seconds`. Archived execution still consumes and finishes
the archive object normally.

Validation on 2026-09-16:

- `cargo test -p mgbfs-core --test reference_selection`: 7 passed.
- `cargo test -p mgbfs-cli --test bench`: 4 passed.
- Linux container `cargo check --locked -p mgbfs-runtime -p mgbfs-cli
  --features cuda,library-owner --all-targets`: exit 0.

The Linux check found an ownership error in the first draft (`finish` consumes
the archive); this was corrected with `Option::take` and the check repeated.
These are contract/type checks, not H200 execution or performance evidence.
