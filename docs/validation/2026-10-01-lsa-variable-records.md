# Exact variable-width records in the existing LSA transport

Hardware: Vast 53673926, two RTX A4000, driver 580.173.02; CUDA 12.8
native build, sm86, NCCL wheel 2.29.7. This is not the required 2xT4 gate.

`mgbfs_nccl_lsa_exchange_rows` uses the same symmetric window, count kernels,
cancellable rendezvous, sole communicator owner and receive-slot lifetime
contract as the existing DENSE exchange. A nonzero 16-byte-aligned record
width must fit the prepare-time maximum. Source/receive records are dense at
the actual width. Null hash input omits hash traffic and preserves its plane.
The old DENSE ABI delegates to this implementation with full state width.
No second receive slot, allocation, host count readback or maximum-payload
padding was added. Ranks must agree on record width/hash presence per epoch.

RED: the modified physical Rust fixture ran against the old production
library and failed on both ranks: `LSA variable row-width ABI is absent`.
The test used runtime symbol resolution only to observe the missing feature;
its final version uses the normal Rust FFI declaration.

GREEN: the real production library passes legacy hashes+states followed by
16/32/16/64-byte compact exchanges in one receive slot. Literal unequal
partitions include an empty sender and an all-empty epoch. Sentinel checks
verify untouched tail bytes and an untouched hash plane. Widths 0/15/80 are
rejected before submission. Counts [2,2] overflow capacity 3 despite each
partition fitting; both ranks publish zero rows/sticky fatal without touching
payload. Source/receive leases close before test buffer reuse.

The changed native library SHA-256 is
ad5b06f3a211bd705aa6f99ea46de3cf56787817e2f83575fb8600210d663270.
Full BFS regression passes nine independent-rank-process cases: healthy S4
full-state/archive oracle and startup/owner/archive-admission/archive-finish
faults on each rank. Additional U/S rank-map/pre-dedup oracle and owner
capacity tests pass. No forced cleanup or false group COMPLETE in faults.

Compute Sanitizer 2025.2.1, CUDA 12.9, variable-record fixture:

| Tool | Result |
|---|---|
| racecheck | exit 0; 0 hazards/errors/warnings; test passes |
| synccheck | exit 0; ERROR SUMMARY 0; test passes |
| memcheck | exit 97; 26 API errors across two ranks; test passes, gate FAIL |
| initcheck | exit 6 at NCCL LSA activation; no forced cleanup; gate FAIL |

Sanitizer runs used the same test body with runtime symbol resolution before
switching to its final static FFI call. Neither failure is suppressed or
reported as a pass. Logs and full-BFS archives live in
`test_results/vast_a4000_cancel_20261001/mgbfs-lsa-row-width-validation`.

Final static-FFI physical test passes again (exit 0). CPU Python suite passes
200 tests with 8 skips. All four production Rust crates pass their CPU suite
(`cargo test --locked -p mgbfs-core -p mgbfs-runtime -p mgbfs-cli -p mgbfs-cuda`).
The broader `cargo test --workspace` does not build the legacy root prototype:
its build.rs requires the separate MULTIGPUBFS_CUDA_LIB_DIR library. This is
reported, not counted as a green workspace. The remote source also retains
the preexisting unused-mut warning in reference_bench.rs.

This closes the exact-width transport prerequisite, not HASH_FIRST. Its owner
pending counts, descriptors, request/response decisions and materialization
publication still require device-resident integration before enabling LSA
there. DENSE timeline evidence remains scoped to previous S4/S8 measurements;
no new throughput or overlap claim follows from this transfer test.
