# Macro merge active-work bound

The existing MacroNativeBfs producer now calls the existing bounded future
merge API instead of scanning the full reserved future arena every batch.
Each slot carries a host-only conservative record bound: after each enqueue
it becomes min(capacity, old_bound + generated_rows). Depth finalization
refreshes it from the already-required device count inspection; recycling
resets it to zero. This is not a live device count or an extra readback.
The existing CUDA begin/publish guards retain capacity, reference and sticky
fatal checks. No C ABI/kernel/library allocation, stream or transport change.

Actual producer capture regression was RED before the change: real future
live count 6 exceeded the stale bound 5. It is GREEN after bound publication,
still executing three parent batches and reusing both banks under capture.
All 18 CUDA/library-owner runtime unit tests, seven macro integration tests
and 12 CUDA CLI tests pass (session 57479). The preexisting unused_mut warning
in reference_bench is unchanged.

Four fresh mandatory-archive S8 runs (K2/K3 x pre-dedup OFF/ON, batch 1024,
40320-record layer and per-depth future capacity) each completed. Independent
canonical CPU verification confirms 40320 unique full matrix states and all
29 exact distance layer sets in every archive. Artifacts and pinned test
binary hashes: test_results/macro_merge_bound_20261002/.

Four unfiltered sanitizer tools completed sequentially on those exact
test binaries (session 57576, exit 0). Memcheck, initcheck and synccheck each
passed capture plus seven macro tests with zero errors. Racecheck also passed
capture plus all seven tests: zero errors, warnings and hazards, 656.91 seconds
for integration. No filters or suppressions; error-exitcode 97. Full
default-members CPU suite passed with exit 0
(session 56417, including the depth-ten weighted oracle). Do not
reuse the prior candidate's four-tool pass for this new bound change.

This improves an existing prerequisite component, not distributed macro
integration. MACRO_MULTI_GPU_UNSUPPORTED remains. No paired performance
claim, T4 gate, two-rank cancellation or complete-goal claim follows.
