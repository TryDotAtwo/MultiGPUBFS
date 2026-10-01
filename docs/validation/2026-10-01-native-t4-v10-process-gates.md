# Native owner T4 v10: correctness/failure results, not full acceptance

Immutable source: 3d7297e7fa8065ea2b61c5a300cb6a5066f8d1e8.
Notebook: trydotatwo/mgbfs-native-rank-owner-t4, version 10.
Hardware: two actual T4s, P2P verified in both directions.

The notebook ran 16 single-rank S4 searches: each physical GPU separately,
CUB/BMMA, DENSE/HASH_FIRST, local pre-dedup ON/OFF. Every checksummed archive
was also downloaded and independently reverified locally: 24 unique states,
layer sizes [1,3,5,6,5,3,1], no duplicate states or rank-count mismatch.

Four independent-two-process suites (both backends and profiles) report
60/60 passing cases. These include four healthy full-state searches and 56
asymmetric faults: startup, early/late constructor, owner host error,
pre-communicator archive admission, archive finish and device capacity,
injected on each rank. No forced process cleanup or false group COMPLETE
was reported. Maximum fault duration: 2.259264798999993 seconds.

Twelve additional healthy independent-two-process runs cover reversed owner
maps and pre-dedup OFF. All pass their full-state oracle. The downloaded
process summaries are available; not every raw two-rank archive has yet been
downloaded, so independent LOCAL revalidation of all two-rank archives is
not claimed. Single-rank local revalidation above is complete.

Full-BFS racecheck and synccheck pass for both backends/profiles (8 runs).
Memcheck and initcheck fail for all four combinations (8 runs). The observed
memcheck report contains 13 NCCL startup API errors, including unsupported
kernel availability probes. Initcheck returns CUDA_STATUS_1 before successful
completion despite reporting zero memory errors. No suppression or API-error
filter was used in these full-BFS gates. They remain FAIL, not waived.

All four full S8 Nsight attempts failed before BFS with PROFILE_NVTX_NOT_BUILT.
No successful production timeline or overlap claim follows from these traces.
aa97205 adds the checksummed CUDA 12.9.79 NVTX3 component and enables the
existing CMake NVTX option. One v11 notebook tests that source; results pending.

Evidence directories:
- test_results/kaggle_native_rank_v10_summary/lsa-bfs-gate/
- test_results/kaggle_native_rank_v10_20261001/lsa-bfs-gate/
- test_results/kaggle_native_rank_v10_trace_logs/lsa-bfs-gate/

The installed Kaggle CLI ignores its parsed version argument for output API
requests. After v11 started, a retry without a pinned session is not proof
of v10 provenance; the attempted SDK version_label=10 request returned 404.
Existing files are preserved, and missing historical outputs are not silently
replaced with a later run. Summary source SHA binds the saved reports above.

Still open: all four unfiltered full-BFS sanitizer gates; successful full
Nsight traces and batch attribution; larger full-state graph gates; paired
A/B time/durable/VRAM; other original feature, macro and HF completion gates.
