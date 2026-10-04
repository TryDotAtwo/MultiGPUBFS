# Two-T4 NCCL initcheck localization and candidate fix

Hardware: Vast instance 54182144, machine 136299, Tesla T4 x2, driver 595.71.05. Runtime source 52bb77fd8e4cf7ed4187ceeddce598ae5880af43, separate rank processes, native NCCL LSA and CUCO_RANK. No resource disabling or sanitizer filtering.

## Reproduction and change

The unchanged NCCL 2.29.7 build fails initcheck before BFS. Diagnostic host checkpoints isolate the first CUDA 719 to the internal resource-window memset on rank 1; window-table initialization and window creation complete successfully. Replacing runtime memset with Driver API async memset does not resolve it. Mapping the local allocation first, then peers, resolves the reduced probe. The existing rank-indexed virtual-address layout and collective order are unchanged. This establishes a mapping-order-sensitive failure, not a proof of its internal instrumenter/driver cause.

The production candidate contains only local-first cyclic mapping order. Diagnostic host drains, pointer prints and Driver API replacement are absent. The versioned patch is patches/nccl-2.29.7-local-first-map.patch; its SHA256 is 95a042db1698504182c0cf0ee34c8c74dbaf5cb376d1aeeae93d464b0378b5b1. The Kaggle build verifies and applies it and records the digest.

## Measured scope

local-first-production-gate: eight healthy full typed S4 runs pass, both DENSE and HASH_FIRST, four unfiltered Compute Sanitizer tools, actual instrumenter 2026.1.0.0. Every run compares canonical full-state archives and layers to the independent CPU oracle, not just layer counts. Two failure suites contain 19 cases each, all pass: healthy plus asymmetric startup/constructor/owner/archive-admission/archive-worker/archive-finalization/capacity failures. No false COMPLETE and no forced process cleanup. Maximum whole-case duration 2.511363 seconds. These are diagnostic/correctness timings, not search benchmarks.

The first local-first-full-bfs runs returned success on both ranks but their verifier failed because the orchestration interpreter lacked pyarrow. They are retained as FAILED verification evidence; not counted as oracle passes. The subsequent production-gate runs use the prepared dependency environment and pass verification.

The candidate binary includes only whitespace differences from the final pinned patch; the exact pinned patch has been rebuilt with library SHA256 340237ca761410da432e8d36fe01b2824eb86c55cab1c1670129d7d69f76c00c. The exact pinned build passes all 12 repeat panels / 48 cases: both profiles under original 2025.2.1.0 initcheck, both under all four 2026.1.0.0 tools, and two 19-case fault suites. Of these, 36 are injected asymmetric failures; none publishes false COMPLETE or requires forced cleanup. Maximum injected-failure duration is 2.510993s. Raw repeat evidence and checksums are in pinned-local-first-gate/. No complete-project, all-owner, macro-depth, timeline or performance claim is made.

receipt.json checksums retained small logs/JSON. Graph archives remain on the remote instance; these are not HF dataset publications.
