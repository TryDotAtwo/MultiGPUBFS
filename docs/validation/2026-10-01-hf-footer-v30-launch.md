# S13 HF footer audit v30

The previous v29 audit ended without a full proof: 5,448 of 6,228 immutable objects were processed before its 7,200-second limit.

On 2026-10-01, the existing private CPU-only notebook trydotatwo/mgbfs-s11-hf-stream was changed from one footer reader to four. No source/runtime or remote dataset objects changed. The existing auditor is pinned to fc8efe0660f377931995654f9cd6c0b35e6b0d3e and checked by SHA-256; the dataset revision remains d43c3aa640ef12935ff12f986e53d3e6fef6e92f and manifest runs/s13-native-2xt4-20260905-152407.json.

Local Python AST/metadata validation passed: private=true, GPU=false, TPU=false. The existing remote-footer unit tests passed 2/2. Wrapper configuration is published as 89897f5.

Kaggle explicitly accepted version 30; subsequent status was RUNNING. No second notebook was launched and no GPU quota or paid rental was used. Results must still be downloaded and checked; this launch is not a footer-verification claim. No state payload is downloaded to the user's computer.

## Completed result

Kaggle subsequently reported COMPLETE. Downloaded footer-summary.json reports VERIFIED_FOOTERS, 6,228 files and 6,227,020,800 rows, matching 13 factorial. Rank 0: 3,113,547,204 rows; rank 1: 3,113,473,596 rows; each has 3,114 contiguous parts. The schema is consistent across all files. Audit elapsed time: 1,110.159 seconds.

Evidence: test_results/kaggle_hf_footer_v30/s11-hf-stream/footer-summary.json, SHA-256 562e45c1e2e37a71b5a52a4926b81dfd0c1739bfea649f4d7c71f65aea66584a; notebook log is retained alongside it. Download contained only logs/summary, not graph states. No remote dataset objects were rewritten.

This closes the previously incomplete all-file footer-count/schema check at the pinned revision. It does not prove payload checksums, uniqueness of every state, per-depth contents or correctness of the newer runtime; footer row totals are not an independent full-state oracle.
