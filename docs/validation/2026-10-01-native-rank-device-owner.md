# Native rank device-count candidate

The existing CUB/BMMA bounded owner now builds bucket jobs from device window
counts and immutable depth directories. Compare/commit reuse job_buckets*K
merge scratch; only job/count metadata scales with buckets. Empty buckets do
not scan or rewrite history. Selected indices retain absolute frame offsets.

DistributedNativeBfs connects this path to the existing LSA epoch credits,
single-communicator dispatcher, fatal gates, StateRing rank reservation,
DENSE materialization/extent publication and HASH_FIRST requests/responses.
History directories are uploaded at depth boundaries, not read inside batches.
Host-sized NCCL remains explicit and unchanged; there is no fallback.

Verified locally: Rust checks with cuda and cuda,library-owner; twelve memory
planner tests; CUDA sm75 compilation; CUB leaf on RTX 3070 Laptop (sm86),
driver 572.70, CUDA 12.8. The leaf verifies a nonzero input offset, four
rejection categories, empty buckets, insufficient grant, sticky fatal and
bucket capacity rejection before commit. Four unfiltered leaf sanitizer tools
pass. Raw logs/binary: test_results/native-rank-device-20261001/.

These checks do not prove full native BFS, BMMA on sm75, inter-rank ordering,
fault propagation, full-BFS sanitizer gates or performance. The existing
Kaggle provisioner has a native_rank_gate for separate single-GPU T4 runs
of both backends/profiles/pre-dedup choices and actual owner-DAG capture.
Two-process fault/oracle suites run only after a verified P2P preflight.
Missing P2P is explicitly multi-GPU UNSUPPORTED, never a passing acceptance.

Kaggle v6 (42ba753) and v7 (81257b6) both ended ERROR before BFS. Both
verified real 2xT4 P2P in both directions and all eight unfiltered native
leaf sanitizer gates (CUB/BMMA x four tools). This is leaf evidence only.
v6 failed configuration selection (REFERENCE_TRANSPORT_BACKEND); 81257b6
fixed that and its regression test was observed RED then GREEN. v7 then
failed the separate runtime constructor guard (LSA_REQUIRES_DENSE_RANK_OWNER).
9bcafdd fixes that guard for the integrated native device-count path,
including world=1 for the single-GPU reference gate; Rust feature check passes.
Raw v7 evidence: test_results/kaggle_native_rank_v7_20261001/lsa-bfs-gate/.
The CLI output downloader reported a Windows charmap error after downloading
the summary and rank log; those two files were inspected, not inferred from
download exit status. Full native BFS, faults and performance remain unproven.

One existing notebook trydotatwo/mgbfs-native-rank-owner-t4 v8 is launched
with immutable source 9bcafdd07f347225cedd2a6f5034a03f04f2b2f8. The old
slug was renamed by Kaggle; it is not a second notebook. v8 adds reverse
rank maps, both pre-dedup settings and full two-process sanitizer runs.
