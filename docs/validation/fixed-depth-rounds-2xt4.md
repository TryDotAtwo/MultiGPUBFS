# Fixed-depth peer-round schedule, two-T4 gate

Source `6c74077988114da59043181bd3e49c6b984fe5c5` replaces the
per-parent-batch host `all_max(more)` with one agreed maximum number of
physical parent batches at the depth boundary. Every rank then issues that
many peer epochs, using zero payload after its own parents are exhausted.
The schedule counts physical extents separately because batches cannot cross
a StateRing wrap. Local unit tests cover empty ranks, partial tails, wraps,
invalid extents and unrepresentable counts; the CUDA/library feature
type-check and the core/runtime/CLI CPU suite passed locally.

Private Kaggle notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v10 built
that exact source on two physical P2P-capable Tesla T4s. Three plain
integration tests passed, each comparing full layer state sets and archives
with the independent CPU oracle:

- LSA `CUCO_RANK` DENSE: eight S4/UT fixtures across owner maps and
  pre-dedup modes.
- Host-sized NCCL `CUCO_RANK` DENSE: the same eight configurations.
- Host-sized NCCL native/CUDF/CUCO indexed: DENSE and HASH_FIRST across
  both graph families and both owner maps.

The notebook's v9 attempt at the same source never built: a network read
timeout interrupted the pinned cuDF wheel download. V10 increased only pip's
read timeout/retries and completed. Neither run used Compute Sanitizer for
this change. The gate proves small-graph rank/profile semantics, not large
frontier performance or absence of all remaining host waits. The paired S10
transport screen at source `34b1c81` predates this schedule revision;
a follow-up screen is required before attributing any new timing to it.

Raw evidence: `test_results/kaggle_lsa_full_bfs_v10/lsa-bfs-gate/summary.json`
and its three test logs. The v9 dependency failure is recorded under
`test_results/kaggle_lsa_full_bfs_v9/lsa-bfs-gate/`.
