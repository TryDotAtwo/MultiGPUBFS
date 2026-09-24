# Asymmetric archive-slot fatal, physical 2×T4 (Kaggle v42)

Private `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v42 ran source
`b93d2a2701b9ebfe64674178d979dc544f3eaad0` on two P2P-capable T4s.
The rank-0 pinned archive ring had two slots and a five-second disk worker;
rank 1 had ample slots. Both exact integration tests passed:

| Transport | Test | Result |
|---|---|---|
| Host-sized NCCL | `archive_slot_failure_votes_group_fatal_before_exchange` | 1 passed, 45.82 s |
| NCCL LSA / CUCO_RANK | `archive_slot_failure_votes_group_fatal_before_lsa_exchange` | 1 passed, 45.81 s |

Each fixture asserts that rank 0 sees `ARCHIVE_PIN_RING_FATAL`, rank 1 sees
`REMOTE_ARCHIVE_FATAL`, and both worker threads exit. This reconfirms the
per-batch archive failure vote at the later runtime source. It does **not**
justify deleting that host vote or prove all other asymmetric failures safe.

Raw evidence: `test_results/kaggle_archive_fault_v42/lsa-bfs-gate/summary.json`
and the two `archive_slot_failure_votes_group_fatal_*.log` files.
