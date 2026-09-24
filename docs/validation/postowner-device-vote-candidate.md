# Post-owner device vote: rejected after host-fault gate

Commit `0ceb7c797687311792f926044a4e79e044dfc7f0` replaces the normal
per-batch host readback after CUCO_RANK DENSE + NCCL LSA owner work with a
stream-ordered GPU group fatal vote. A host API failure still enters a matching
NCCL max-reduction epoch before communicator abort. The device vote is ordered
after local and remote owner jobs and poisons the ring/owner control on every
rank before another owner commit. This is only the LSA two-rank round-1 path;
HostSized, HASH_FIRST, archive-error votes, and later peer rounds retain host
dependencies.

Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v28, exact source `0ceb7c7`,
completed on two peer-accessible T4s. Its full-state DENSE LSA layer/archive
oracle and an asymmetric rank-0 owner-capacity failure test both passed. Raw
logs: `test_results/kaggle_postowner_gate_v28/`. Version 29, source `5b4bfe8`,
was `UNSUPPORTED_HOST` because neither T4 could access the other as a peer;
it did not execute the new host/API-failure test. Version 30 ran the same
source on a P2P-capable pair: no-fault full BFS and device capacity failure
passed, but the injected one-rank host owner error hung until the 300-second
external timeout. Raw logs: `test_results/kaggle_postowner_host_fault_v30/`.
This is a confirmed protocol failure for that injected scenario, not a
sanitizer finding. The exact blocked call is not localized. NCCL's
[fault-tolerance guidance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
requires all active ranks to abort and warns that blocking communicators can
hang in outstanding NCCL calls. This runtime currently uses blocking
`ncclCommInitRank`; unilateral abort is therefore an unsafe assumption, but
the documentation alone does not pinpoint the v30 wait. Commit
`6077bf8470522700c9a88246ff2749e216775381`
restores the blocking post-owner group vote while retaining the fault fixture.
Kaggle v31 reached the capacity fixture, but its assertion expected the
pre-restoration error spelling and rejected the legitimate synchronized
`GROUP_OWNER_OR_PRE_OWNER_FATAL` result. Kaggle v32 did not fetch source
because its abbreviated commit ID was not a fetchable remote ref. Kaggle v33,
exact source `db3e81c2471b045ea36bb2367bd76677166ab366`, completed the
full-state DENSE LSA layer/archive oracle, asymmetric rank-0 owner-capacity
failure, and asymmetric injected rank-0 host owner error on two P2P-capable
T4s. Each fault test ended on both ranks in about 1.3 seconds. Raw test logs:
`test_results/kaggle_postowner_restored_v33/`. This verifies the restored
blocking protocol for these fixtures; it does not verify every API error,
large-frontier termination latency, or an asynchronous owner pipeline.

Under the rejected GPU-only candidate, the successful capacity test did
**not** establish prompt termination: the host could continue the fixed
`scheduled_rounds` loop after a device fatal and learn the sticky error only
at `FinalizeDepth`. Later owner jobs could not commit, but generation, route,
exchange and archive submission could continue. The restored blocking vote
at `6077bf8` reads the group result after each owner batch and returns from
that loop when nonzero; the v33 small-graph fault fixtures exercised this.
Large-frontier failure latency is still unmeasured, and any future removal of
that host readback needs a bounded cancellation protocol. For scale, existing
8-rank data at batch 262,144 had a peak
of 177 scheduled rounds on S13 (depth 54) and 1,261 on LRX15r4 (depth 68).
An error in the first round could leave 176 or 1,260 rounds of the same depth
to issue. These are structural upper bounds from different runs, **not**
measured fatal-tail durations for this candidate.

The candidate is rejected, not selected as runtime behavior. A replacement
needs an explicit asymmetric host/API failure protocol and bounded device-fatal
cancellation before removing the host vote. Full-app four-tool sanitizer,
stage-bounded Nsight timeline, and repeated speed/VRAM A/B remain open.
