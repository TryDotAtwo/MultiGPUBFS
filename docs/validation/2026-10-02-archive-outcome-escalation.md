# Archive failure after provisional search success

Production change: `crates/mgbfs-runtime/src/bootstrap.rs`.

The rank-side search loop previously sent only one outcome (`sent: bool`).
An asynchronous archive failure published after search-success therefore
could not notify ranks still searching. The coordinator also rejected a
success-to-failure update as a duplicate. Final archive voting still protects
the durable completion contract; this finding is not evidence of a corrupt
graph or an observed false group COMPLETE.

The same sideband now tracks unknown/success/failure and permits exactly the
monotone success-to-failure escalation. Exact duplicates and reversal remain
invalid. The retirement loop consumes a queued escalation independently of
reader-retirement ACKs; failure alone never authorizes window release.
No GPU buffers, payload padding, NCCL owner thread or healthy batch readback
were added. A failure after the search sideband has completed is still handled
by the existing final archive boundary vote.

## Verification

Real loopback TCP connections exercise the actual production sideband loops.
Before the fix, peer escalation was missing and the coordinator closed the
connection on the update. After the fix, both regressions pass. A third test
holds the reader ACK back and checks that queued archive failure does not
authorize retirement.

- `cargo test --locked -p mgbfs-runtime --features cuda,library-owner --lib
  --test bootstrap -- --test-threads=1`: exit 0; 23 unit tests and 20 bootstrap
  integration tests pass, including all three new cases.
- `cargo test --locked -- --test-threads=1`: exit 0, full default-members CPU
  suite passes.
- `cargo test --locked -p mgbfs-cli --features library-owner --
  --test-threads=1`: exit 0; all 14 CLI tests pass, including real single-GPU
  archive failures, full-state/hash archive checks and owner capture.
- The ordinary parallel `cargo test --locked` is NOT green: bootstrap tests
  below failed with `BOOTSTRAP_TIMEOUT`. This repeats the prior unresolved
  bootstrap timeout; no timeout increase or suppression was applied.

Parallel failures:

- archive_admission_failure_reaches_both_ranks_before_nccl_setup
- archive_worker_signal_cancels_peer_without_a_producer_poll
- configuration_agreement_rejects_asymmetric_error_and_digest_mismatch
- file_rendezvous_shares_coordinator_nccl_id_and_control_connections
- group_publication_failure_is_reported_to_both_ranks
- search_sideband_reports_asymmetric_failure_and_preserves_healthy_boundary
- startup_boundary_leaves_control_connection_admissible_to_dispatcher

This is CPU protocol evidence, not independent-process two-GPU failure
acceptance. Required 2xT4 gates, registration initcheck, full distributed
timeline and paired A/B remain open. Existing unused_mut warning at
reference_bench.rs:578 remains unchanged.
