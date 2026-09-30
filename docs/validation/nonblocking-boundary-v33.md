# v33 ordinary NCCL asymmetric fault control

Source e8f8fa6, two physical T4, two independent rank processes. Explicit
HOST_SIZED_NCCL. All six startup/owner/archive-finish faults (rank0 and rank1)
completed with both rank exitcodes1. No launcher SIGTERM or supervisor forced
cleanup markers. Expected original and remote errors printed in each log.
No group COMPLETE was accepted. Harness elapsed33-36 seconds per case includes
the deliberate torchrun monitor interval30; it is not BFS or abort latency.

Artifacts test_results/kaggle_nonblocking_boundary_v33/lsa-bfs-gate. Existing
Linux supervisor and independent-session cleanup regression files each2/2PASS.
Summary conservatively retains SUPERVISED_NO_COMPLETE and
runtime_bounded_exit_proven=false; the explicit two-rank exit/error log evidence
is stronger than that older summary label for these exact ordinary-NCCL cases.

No LSA/CPU-free/large graph/sanitizer/timeline or paired performance claim.
Full owner capture probe ab65ce1 is newer and not covered by v33.
