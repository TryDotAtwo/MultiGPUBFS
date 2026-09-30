# v36: full owner capture passes; LSA fatal termination fails

Source c49f5b9, actual two-T4 supported P2P1 host. Full captured owner DAG
plain, memcheck, racecheck, initcheck and synccheck all PASS. Validated one
shot graph execution and final state/key/publication assertions, not repeated
graph replay, whole-BFS sanitizers or no-host-readback timeline.

First LSA owner-fault rank0 case fails strict termination. Originating
TEST_INJECTED_OWNER_HOST_ERROR and peer REMOTE_SEARCH_CANCELLED are printed.
Only peer emits final ERROR; after30-second monitor interval torchrun sends
SIGTERM to rank0. No graceful rank0 exit, five remaining cases NOT_RUN.
Summary INCOMPLETE, PROCESS_FAULT_LAUNCHER_KILLED_RANK: owner.
Artifacts test_results/kaggle_nonblocking_boundary_v36/lsa-bfs-gate.

This is a substantive LSA error-path problem, unlike the runner/fixture
failures in v34/v35. Ordinary NCCL six faults passed in v33. Abort versus
owner/stream teardown stall is not yet distinguished. Next diagnostic adds
only stage logging around existing abort and teardown waits, without
TRACE_ROUTE or new waits. No wait removal, forced-free or latency claim.
