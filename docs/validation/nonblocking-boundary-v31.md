# v31: originating owner error obscured by launcher cancellation

Terminal Kaggle ERROR, source928c25e, two physical T4 with bidirectional
P2P1. Explicit LSA ON, two independent rank processes. Downloaded artifacts:
test_results/kaggle_nonblocking_boundary_v31/lsa-bfs-gate.

First owner-fault case stopped the gate: only REMOTE_SEARCH_CANCELLED was
printed by rank1. Torchrun then terminated rank0 with SIGTERM. Harness
reported PROCESS_FAULT_NOT_REACHED because the original injected error was
not printed. This proves neither non-execution of the injection nor bounded
independent runtime termination. Later five cases did not run.

Corrective runtime change publishes originating errors before abort/cleanup.
The next fault fixture sets torchrun monitor interval30 seconds and rejects
launcher SIGTERM plus supervisor forced cleanup. Both processes must have
returned independently before the launcher inspection for that gate to pass.
This changes only diagnostic timing, not production healthy batch scheduling.

No correctness/performance gate is closed by v31. Do not retry to select a
different P2P host: this host was supported and the result is substantive.
