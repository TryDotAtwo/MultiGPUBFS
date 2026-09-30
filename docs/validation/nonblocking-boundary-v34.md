# v34 pre-test runner failure

TerminalERROR, sourceab65ce1. SummaryINCOMPLETE with
No module named 'eight_gpu_gate'. Existing module import path was missing
in the newly added capture-runner branch. No owner probe built/launched;
no GPU assertion or sanitizer result. Logs retained under
test_results/kaggle_nonblocking_boundary_v34/lsa-bfs-gate.

8b5c954 fixes the runner import; 248434d separately corrects reviewer-found
lease completion ordering in the capture fixture. Host-only completion is
not a GPU graph node and occurs after launch/drain. Next declared package
validates full owner capture first and, only on a P2P-capable host, the six
LSA process faults. Missing P2P reports that second gate unsupported, never
substitutes HostSizedNccl or converts partial acceptance to full PASS.
