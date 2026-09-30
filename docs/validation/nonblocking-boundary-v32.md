# v32 unsupported LSA host

Source e8f8fa6ab909263fb8bd84fde8d1a908a1ad0e5d, Kaggle terminalCOMPLETE,
validation UNSUPPORTED_HOST. Both actual T4 returned cudaCanAccessPeer
status0/allowed0. Preflight stopped before build and any GPU fixture.
Artifacts: test_results/kaggle_nonblocking_boundary_v32/lsa-bfs-gate.
This closes no fault/LSA/oracle/sanitizer gate. No P2P-host retry loop.

Next run is a separately named HOST_SIZED_NCCL fault control on the same
source: independent processes, six asymmetric faults, original-error
logging, strict rejection of supervisor/launcher kills. This is not an LSA
fallback or CPU-free/performance acceptance. Its purpose is to isolate
rank cancellation from device-LSA lifetime in the supported-host v31 error.

Independent reviewer executed immutable17edc34's two real Linux subreaper
tests in existing isolated CPU Docker:2/2 PASS, no skips. Exact evidence:
C:/Users/Иван Литвак/Documents/Codex/2026-09-30/task-8/process-scope-17edc34-linux-review/result.json.
Normal exit7 and orphan-kill exit99 verified; GPU reader teardown not covered.
