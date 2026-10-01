# NCCL window registration and sm86 build isolation

Vast 53673926, two RTX A4000, driver 580.173.02. Compute Sanitizer
2025.2.1 (CUDA 12.9). No filters, suppressions or disabled API reporting.

The existing experiments/nccl_window_isolation.cu now supports a diagnostic
two-process registration mode. Set MGBFS_WINDOW_RANK to exactly 0 or 1 and
MGBFS_WINDOW_BOOTSTRAP to the same fresh file; each process selects its own
physical GPU and enters the two-rank nonblocking communicator. The unique
ID is published by atomic rename; malformed/missing bootstrap times out
after 30 seconds. The existing thread-mode GPU read/write sample is retained;
process-mode read_write is explicitly rejected until its rendezvous exists.

RED: the old binary ignored rank=7 and returned 0 instead of rejection.
GREEN: the extended probe rejects it with exit 1. Two independent processes
pass ordinary window registration (exit 0/0, 2.17 seconds), each log reports
only its own rank's success. Under initcheck the same minimal probe fails
registration (exit 7/7, 44.44 seconds), without forced cleanup or BFS code.
This narrows the failure beyond BFS owner/generation, not to a proven cause.

A previous one-process/two-thread probe passed initcheck once. A control
execution with the extended same binary in thread mode instead fails with
exit 7 and unspecified launch failure. Therefore process count alone is NOT
an established explanation; the earlier pass does not demonstrate stable
compatibility. Keep both observations, do not report a universal failure or
universal success.

Separate official NCCL source build:

- tag v2.29.7-1, Git b91894bd5b190c874d98a017f93f5daa515b65d0;
- make -j4 src.build NVCC_GENCODE='-gencode=arch=compute_86,code=sm_86'
  CUDA_HOME=/usr/local/cuda (CUDA 12.8);
- libnccl.so.2.29.7 SHA-256
  4a4fc19a684a16c244013b0ab10be5bcc32226d1cd8811294d484fc7dfcda698;
- separate /tmp/mgbfs-nccl-sm86-sanitizer/build/lib, production wheel untouched.

The thread-mode leaf passes initcheck against this build (exit 0, both
window_register PASS, ERROR SUMMARY 0). Full two-independent-process BFS
still fails initcheck without forced cleanup or group-complete. Full BFS
memcheck still exits 97/97 with 13 API errors/rank although group-complete
exists. Thus narrowing NVCC architecture does NOT resolve full BFS gates.
NCCL INFO explicitly identifies the source library as 2.29.7+cuda12.8,
distinct from the previous wheel's 2.29.7+cuda12.9; backtrace offsets differ.

The source allocator explicitly retries without FABRIC after NOT_PERMITTED;
enqueue.cc intentionally ignores unsupported kernel attribute-query errors.
These explain intended probing mechanisms but do not turn any full BFS
sanitizer failure into a pass, and do not explain launch failure 719.

Raw output directories: test_results/vast_a4000_cancel_20261001/
mgbfs-window-process-lvvjma5p, mgbfs-a4000-source-sm86-initcheck,
mgbfs-a4000-source-sm86-memcheck. Separate logs include the source-build
log and leaf sm86 initcheck. Source-override runs reused the same replay
helper, with only rank LD_LIBRARY_PATH prefixed by the isolated build and
the sanitizer executable changed to CUDA 12.9. Runtime candidate is the
previously validated startup-cancellation patch, SHA-256
45f780db117558472846c082a4eb955ad091c1bf9d5a8499bb6ba52936e76186
against 705684ffce9dfd2dde3f63e1c0de798b412319d6.

T4 and unfiltered memcheck/initcheck acceptance stay OPEN.
