# Device communicator without a user window: diagnostic preflight

The existing `device_comm` fixture registers a user window first. It cannot
distinguish registration failure from independent device-communicator activation.
`experiments/nccl_window_isolation.cu device_comm_only` now creates the same
16-barrier LSA device communicator without `ncclMemAlloc` or
`ncclCommWindowRegister`, then destroys it and the host communicator. It uses
the existing nonblocking progress polling and independent-process bootstrap.

Compiled successfully with CUDA 12.9 nvcc, sm75, and the existing pinned
NCCL 2.29.7 explicit-POSIX candidate in the Linux build volume on 2026-10-03.
Compilation is not a hardware or sanitizer result. Neither initcheck nor the
full production correctness/throughput gates are closed by this diagnostic.

Next physical two-T4 run must compare plain and initcheck execution of
`nonblocking`, `device_comm_only`, and `device_comm`, preserving both rank logs
and requiring the requested stage plus natural successful process exits.
Do not filter sanitizer errors, replace LSA transport, or waive initcheck.

No rental was created in this step. Both the legacy billing endpoint and the
current official CLI charges endpoint returned HTTP 401 with the approved key;
remaining project spend is therefore unverified, not zero. Existing rentals
must not be inferred live from their saved receipt files.
