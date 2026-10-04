# Initcheck / LSA IPC provenance

This is a diagnostic hypothesis, not sanitizer acceptance or a runtime fix.

## Confirmed source path

Inspected local NCCL 2.29.7 source under `build/nccl-source-2.29.7`:

- `src/dev_runtime.cc:183` (`symMemoryMapLsaTeam`) imports each other local
  rank's allocation with `cuMemImportFromShareableHandle`, then maps it with
  `cuMemMap` and grants device access with `cuMemSetAccess`.
- The POSIX path obtains a remote file descriptor through
  `ncclProxyClientGetFdBlocking`; this is inter-process sharing, not merely
  peer access within one CUDA process.
- `symMemoryObtain` calls that mapping path for new symmetric memory.
- Device communicator resource allocation creates a CUDA generic allocation,
  calls `symMemoryObtain`, creates its resource window, initializes the local
  resource with `cudaMemsetAsync`, and subsequently synchronizes the stream.
  Thus `device_comm_only` is not an IPC-free control: it removes the user
  window but retains internally shared resource allocations.

## Documented limitation and claim boundary

NVIDIA's current Compute Sanitizer release notes explicitly list initcheck
IPC allocations as unsupported, producing false positives:
https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations
(retrieved 2026-10-04).

The observed v91 failures are unspecified launch errors during registration
or device communicator activation, not an identified uninitialized-access
report. The documentation therefore does NOT establish their cause. Neither
generic VMM nor ordinary peer access should be declared unsupported on this
evidence. No filter, suppression, zero error summary, or successful plain run
closes the requested initcheck gate.

## Pending discriminating result

The sole active Kaggle v92 runs two independent ranks through plain and all
four sanitizer tools for user window registration, internal device resources
without a user window, and combined registration/activation. Collect each
rank exit and the first failing API before selecting the next experiment.

If failure is specific to initcheck before BFS, follow with a bounded direct
CUDA POSIX allocation export/import reproducer, including a local-only
allocation control and explicit initialization. This distinguishes NCCL
resource activation from the underlying imported-allocation instrumentation.
It is diagnostic only and must not replace the full runtime gate.
