# Direct CUDA VMM import diagnostic

`experiments/cuda_posix_import.cu` isolates POSIX VMM import without NCCL or
the BFS runtime. It uses two independent processes, rank-selected CUDA
contexts, exported allocation FDs passed using Unix `SCM_RIGHTS`, and
explicit initialization before import/read. A bounded socket rendezvous
protects export lifetime until both GPU readers finish. The local control
uses the same VMM allocator without an imported mapping.

The existing Kaggle gate offers mode `cuda_posix_import`: local/import ×
plain/memcheck/racecheck/initcheck/synccheck, retaining each rank's raw log,
commands, exit codes and timeout result. No tool filters or API-error
suppressions are added. Two completion markers and zero exit codes are
required; a timeout or CUDA API failure is not a pass.

Local checks on 2026-10-04:

- Actual `nvcc -std=c++17 -arch=sm_75 -lineinfo ... -lcuda` compilation in the
  existing GPU toolchain image succeeded. This is not GPU execution.
- Matrix-selection contract was RED before implementation; the two probe
  tests and five existing typed-followup tests then passed.
- Target-hardware execution is pending. Both existing Kaggle notebooks were
  still RUNNING when prepared; neither was overwritten or duplicated.

This diagnostic cannot close the full BFS sanitizer gate. Its purpose is
to distinguish a CUDA import/instrumentation failure from an NCCL-specific
resource setup failure. Even an import-only initcheck failure must be
compared with the retained full-runtime error and phase before attributing
a common cause. Production barriers/resources are not disabled.
