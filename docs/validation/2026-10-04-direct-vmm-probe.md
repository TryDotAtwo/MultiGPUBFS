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

Follow-up after v101 passed all ten pairs: the same test now optionally
accepts `--runtime-init`. In this variant both ranks map/import before
initializing their own allocation with `cudaMemsetAsync` on an explicit
nonblocking stream. A rendezvous after stream completion precedes peer
reads. Literal expected values distinguish both ranks; the local control
and the original Driver API path remain in the same 20-pair matrix.
This isolates API/order differences, not NCCL shadow-table/address-layout
behavior. Actual sm75 compilation with Wall/Wextra passes; 60 script tests
OK/3 skipped. Hardware execution is not yet claimed for this variant.

The expanded variant passed all20 pairs on the second notebook v6,
including local/import initialization after mapping under all four tools.
All40 rank logs retained. A further `--runtime-init --symmetric` variant
now matches the inspected NCCL2.29.7 address layout: a virtual reservation
for two rank strides, each total-VRAM rounded to4GiB, aligned512MiB, with
recommended physical allocation granularity. Each rank maps its allocation
into its rank-indexed range and imports the peer into the other range.
This is virtual address space, NOT32GiB physical VRAM allocation on a T4.
Only two granularity-sized mappings exist. Reservation lifetime extends
past both mappings; no access crosses an unmapped gap. The 30-pair matrix
retains earlier controls. Compilation sm75/Wall/Wextra and scripts61 pass
(3 skips); the symmetric variant is not yet hardware-verified.
