# Physical source route banks: local validation

Scope: uncommitted runtime patch on public base `5d777e3`. This is not a
two-rank/T4 acceptance receipt and is not a performance measurement.

## Implemented

- Configurable 2/3/4 preallocated source banks, distinct from completion credits
  and the single LSA receive slot.
- Bank-local generation buffers, controls and generation completion events.
- Bounded generation lookahead; bank reuse waits on the last owner reader using
  a device event. Existing receive-slot ownership and FinalizeDepth remain.
- Allocation ledger charges every physical bank; no padded maximum payload is
  introduced.

## Verified locally

- CPU Rust suite: passed.
- CUDA/library-owner all-target compilation: passed.
- Memory planner tests: 13 passed; production-config tests: 5 passed.
- RTX 3070 Laptop GPU: full-state CPU-oracle comparisons passed for native CUB
  and CUCO_RANK, DENSE and HASH_FIRST, banks 2/3/4, including reuse across depths.
  World size was one; these tests do not exercise remote transport.
- Follow-up archived oracle gate: both native CUB and CUCO_RANK passed all six
  bank/profile combinations with full archived state sets and per-state CPU
  hash checks, checksummed archive verification and successful archive finish.
  The test extent is memory-backed, so this is not physical disk fsync evidence.
- Project Python suite (`tests scripts`): 244 passed, 14 skipped. Bare pytest
  cannot collect this checkout because two unrelated vendored CayleyPy copies
  have the same module names (23 import-file-mismatch collection errors).

## Sanitizer results, not an acceptance pass

Raw logs: `build/route-banks-local-20261003/` (ignored build output).
Compute Sanitizer version: 2025.1.0.0, build 35583870.

- memcheck: exit 99, 624 reported errors. Printed errors are CUDA API errors
  `cudaErrorSymbolNotFound` and `cudaErrorNoKernelImageForDevice` during NCCL
  initialization (`ncclInitKernelsForDevice`). 524 errors were omitted by the
  tool's print limit; no blanket assertion about all errors is justified.
- racecheck: timeout, exit 124 after 180 seconds. Not passed.
- initcheck: exit 0, zero reported errors for this local one-rank fixture.
- synccheck: exit 0, zero reported errors for this local one-rank fixture.

The cached native CUDA 12.9/sm75 build failed route-query initialization on this
host. Native CUDA 12.8/sm86 rebuild passed query and oracle tests. Library owner
uses CUDA 12.9/sm86 because its cached RMM headers require newer CUDA symbols.
The mixed toolchain/NCCL initialization errors need isolation before claiming a
memcheck pass; they are not suppressed or declared harmless.

## Replay integration

The replay CLI now selects physical route banks independently of K, takes typed
bank count from RunConfigV1 rather than inherited environment, and checks the
observed bank count in each rank result. The typed hardware harness now creates
48 configurations (profiles x banks x credits x pre-dedup x rank maps); sanitizer
output labels include bank count to avoid overwriting another case's evidence.
No new Kaggle run has been launched; the source pin will be advanced only after
the runtime change is committed.

## NCCL architecture isolation follow-up

Rebuilt the same cached `nccl-posix-source` with CUDA 12.8 and
`NVCC_GENCODE=-gencode=arch=compute_86,code=sm_86` into a separate cache directory
`/linux-build/nccl-posix-sm86-build`; the source was not changed. Re-ran the
archived CUCO_RANK fixture with unfiltered memcheck, unlimited error printing
and CUDA API reporting enabled. Result: exit 0, zero errors, all six cases
passed (18.44 seconds instrumented test time, not a search benchmark).
Original failing logs are retained. Binary/NCCL SHA-256 values are in
`build/route-banks-local-20261003/sm86-stack-sha256.txt`.
Other sanitizer tools on this rebuilt stack remain pending until their results
are observed. This local pass does not resolve the separate two-T4 NCCL
registration/initcheck failure.

## Remaining gates

Two independent ranks on actual 2xT4; asymmetrical failure injection; archive
durability; all four sanitizer gates on the supported target stack; full BFS
Nsight timeline; paired performance/memory measurements. Existing NCCL
registration initcheck issue on the two-rank target remains open.
