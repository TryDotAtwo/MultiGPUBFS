# Physical source route banks: local validation

Scope: runtime commit `28b1f29fd85ab3ddc1ddc2c3183a49bac6a5655c`, built from
the previously uncommitted patch on public base `5d777e3`. This is not a
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
  HASH_FIRST used scalar generation on this sm86 host, not its sm75 Tensor Core
  generation backend. DENSE used its existing CUTLASS generation path.
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
This local pass does not resolve the separate two-T4 NCCL
registration/initcheck failure.

Follow-up initcheck and synccheck on the archived sm86 fixture passed with zero
errors (17.15 and 8.83 seconds instrumented test duration). Racecheck at 180
seconds made progress through three complete cases and into the fourth case,
then timed out. A 600-second repeat was justified by those flushed phase markers,
not by assuming a timeout is harmless. The repeat completed in 372.44 seconds:
exit 0, zero hazards, zero errors and zero warnings, all six cases passed.
The 180-second timeout logs are retained. All four tools therefore passed this
local one-rank archived fixture on the sm86-only NCCL stack, not the two-T4 gate.

Pinned diagnostic binary SHA-256 values:

- NCCL 2.29.7 sm86:
  `874b8a3497c65362e07f29285ad984332dd4c93e698e8449073013e9828cd6e1`
- Rust GPU fixture:
  `860a348eb06f16654907905a9db8ab821b7b5fe6fdc196de487ba65ccc01474a`

## Local full-BFS timeline

Nsight Systems 2025.6.3; same runtime fixture, native library rebuilt with NVTX
enabled, same sm86 NCCL. Full-state/archive checks passed during capture.
Raw report and SQLite: `build/route-banks-local-20261003/cuco-route-banks.*`.
This is world1, six separate tiny BFS fixtures, not a 2xT4 throughput run.

The trace contains 384 `mgbfs.batch` ranges and 54 `mgbfs.FinalizeDepth` ranges.
Joining CUDA runtime APIs to same-thread NVTX ranges by API start timestamp:

- Inside batch: zero `cudaStreamSynchronize`, `cudaEventSynchronize` and
  blocking `cudaMemcpy` calls.
- Batch-correlated D2H: 768 transfers, each 16 bytes, total 12,288 bytes.
  All 768 also fall inside `mgbfs.archive_d2h`, matching the state/hash archive
  payload; this is not count/control readback evidence.
- Blocking copies and stream synchronization remain at FinalizeDepth, as
  required by the semantic layer boundary.
- The export lacks `CUDA_CALLCHAINS` on this local stack, so the existing
  callchain analyzer explicitly rejected it. The range-based findings do not
  claim stack attribution or complete absence of all possible host dependencies.

Recorded GPU intervals across this entire diagnostic capture: busy 143,417,135
ns; multi-stream activity 3,955,983 ns; compute/copy overlap 1,082,446 ns. These
include setup and tiny fixtures, not occupancy or a saturated performance gate.
No ideal overlap or target-hardware speedup is claimed.

Reproduction of the batch API check:

```sql
SELECT s.value, COUNT(*), SUM(a.end-a.start)
FROM CUPTI_ACTIVITY_KIND_RUNTIME a
JOIN StringIds s ON s.id=a.nameId
JOIN NVTX_EVENTS n ON n.globalTid=a.globalTid
  AND a.start>=n.start AND a.start<n.end
WHERE n.text='mgbfs.batch'
GROUP BY s.value;
```

## Remaining gates

Two independent ranks on actual 2xT4; asymmetrical failure injection; archive
durability; all four sanitizer gates on the supported target stack; full BFS
Nsight timeline; paired performance/memory measurements. Existing NCCL
registration initcheck issue on the two-rank target remains open.
