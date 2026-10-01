# NCCL source-build admission, 2026-10-02

Scope: pinned NCCL v2.29.7-1, b91894bd5b190c874d98a017f93f5daa515b65d0,
stock versus minimum-architecture experimental patch. No production default change.

## Observed failures before GPU acceptance

- Stock sm86 source build with NVTX=0 completed, but the actual CLI under
  unfiltered memcheck exited 1 before its first instrumented CUDA API:
  `undefined symbol: _Z23initNvtxRegisteredEnumsv`.
- `ldd` resolved libnccl.so.2 to the intended stock source-build directory,
  not the wheel. Raw log is in
  `test_results/local_lsa_20261001/stock-source-sm86-memcheck/rank0.log`.
- Upstream src/Makefile excludes init_nvtx.cc with NVTX=0, while init.cc
  calls initNvtxRegisteredEnums. This run is not a sanitizer pass or a BFS result.
- Candidate source copied from Windows initially failed with
  `/usr/bin/env: 'python3\r': No such file or directory`.
  Build-copy tracked files were mechanically normalized to LF. Its git diff
  remained exactly five files and 19 additions matching the experimental patch.

## In progress

Candidate full CUDA build resumed successfully after line-ending normalization.
Stock host sources are being rebuilt with the upstream default NVTX=1.
Device source search found no NVTX use in src/device; existing device objects
remain the full, unchanged sm86 collective families. Candidate must also use
NVTX=1 before the paired test. No sanitizer acceptance is claimed here.

Both builds emit upstream dev_runtime.cc:1042 maybe-uninitialized warning for
outDevCommPreserve. It is retained as a warning, not promoted to a proven bug.
Two-rank T4 acceptance and the registration initcheck failure remain open.

## Stock NVTX=1 actual BFS result

The stock host rebuild completed. The same CLI/config with the verified stock
library path then executed S4 with capture enabled under unfiltered memcheck.
Observed process exit 1, ERROR SUMMARY: 12 errors, all reported as error 209
at cudaFuncGetAttributes/cudaGetLastError in ncclInitKernelsForDevice.
Thus compiling pristine source specifically for sm86 does not remove the
previous wheel's architecture-probe failures.

Raw log and archive:
`test_results/local_lsa_20261001/stock-source-sm86-nvtx1-memcheck/`.
Seven owner DAG launch markers were present. Independent checksummed archive
reader plus canonical CPU oracle passed all 24 unique states, with layer sizes
`[1, 3, 5, 6, 5, 3, 1]`. Command:
`verify_process_archives(case, n=4, world=1)` from the existing replay script.
This is correctness evidence for one rank only; memcheck remains FAILED.
Candidate comparison is still pending its live full CUDA build.
