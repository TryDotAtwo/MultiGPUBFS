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

## Stock NVTX=1 initcheck

Same one-rank S4/config, unfiltered initcheck: exit 0, ERROR SUMMARY: 0 errors.
Seven capture launch markers and durable archive are present. Independent
full-state oracle again passed 24 states with the same seven layer sets.
Raw artifacts: `test_results/local_lsa_20261001/stock-source-sm86-nvtx1-initcheck/`.
This does not close the previous two-rank T4 registration initcheck failure.

## Remaining stock local tools

Unfiltered synccheck and racecheck both exited 0. Synccheck reported zero
errors; racecheck reported zero hazards, errors and warnings. Each produced
seven capture launch markers. Each separate archive passed the full-state
CPU oracle, 24 unique states and the same seven layers.
Artifacts use the same prefix with `-synccheck` and `-racecheck` suffixes.
No filtering or suppression was used. Overall four-tool acceptance still
fails because memcheck failed; this is one rank, not T4 group acceptance.

Hardware: RTX 3070 Laptop GPU, 8192 MiB, driver 572.70.
Compute Sanitizer 2025.1.0.0 build 35583870; CUDA build 12.8.93.
SHA256 of actual tested binaries:

- stock libnccl.so.2.29.7: a7b0971d91e22e28a8dcb2a4b0901778e367d4a50b6d53a1d970656a5ef5a49d
- libmgbfs_cuda.so: 5ab7a5ec3d0c2634f73584b79e791c1b84063a61738787103231d6f7ae3a03de
- mgbfs: 438a1a92a66a26b8304a0675537101d4a53fa81136bf59760f35f97d55a8f747

## Candidate result: four local tools PASS

The NVTX=1 candidate host rebuild completed and `ldd -r` showed no unresolved
symbols. Actual candidate NCCL SHA256:
`dc42ed2f993d31908ca248cc9994bc0b46d3e124ecb4a509d7dd14f145a18856`.
Actual runtime resolution selected /linux-build/nccl-archguard-build/lib.
Same CLI binary, native library, S4 config and GPU as stock above.

All four unfiltered tools exited 0: memcheck/initcheck/synccheck report zero
errors; racecheck reports zero hazards, errors and warnings. Each run launched
seven captured owner DAGs. Each independent archive passed the canonical CPU
oracle at every depth, 24 unique states, [1,3,5,6,5,3,1]. Raw logs and archives
are under `test_results/local_lsa_20261001/archguard-sm86-nvtx1-<tool>/`.
No sanitizer API reporting was disabled or errors suppressed.

Stock versus candidate isolates a local minimum-architecture probe fix:
stock memcheck 12 errors versus candidate zero, with both full-state oracles
passing. Generated normal/symmetric tables retain all stock tokens except
added metadata, respectively 40 and 50 aligned entries. Comparison initially
failed byte equality because of one extra blank line; token comparison passed.
This does NOT establish two-rank T4 registration correctness, full mode
coverage, protocol races, production-default admission or performance benefit.
