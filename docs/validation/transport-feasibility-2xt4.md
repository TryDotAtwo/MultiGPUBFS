# Device-driven transport preflight on Kaggle 2xT4

Private Kaggle `trydotatwo/mgbfs-transport-feasibility-t4` v1 completed on two
physical Tesla T4s. Raw `transport-probe.json` and notebook log are retained
under `test_results/transport-probe-t4-v1/`.

- GPU0↔GPU1 topology: `PHB`; `nvidia-smi topo -p2p p` says `OK` in both
  directions. `cudaDeviceCanAccessPeer` returns 1 in both directions.
- The runtime `libnccl.so.2` reports NCCL 2.25.1 (`22501`); no
  `nccl*device*.h` header was found in the checked system/CUDA include roots.
- NVCC is CUDA 12.8.93. The probe did **not** create a peer mapping, register
  symmetric memory, compile a device-API kernel, exchange payloads or measure
  bandwidth/latency.

NCCL's device-initiated API begins with 2.28 and its LSA path can use PCIe
peers with P2P connectivity. Therefore the existing Kaggle NCCL 2.25.1 stack
cannot compile/run an NCCL device-API transport as-is; it requires a pinned
newer NCCL build. P2P support
also makes a CUDA IPC peer-memory protocol a candidate, but the current
preflight does not prove its safe publication or performance. Neither option
is an implicit production fallback, and neither removes the current runtime's
CPU-sized NCCL submit/readback today.

## Upgraded NCCL LSA gate

The private notebook then installed `nvidia-nccl-cu12==2.29.7` into a
temporary directory without replacing the system NCCL. Version v2 confirmed
the packaged headers and library report 2.29.7 and a device-API kernel
compiles for `sm_75`. V3 exposed a probe-only linker mistake (system 2.25
selected via `-lnccl`); v4 linked the exact wheel library, and v5 corrected
the test's expected untouched word on rank 1.

V5 **completed** on two T4s. Both ranks reported `deviceApiSupport=1`, one
LSA team, and `LSA_ROUNDTRIP_PASS`. The test used two host threads/ranks,
`ncclMemAlloc`, symmetric window registration, `ncclDevCommCreate`, a CUDA
kernel with acquire/release LSA barriers, remote GPU writes, and exact D2H
verification on both ranks. The wheel stayed isolated under `/tmp` in the
notebook. Raw results: `test_results/transport-probe-t4-v5/`.

This proves a feasible device-driven transport primitive on Kaggle's T4 pair.
It does **not** yet prove variable device-count exchange, throughput, error
propagation, buffer lifetime, zero-payload epochs, 8-rank topology or complete
BFS correctness. The current production runtime still links system NCCL 2.25.1.

## Counted exchange and Kaggle host variation (2026-09-23)

The isolated counted LSA leaf in private notebook v6 passed on 2×T4 for
`(3,1)`, `(0,5)` and `(33,1)` with 32-row capacity. The over-capacity case
set a global fatal on both ranks without payload writes. This is leaf evidence,
not runtime integration.

An opt-in production C ABI candidate was committed as `3bbb416` with
`MGBFS_NCCL_LSA=ON`. Notebook v9 compiled its `nccl_transport.cpp` against
pinned NCCL 2.29.7; v10 compiled and linked a production C ABI harness.
However, v10 could not execute `mgbfs_nccl_lsa_init`: it returned 4 on both
ranks because `deviceApiSupport=0`. That host's `nvidia-smi topo -p2p p`
reported `NS` between the two physical T4s. V8 landed on a different 2×T4
host with P2P `OK`, `deviceApiSupport=1`, and passed all three isolated
counted cases. Thus “2×T4 Kaggle” does **not** guarantee LSA support. No
BFS runtime path uses the production C ABI yet. Raw results are in
`test_results/kaggle_transport_probe_v8/`, `_v9/`, and `_v10/`.

V13 subsequently landed on P2P `OK` T4s and the production C ABI harness
passed exact hash/state payload checks for `(3,1)` and `(0,5)`, plus a
no-payload global-fatal check for `(33,1)`. V11 exposed an incorrect expected
source offset in the harness; v13 has the corrected oracle. These results
still cover only a single two-rank exchange, not repeated epochs, 8 ranks,
full owner integration, sanitizers or performance. Raw evidence:
`test_results/kaggle_transport_probe_v13/`.

V15 validated the subsequent source-capacity fix at `ae22e29` on P2P `OK`
2×T4. The production C ABI compile/link and all four cases passed:
`(3,1)`, `(0,5)`, `(20,20)` and `(33,1)`. In the latter two, aggregate
source capacity 32 is exceeded and both ranks observe fatal with no payload
writes. Evidence: `test_results/kaggle_transport_probe_v15/`.

V16 ran the production C ABI harness under all four Compute Sanitizer tools
on P2P `OK` 2×T4. The four plain cases again passed. `racecheck` reported
zero hazards and `synccheck` zero errors. **The sanitizer gate is not clean**:
`memcheck` exited 86 with 26 CUDA API errors during NCCL initialization,
including `cudaErrorNoKernelImageForDevice` from NCCL kernel-attribute queries
and `cuMemCreate` not permitted; the harness still printed both rank PASS
lines. `initcheck` timed out after 300 seconds. These failures do not prove
an LSA payload memory bug, but also cannot be dismissed as harmless. V17
therefore runs a narrower kernel-filtered memcheck/initcheck probe with CUDA
API error reporting disabled; even if it passes, it is **not** equivalent to
a clean unfiltered end-to-end sanitizer gate. Raw v16 output:
`test_results/kaggle_transport_probe_v16/`.

V18 landed on P2P `OK` T4s. All four plain production cases passed again.
With Compute Sanitizer restricted to `lsa_publish_count` and `lsa_copy_exact`
and CUDA API error reporting disabled, `memcheck` finished with zero errors.
The similarly filtered `initcheck` still timed out after 600 seconds before
producing a rank result. Thus there is focused memory-access evidence for our
kernels, but **not** an initcheck pass or a clean unfiltered sanitizer gate.
Raw result: `test_results/kaggle_transport_probe_v18/`.

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
