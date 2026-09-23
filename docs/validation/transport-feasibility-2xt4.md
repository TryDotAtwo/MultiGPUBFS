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

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
