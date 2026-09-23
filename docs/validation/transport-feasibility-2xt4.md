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
cannot compile/run an NCCL device-API transport as-is; a pinned newer NCCL
build plus an actual LSA setup/transfer test would be required. P2P support
also makes a CUDA IPC peer-memory protocol a candidate, but the current
preflight does not prove its safe publication or performance. Neither option
is an implicit production fallback, and neither removes the current runtime's
CPU-sized NCCL submit/readback today.

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
