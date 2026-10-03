# 32-batch CUDA Graph hardware gate

Explicit switch: `MGBFS_CUDA_GRAPH_BATCHES=32`; default remains 0 until the
hardware gates pass. Bootstrap includes this value so all ranks agree. The
initial implementation requires DENSE rank-owner device-count epochs with
NCCL_LSA. HASH_FIRST, HostSizedNccl and synchronous route tracing are rejected.

Each window captures up to 32 complete existing BFS batches across generation,
exchange and owner streams. The native helper explicitly forks auxiliary streams
into capture and joins them back before EndCapture. It updates the prior graph
executable when topology permits, otherwise instantiates a new executable.
Thus capture/update CPU costs still occur per window and must be measured.
This is not a static graph prebuilt for the entire traversal.

Archive copies are submitted before capture for only the next bounded window.
CPU archive workers observe ordinary D2H events, never captured events. Owner
retirement uses external archive-event waits inside capture. Archive credits
cover in-flight windows plus preparation of the next window; insufficient
credit configuration fails explicitly. Lookahead generation does not cross a
window boundary. Completion credits are recorded after graph launch, outside
capture; finalization remains at the existing whole-layer boundary.

Cancellation ends capture before communicator abort and resource teardown.
Graph stats are host counters collected after BFS: launches, actual full windows,
submitted batches, successful executable updates and executable rebuilds. A
configured window size by itself is not proof that a 32-batch graph executed.

CPU validation: seven reference-launch/resource-contract tests, ten event
lifecycle tests and four parent-cursor tests pass. Rust type checking with
cuda,library-owner enabled, including the new GPU oracle, passes on Windows.
None of these checks compile CUDA or prove execution on physical hardware.

Required physical gates (still open):

1. Build with pinned CUDA and explicit NCCL >=2.29 root. remote_build supports
   --nccl-lsa-root and records header/library SHA-256 and header version. Check
   P2P and actual ncclDevComm LSA topology before declaring support.
2. Build/run mgbfs-batch-graph-window-test. It tests two full 32-batch windows,
   a short tail, multi-stream dependencies, executable update/rebuild, external
   archive events, exact output and cancelled capture cleanup.
3. Run isolated lrx_multiset_two_rank_graph_windows_full_state_oracle with the
   switch set. Compare every layer's complete states for (7,1) and (8,4), both
   owner maps and prededup settings. Require observed full_windows >0 on both
   ranks. CPU model equality is not replaced by count equality.
4. Archived native CLI runs with graph off/on, same parameters, exact retained
   state comparison and whole-layer counts/times; VRAM sampled independently.
   Include zero/asymmetric payloads, wrap, final short windows and cancellation.
5. Matched larger speed trials plus memory admission with graph resources.
   There is no proven speed improvement or no-loss guarantee yet. Calibrate
   graph-resource headroom and automatic profile selection before enabling the
   switch in all new automatic sweeps.
6. Publish real shared Parquet cohorts on the GPU host; stream-check hashes and
   case manifests from HF and compare case-filtered rows with CPU/native data.

CUDA capture/update constraints checked against NVIDIA CUDA 12.9 documentation:
https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html
https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/structcudaGraphExecUpdateResultInfo.html

The active goal is not complete while these gates remain open.

2026-10-03 rental observation: isolated 2x RTX A4000 instance 53942300
confirmed bidirectional P2P OK and dispatched build of 3b29457. The build
process PID 393 was observed alive during dependency fetch. A subsequent
local observation gap of about 16.5 hours ended with watchdog ABSENT; an
independent API GET also confirmed instances=null. No final build summary or
GPU test output was recovered. Therefore this rental proves neither successful
CUDA compilation nor graph execution. The local watchdog cannot enforce a
hard deadline during host/network downtime; quoted cost is not billed cost.

remote_build --graph-smoke-test now builds and runs the native graph test
automatically as part of the remote job, with output in graph-test-run logs
and a PASS field only after the expected marker. This remains opt-in and
unverified on hardware; future test dispatch must enable it explicitly.
