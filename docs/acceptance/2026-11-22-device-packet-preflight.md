# Device packet and topology checkpoint, 2026-11-22

This checkpoint adds a **preliminary** CUDA/NCCL topology query and bounded
SPSC packet-ring primitives. It does **not** connect them to production BFS,
CayleyPy, persistent workers, or an eight-rank scheduler. Preliminary topology
success alone is not sufficient to enable device execution: actual group
agreement, NCCL communicator/window setup and a bidirectional packet smoke are
still required. No performance claim is made here.

The ring has separate close and data metadata, sequence/generation checks,
and release/acquire publication. A receiver releases a slot only after the
caller has made its payload durable. The caller must satisfy the documented
ownership and buffer-lifetime requirements. The test BFS below is deliberately
a fixed-size experimental bitmap model, not the segmented production backend.

## Test environment

- Eight NVIDIA Tesla V100-SXM2-32GB, SM70, CUDA toolkit 12.8.93, driver API
  13000, CUDA runtime 12080, actual NCCL 2.30.7.
- Source base: 94e393f98325dbbacf737ebdea199a3ccc092093 plus the
  changes in this checkpoint. Native library built for SM70, Release,
  CMAKE_CUDA_RUNTIME_LIBRARY=Shared, MGBFS_NCCL_LSA=ON,
  MGBFS_EXPERIMENTAL_PACK4=ON, and BUILD_TESTING=ON.
- libmgbfs_cuda.so SHA-256:
  05d64833c1b75e227d0be9ff3e88a4a60c11e37b6b2c215709c69c72d64a1ee6.
  Pinned libnccl.so.2 SHA-256:
  2d19623c75a1dc0f4296fa9b9ff5cb7f38f18e42b10076636eb4e144e9010d07.
- Physical GPUs 0–3 form a directed all-pairs native-peer-atomic group.
  The full eight do not: in this allocation, device 0 to device 5 was one
  failing direction. The preliminary query reported the direction and a
  non-success result; it did not allocate communication windows.

A first baseline attempt was built for SM75 and failed on SM70 with
cudaErrorNoKernelImageForDevice. The library was rebuilt for the physical
SM70 GPUs. The existing host-driven segmented-distributed test then passed
on two GPUs. This validates only that tested legacy path, not the new device
scheduler.

## Checks

The CTest tests device-topology-cpu, device-topology-gpu,
device-packet-ring-unit, device-packet-ring-lsa and
device-packet-bfs-experimental passed on two physical GPUs. The four GPU
tests report **SKIP**, not PASS, if no GPU is visible. The non-LSA topology
build reports that LSA was not compiled.

Separate physical packet tests passed with two and four GPUs, normal and
reversed device order, one and two slots, empty directions, sparse and
all-pairs traffic, full queues, release-after-consume, separate close markers,
and multiple phases and job generations. The four-GPU all-pairs test checks
every directed peer path within that group. With a group containing GPUs
0, 1, 4 and 5, all four processes reported unsupported native peer atomics
before the packet exchange and the test returned its skip code 77.

The experimental packet BFS matched independent CPU layer **multisets** for
its bounded fixtures on two and four physical GPUs, including a full queue,
a duplicate while full, a resource stop and device-order reversal. The
four-GPU adjacent-transposition S8 fixture completed 40,320 states in 29
layers, including every state in the comparison. These observations are
specific to this test program and do not validate production segmented
storage, cancellation, persistent reuse, distributed failure handling,
8-GPU BFS, or S12/S13.

Single-GPU ring unit tests had zero Compute Sanitizer memcheck and synccheck
findings. A two-GPU packet test passed functionally, but unfiltered memcheck
reported 42 CUDA API errors from NCCL-internal feature/allocator probes
(error 209 and cuMemCreate error 800). Separate memcheck and synccheck runs
with --report-api-errors no each reported zero findings; this filtering
hides API errors and is not evidence that those 42 reports did not occur.
Neither run proves the correctness of cross-process atomics in general.

## Reproduction

Use the NCCL 2.30.7 headers and library matching the checksums above. On this
host, NCCL_ROOT=/workspace/nccl-2.30.7/nvidia/nccl:

    cmake -S cuda -B build-sm70 -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DBUILD_TESTING=ON \
      -DMGBFS_NCCL_LSA=ON -DMGBFS_EXPERIMENTAL_PACK4=ON \
      -DMGBFS_NCCL_ROOT="$NCCL_ROOT"
    cmake --build build-sm70
    CUDA_VISIBLE_DEVICES=0,1 ctest --test-dir build-sm70 --output-on-failure \
      -R 'device-topology|device-packet'

The direct 2/4-GPU builds, normal and reversed rank order, four-GPU S8
fixture, original-mailbox control, unsupported-topology test, logs, CUDA
toolchain receipts, and binaries are retained in the accompanying evidence
archive. The direct test scripts use short **test-only** process timeouts;
they do not add a timeout to production BFS.
