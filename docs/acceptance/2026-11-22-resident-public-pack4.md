# Experimental resident pack4 bitmap through CayleyPy

This is an **opt-in development path**, not the default or a released native
wheel. execution_engine="device" requires a matching native executable and
CUDA library with device_resident_pack4_v1; old bundles reject it. The
supported initial configuration is permutation bitmap, pack4 encoding,
atomic, balanced, no autotune and no time deadline. Unsupported choices or
topologies fail before the resident BFS begins. The ordinary host engine and
the hash backend remain available separately.

The resident path starts one CUDA kernel per rank for the traversal. Its
bounded per-peer SPSC mailboxes use a registered NCCL LSA window and system
scope atomics. Before use, ranks check the real device kernel, peer atomic
capabilities and LSA mapping, and exchange data and credits through the
actual window. A layer is reported only after all outgoing directions close
and incoming packets drain. The storage extents are preallocated; their
per-shard leases and byte ranges are checked before launch. A failed resident
job poisons the worker generation and the supervisor retires it rather than
reusing potentially live buffers.

The layer log is a preallocated GPU resource. Its default capacity is the
lesser of the domain and the weakest rank's available bytes divided by 8192;
it is reported per rank and can be set with MGBFS_RESIDENT_LAYER_CAPACITY.
If that capacity is exhausted, the exact completed prefix is reported as
INCOMPLETE, with reason RESIDENT_RESOURCE_LAYER_LOG_CAPACITY.
This is an experimental limit, not a claim of unlimited traversal depth.
Per-layer time and peak live rows are not measured by this path.

## Checked configurations

The following tests used Tesla V100-SXM2-32GB, SM70, CUDA 12.8.93,
NCCL 2.30.7, CayleyPy 0.2.0 and PyTorch 2.6.0+cpu. Python uses CPU tensors to
construct CayleyPy graphs; traversal ran on physical GPUs. The installed
native binary and library were explicitly supplied from the development
build. For the final rank-reuse revision their SHA-256 values were:

- executable: 09065a90031c8cc15250435d0f24716335d5312bf26b87c41cea32b17901b084
- CUDA library: 7b94710bc928afb5adb4cb7c45963f1df5950453a48c77559d6bb93c9cd198f2

Four-GPU run_graph tests compared the **complete multiset of every completed
layer** against an independent CPU BFS on S4, directed C6 and a graph with
repeated symbols. They checked a tight per-rank cap with a matching complete
run and one-row-smaller incomplete run, a duplicate at full cap, two active
shards, and reused the same persistent worker PIDs. The public
CayleyGraph.bfs() integration passed on 2 and 4 GPUs with S4, directed C6,
the repeated-symbol case and LRX3; all layer sizes and the complete final
layer matched the oracle. It also checked a resource stop and a duplicate at
full cap. The public API test does not expose full intermediate states; that
stronger multiset comparison is the run_graph test described above.

On two GPUs a rank was killed during resident BFS. The group reported an
error, did not publish success, saved failure evidence, and rejected reuse
of the poisoned persistent session. An explicitly created new session
completed S4. This fault test used the same executable with the preceding
CUDA library (0da274cbef68e6d3ed142fcb4d4a0fb9fb11cd86c587a7b4efb5ccfc49d5e2da);
the later change only avoids recomputing a receiver-validated rank.
A two-GPU device → host → device sequence also retained healthy worker PIDs.

The eight devices on this particular node do **not** form the required
all-pairs native-atomic topology (for example, device 0 to 5). An explicit
eight-GPU request failed with RESIDENT_TOPOLOGY_UNSUPPORTED before BFS.
This is not an eight-GPU acceptance test. B200, SM100/SM120, large-layer
throughput and capacity have not been tested for this path. Virtual arithmetic
checks for multiple 2^32-row extents are not physical allocations.

## Small performance observation

One S7 Coxeter run on four V100, identical graph, shards=2, batch=128,
segment_rows=4096, and pack4, completed all 5,040 states. The host path took
0.0446 s BFS and the resident path with rank reuse took 0.2936 s BFS. Full
launch times were 16.0 s and 3.10 s respectively and included different
startup/preflight work; they are not a steady-state speed comparison.

A separate alternating small-graph A/B, three observations each with the same
executable and different CUDA libraries, gave resident BFS times:

| Receiver rank | BFS times, seconds |
| --- | --- |
| Recomputed | 0.4555, 0.4122, 0.4065 |
| Reused after validating payload | 0.2836, 0.2780, 0.2799 |

This confirms an improvement only for this small setup; the one-receiver
resident implementation remains slower than the existing host engine on S7.
It says nothing about large graph throughput or eight GPUs.

## Reproduction on a compatible development build

Set MGBFS_EXECUTABLE and LD_LIBRARY_PATH to the matching executable and
library. RESIDENT_TEST_WORLD selects the number of physically available,
mutually compatible GPUs:

~~~sh
export MGBFS_EXECUTABLE=/path/to/mgbfs
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
export MGBFS_SEGMENT_ROWS=2 MGBFS_SEGMENT_TEST_BATCH=1
RESIDENT_TEST_WORLD=4 python tests/device_resident_layers_gpu.py
unset MGBFS_SEGMENT_ROWS MGBFS_SEGMENT_TEST_BATCH
RESIDENT_TEST_WORLD=4 python tests/device_resident_cayleypy_gpu.py
~~~

tests/device_resident_fault_gpu.py deliberately kills one worker and is
intended for an isolated test node. The A/B helper
tests/device_resident_s7_ab_gpu.py requires explicit old/new library
directories and SHA-256 values. These opt-in tests are not ordinary CI jobs.
