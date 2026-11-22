# Device-driven exact BFS and reachability: acceptance plan

Status: design and evidence collection. No device-driven segmented backend is
claimed as implemented or accepted by this document. This extends the active
segmented24/permutation_bitmap objective and preserves the pack4/2^32 work.
Keep graph.bfs(), persistent workers, archives, exact layers and farthest-state
semantics, existing strategies, configurable per-segment row capacity, and
checked wide aggregate bookkeeping.

## Baseline and applicability

The integrated source represented by experimental branch parent
06d132a81aca3d66bd06d0c50e9bbfcc9ae3436e is the implementation base, with
released baseline 72a3e2a0b0521985229823cc6c4c959ec22d81fd. The former is not
the dev44 wheel. Current segmented route finishes each chunk with a GPU-to-host
error/count copy and a stream synchronization, followed by a host-driven vote;
exact exchange further copies per-peer counts to the host before payload
submission. Existing A/B queues overlap some work but do not remove those waits.
Existing nccl_transport.cpp has an LSA path for another native engine. Its
per-exchange host launches and whole-team rendezvous are not evidence that the
segmented path is device-driven. Reuse only interfaces whose semantics and
lifetime are proved appropriate.

The measured eight-RTX-3060 host reported NS for every off-diagonal
nvidia-smi topo -p2p p and no native remote atomic support. It cannot accept
the proposed all-eight LSA peer path. Host-driven NCCL remains a correct
fallback, separately identified and measured. Other device transports and
other topologies were not tested. The NVIDIA NCCL device API documentation
requires LSA-accessible GPU peers and exposes requirements on communicator and
symmetric windows: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html

## Gate 1: exact BFS

1. Record baseline startup/admission, first/warm transport, BFS, I/O, CPU
   submits/waits, queue waves, effective geometry and topology. Bind every
   binary/library SHA and result to the same artifact. Measure link traffic
   independently from logical message and padding counts.
2. Before any device work, agree one communicator-wide device-transport
   capability, native/runtime identity, rank/device order, exact owner map,
   codec, protocol version and allocation geometry. Probe the actual
   peer mapping and transfer/ack; if remote atomics are used, verify the
   needed native operation and memory ordering. Any mismatch or inability to
   initialize is resolved uniformly before the data plane. Preserve an
   explicitly reported host-driven fallback; never label proxy-based work as
   fully device-driven.
3. Use bounded preallocated per-peer ring slots and long-lived job-owned
   sequence/epoch metadata. A producer first writes payload, then publishes
   with the actual transport's documented inter-device release/acquire
   guarantee. The receiver acknowledges only once the exact result and any
   retryable work have durable ownership. Credits prevent overwrite. An
   unaccepted row remains retryable; a duplicate at capacity is not hard-full.
   Size queues, storage, scratch, reserve, and sequences with checked wide
   arithmetic; a per-segment u32 index is not a global GPU capacity.
4. Prove progress with all queues full: reserve execution resources for the
   receiver and never let waiting producer CTAs occupy all resident slots
   needed by consumers. Design cancellation and fatal-rank supervisor outside
   the possibly poisoned communicator. Test lease lifetime, wraparound,
   epoch mismatch, rank failure, and explicit clean next job.
5. For every source/destination pair, including self and empty peers, publish
   the layer-end sequence only after all generators and tails. Drain through
   every end marker, publish all accepted next states, then perform a GPU-side
   layer-completion/count agreement. Only then expose depth d+1. Preserve
   full history for directed inputs and only use rolling history where its
   graph contract has been established. On resource exhaustion publish only
   the exact last completed prefix, never the partially prepared next layer.
6. Gate with independent complete layer multisets, diameter and final layer
   on small dense, repeated, directed, collision and two-shard fixtures.
   Include empty rank, self traffic, tiny queues, delayed source, actual
   overflow/replay, duplicate at full, true hard-full, ring wrap, cross-rank
   duplicate, persistent graph switch and failure. Record an actual supported
   device-path timeline proving no per-chunk D2H counters, host stream waits,
   or CPU submissions. Model tests are useful but do not substitute for GPU.

## Gate 2: separate exact reachability

Start only after gate 1 succeeds. Reachability is a distinct API/CLI and
result algorithm, not graph.bfs(). Exact canonical ownership, full visited
membership, exact collision equality and bounded reliable transport remain
mandatory. Do not report distance, layers, diameter or farthest states.
Report a found target separately from a completed reachable set; return
NOT_FOUND only after verified exhaustion, UNKNOWN after an incomplete search.

At a termination check, quiesce generation, drain every published packet
through its last sequence, and reduce all durable pending work on GPUs before
declaring complete. These rounds are not distance layers. Validate entire
reachable multisets against CPU BFS including cycles, diamonds, colliding
states, delayed packets, bounded buffers and directed paths of unequal length.
See https://www.frontiersin.org/journals/high-performance-computing/articles/10.3389/fhpcp.2024.1285349/full

## Eight-GPU comparison

On the same physical topology compare the old host exact path, supported
device exact path and Explore with equal graph, start, VRAM cap, output and
work definition. First keep geometry comparable, then separately tune each
mode. Cover full S12 and full S13 with Mahonian layer counts and exact final
state checks, or a comparably large complete repeated-permutation graph.
Compare incomplete runs only over a common proven prefix and never as a
whole-graph speedup. Record all times, per-rank allocations/high-water,
NVML sampling method, generated edges, accepted states, logical/physical
transfer, CPU waits, retries, graph and artifact checksums. A host-fallback
run or a virtual boundary is not an eight-GPU device-path acceptance.
B200/Blackwell capacity and speed require actual hardware.

## Initial queue primitive checkpoint, not BFS

An unintegrated experimental single-producer/single-consumer CUDA ring
implements nonblocking operations, per-slot epochs and 64-bit sequence
transitions, explicit acknowledgement after acceptance, and an overflow guard.
It uses system-scoped release/acquire atomics; this does not establish the
capabilities or visibility of any remote mapping. No production API, native
dispatch, exact owner, BFS layer agreement, or Explore uses this code.

On one RTX 3060, two blocks of one test kernel transferred 8192 ordered records
through rings of one and seven slots, including repeated reuse. A separate
single-thread boundary fixture checked backpressure, repeated read before
acknowledgement, epoch mismatch, and counter exhaustion. Boundary-only
Compute Sanitizer memcheck and synccheck reported zero errors. These results
do not establish inter-GPU operation, GPU scheduling fairness, fault recovery,
or whole-program correctness.

An earlier two-stream producer/consumer version timed out with both 8192 and
64 records; its cause is unresolved, and the first test command accidentally
masked timeout in a shell pipeline. A subsequent run used pipefail. The
single-kernel success is a different scheduling configuration, not a fix or
proof that independent kernels make progress. Preserve this failure when
designing a no-deadlock scheduler.
