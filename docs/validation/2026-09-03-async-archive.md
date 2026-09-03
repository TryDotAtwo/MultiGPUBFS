# Asynchronous archive submission

Implemented: a dedicated nonblocking archive CUDA stream, independent Hash GEMM
plan and batch-sized Hash128 scratch; no host wait per successful D2H submission.
Disk serialization/checksum/fsync already had a worker and remains there.

Each preallocated pinned slot owns a disabled-timing CUDA event. Producer records
the event after both state and hash copies, then moves the slot to the bounded
worker queue. The worker selects the creator's CUDA device, waits for that event,
reads the payload, writes it, and only then returns the slot. Error-path slot
destruction drains its recorded event before freeing pinned storage. Failed
enqueue before event publication explicitly drains the archive stream.

The current frontier has at most two contiguous physical extents. Two preallocated
extent events mark completion of all archive reads of each extent. Generation and
owner jobs may run concurrently with archival reads. Immediately before retiring
an extent, owner stream waits for its D2H event; the existing ordered ring-head
upload completes before any subsequent allocation can reuse those bytes. No disk
completion is involved in this dependency. Native destruction drains all three
streams before freeing plans and buffers. An extent event is reused only after
advance has discharged the previous depth's lease. Duplicate archive submission
of one depth is fatal.

Memory: additional device allocation is one queried hash plan of `batch` rows,
plus `16*batch` output bytes (not `moves*batch`). Both are included in preflight
and requested-allocation accounting. Pinned payload capacity is unchanged, with
one CUDA event per slot and two extent events. Runtime/driver event overhead is
not represented as a cudaMalloc request; device consumption is measured separately.
All buffers/events/plans are allocated before depth zero. Queues remain bounded;
exhaustion remains fatal, without disk backpressure or runtime resizing.

Limitations: archive still recomputes hashes and serializes hash/D2H on its own
stream. This is not a claim of full bandwidth overlap, or of increased disk
throughput. Submitting a whole frontier can need more simultaneous pinned slots
than synchronous submission. Disk durability remains a separate timer. Owner
metadata host synchronization is unchanged.

Tests: full archive state/hash oracle; blocked disk through complete BFS with
StateRing wrap (U4(4), F=1536, batch=31); padded U3(3) records; early-failure
cleanup. Target performance and sanitizer evidence must be recorded separately.

Local checks on RTX 3070 Laptop / CUDA 12.5: 9 native tests passed (including
large full-state m5..8), 65 default CPU tests, and 24 synthetic evidence-checker
tests. Compute Sanitizer memcheck of all three archive fixtures passed with
zero errors. Windows sandbox prevented initial checker process injection; the
same command worked with approved execution outside that sandbox.

Local Nsight Systems 2024.2.3 did not produce CUDA event data. An initial attempt
failed parsing a Unicode working directory; an ASCII-directory retry produced
an empty CUDA trace and diagnosed missing Windows Performance Toolkit. This is
not overlap evidence; no overlap percentage is claimed. Artifacts are in ignored
`build/async-archive-v1.nsys-rep` and `.sqlite`.

CUDA event record/wait semantics reviewed against NVIDIA Runtime API documentation:
[events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html),
[streams](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).

Private Kaggle native-runtime version 4 pins implementation
`5482f3bb9d20db5780bf6b5c915c4d93c8cd321c`. Its mandatory evidence check is:

```
python scripts/verify_native_runtime.py test_results/native-runtime-v4/native-runtime --source 5482f3bb9d20db5780bf6b5c915c4d93c8cd321c --require-async-archive
```
