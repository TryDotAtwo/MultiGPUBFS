# Device-driven BFS: bounded two-GPU research prototypes

Status: **experimental tests only**. The public CayleyPy API, `graph.bfs()`,
production `segmented24` and `permutation_bitmap` backends, persistent workers,
release artifacts and default transport are unchanged. None of these files is a
production engine. Explore has not been implemented; its earlier exact-BFS gate
is still open.

## Hardware and provenance

* Source base: `4cd57898f127fa2aad20510b98ce3262aaa8219b`; individual
  test-source and executable SHA-256 values are recorded in the evidence archive.
* Two physical Quadro RTX 6000 GPUs, SM75, NVLink 2; two independent GPU
  processes. Both directions reported peer access and native peer atomics.
* CUDA compiler 12.8; isolated `nvidia-nccl-cu12` 2.30.7 wheel SHA-256
  `8ce1b8213f61f2bfac132e6df890af6450b77cbd140c6ce4e98cb0c2d8e678c9`.
  The actually loaded `libnccl.so.2` SHA-256 was
  `2d19623c75a1dc0f4296fa9b9ff5cb7f38f18e42b10076636eb4e144e9010d07`.
  The tests did not replace the machine's system or Python NCCL library.
* NCCL reported device API support and one LSA team for both ranks. Source,
  executable hashes, GPU UUIDs, tool versions, logs and traces are retained in
  the separately checksummed experiment archive.

## What was checked

The ring tests exchanged 8192 ordered records each way with one and seven
slots. The two-process test also checked an empty sender and delayed receiver;
the producer really encountered backpressure. Release/acquire system-scope
control words are used only after the topology reports native peer atomics.

The standalone exact-layer tests keep work on GPUs between a cooperative
kernel launch and the final host wait. CPU code constructs *small test fixtures*
and independently runs FIFO BFS to compare sorted full `(layer,state)`
multisets, retaining multiplicity. Cases cover directed cycles, duplicate paths,
an empty rank, self traffic, delayed receipt, wraparound at one slot, normal
and reversed GPU mapping, and rank capacity exhaustion. An incomplete run is
compared only through the last proven complete depth. A duplicate arriving
after the permitted space fills is explicitly counted and accepted as a
duplicate without a false capacity error.

The permutation test keeps packed frontier rows and an exact rank bitmap on
the device, using the existing `BitmapPack4Action` and `bitmap_rank` helpers.
Fixtures include S5 and S6 adjacent transpositions, repeated symbols, arbitrary
root and signed symbols. The CPU enumerates the small domain only to construct
the independent test oracle; the GPU traversal uses no domain lookup table.

A separate serial hash-table prototype uses full packed-state equality and the
existing hash mixing/bucket helpers. Forced full-hash collisions and two
simultaneously used shard numbers per rank are counted, and every completed
layer is checked. This is **not** the existing concurrent `segmented24/warp`
insertion algorithm and is not evidence for its pending/deferred-claim path.

A further bitmap prototype assigns producer, consumer and coordinator roles to
different warps within one cooperative block per rank. Its first version had a
shared start-status read/write race detected by Compute Sanitizer Racecheck.
The initial read now uses immutable bootstrap data. The red log and binary are
preserved together with the corrected test. Racecheck only checks its supported
on-device shared-memory hazards; it does not prove inter-GPU memory ordering.

For the corrected parallel bitmap prototype, a CUDA Runtime Nsight Systems
trace captured 52 cooperative kernel intervals (26 per GPU). No CUDA Runtime
operation started within a kernel interval except the host wait for that kernel
to finish. This trace covers a standalone small prototype, not production or
all CUDA driver/internal calls. Eight repeated suites, reversed device mapping,
memcheck, synccheck and the corrected shared-memory racecheck passed.
Racecheck does not establish inter-GPU ordering or global-memory race freedom.

Compute Sanitizer memcheck/synccheck were run as reported in the archive.
For the NCCL-linked tests, `--report-api-errors no` suppresses API error
reporting, not memory or synchronization diagnostics. The retained earlier
unsuppressed log recorded NCCL fallback probes returning `cudaErrorNoKernelImageForDevice`
and `CUDA_ERROR_NOT_PERMITTED`; do not interpret the suppressed run as proof
that all NCCL API calls succeeded.

## Reproduction and boundaries

With CUDA and NCCL already available in isolated locations, run on an otherwise
idle, peer-atomic capable two-GPU node:

```bash
NCCL_ROOT=/path/to/nvidia/nccl MGBFS_TEST_ARCH=sm_75 \
  bash scripts/run_experimental_device_lsa_checks.sh /path/to/results
```

The machine must support cooperative launch, full-team NCCL LSA and the native
peer atomics checked by the tests. The probes deliberately use tiny fixed
bounds and one or a few warps; they are not a throughput benchmark or a
production admission policy. No eight-GPU device-path acceptance, B200 result,
S12/S13 performance comparison, multi-segment allocation, 2^32 boundary,
production persistent/API integration, lost-rank cleanup or automatic host
fallback is established here. No claims are made about asynchronous Explore.
