# Dynamic packet mailbox and segmented bitmap storage checkpoint

This checkpoint is an experimental fixture, not a production BFS backend or
a CayleyPy feature. It follows 54aa009f5644abe035503190402d5717c5305653.
The existing BFS, persistent workers, public results and archive format are
unchanged. Device exact BFS, its host admission and error handling, an exact
hash owner, production multi-GPU acceptance and Explore remain open.

The resident fixture now uses the checked mailbox planner in its BFS kernel,
including packet, payload and close-record offsets, instead of a fixed embedded
mailbox. One registered window per rank is allocated for the admitted maximum,
and a checked immutable descriptor for the current slot count determines the
actual layout. Two existing fixed-mailbox fixtures remain available as controls.

The new adapter maps the existing MgbfsSegment state pointer, per-bank capacity
and bank count plus a separate count buffer into the resident bitmap owner's
extent descriptor. It rejects invalid shapes without changing its output.
The fixture allocates two separate device segments, each with two physical
banks, and rotates those banks between completed layers. Its producers read
the current frontier from both segments, and the owner writes the next layer
into the separately allocated segments. The adapter does not validate
allocation lengths, leases, lifetime, rank-wide agreement or budget: admission
and the future scheduler must establish these. The tested descriptor shapes
match the library, but this fixture does not call its production allocator.

## Physical coverage

Hardware: V100-SXM2-32GB, SM70; CUDA toolkit 12.8.93, NCCL 2.30.7.
All reported runs use actual GPUs. Four V100s in one peer-atomic island are
supported; peer atomics were not available for all eight on this instance.

- Two-GPU CTest: the new two-segment fixture, dynamic-mailbox fixture and both
  prior controls passed. Full per-layer multisets were checked against the
  independent CPU BFS for multiple roots, repeated values, a directed graph,
  duplicates at capacity, one/seven mailbox slots, and exact completed prefixes
  for an intentionally insufficient resource cap.
- The independent-segment fixture ran the same five graph families on four
  GPUs in both physical orders: 104 rank/case results passed in each order.
  In each order, 46 rank/cases independently required and observed use of a
  second segment. Assertions also require both banks when the CPU oracle
  contains accepted states on both parities.
- Four-GPU S8 adjacent-transposition fixture completed 40,320 states, all
  29 layers, with full per-layer per-rank multiset comparison; this is a
  correctness test, not a performance benchmark.
- The host-side shape adapter accepts the legal configured capacity 2^32 and
  rejects 2^32+1 without mutation. This is an arithmetic/descriptor test,
  not an allocation of 2^32 rows.

Two-GPU Compute Sanitizer memcheck and synccheck reported zero findings with
CUDA API error reporting disabled to filter NCCL-internal feature probes. These filtered runs
do not erase the API errors recorded without filtering in the preceding
checkpoint; they do not establish global-race freedom. Exact commands and raw
logs are retained in the evidence bundle.

The fixture has one serialized owner per rank, predetermined allocations and
test-only finite spin protection. It does not prove production allocation or
dynamic growth, multi-shard scheduling, generic graphs, rank failure recovery,
unbounded production progress, full archiving, eight-GPU device execution or
performance. None of its test-only deadlines apply to production calculations.

## Reproduction

With the previously configured build and pinned NCCL:

    cmake --build build-sm70 --target mgbfs-device-pack4-real-segments-test
    CUDA_VISIBLE_DEVICES=0,1 ctest --test-dir build-sm70 --output-on-failure \
      -R '^device-pack4-real-segments-experimental$'

For the 4-GPU fixture, compile the same source with MGBFS_EXPERIMENTAL_WORLD=4,
MGBFS_PACKET_MAILBOX=1, MGBFS_SEGMENTED_PACK4_OWNER=1,
MGBFS_DYNAMIC_PACKET_MAILBOX=1, MGBFS_DEVICE_SEGMENT_STORAGE=1, link
cuda/device_mailbox_layout.cpp and pinned NCCL, and set the visible four
mutually peer-atomic GPUs. MGBFS_DEVICE_TEST_SWAP=1 reverses device ordering.
The larger fixture uses MGBFS_EXPERIMENTAL_MAX_N=40320 and
MGBFS_DEVICE_TEST_EXTENDED=1. Exact shell scripts and logs are in the bundle.

The Nsight Systems collection produced no CUDA data for the fork/exec fixture;
it is not evidence about host waits or the resident path. No further trace of
this fixture was attempted. The four-GPU binary predates correction of two
error-message newline literals; the exercised kernel logic is unchanged.
