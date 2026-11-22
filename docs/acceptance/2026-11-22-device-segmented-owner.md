# Device mailbox layout and packed bitmap owner checkpoint

This is an incremental experimental checkpoint after
486cf2d4a1e37ed0cb15b48b6f0590097d351dd5. Neither the resident owner
primitive nor the test scheduler is connected to public CayleyPy or the
production BFS runner. The ordinary BFS, persistent protocol and archive
semantics have not been replaced. The production device scheduler, hash/warp
backend, eight-GPU acceptance and Explore remain open.

The new mailbox planner uses checked 64-bit arithmetic, returns the exact
preallocated window requirement, checks the caller's byte budget, and keeps
data slots, payload and close records disjoint. Ring capacities and peer/lane
counts are runtime inputs. A 2^32 ring-count arithmetic case is virtual only;
no corresponding allocation was attempted.

The internal packed bitmap owner uses the existing exact permutation rank,
lossless symbol dictionary and owner mapping. A single scheduler lane owns a
shard transaction; independent shards may atomically update different bits in
the same visited word. A new state receives a row in a preallocated extent
before its bit becomes visible. Duplicate lookup happens before the resource
verdict; the caller retains a candidate on extent exhaustion. Any malformed
state, invalid descriptor or ownership race is fatal rather than a successful
resource stop. These invariants require the future production scheduler to
uphold the documented ownership, admission, publication and lifetime rules.

## Physical checks

Hardware: NVIDIA V100-SXM2-32GB, SM70; CUDA toolkit 12.8.93 and NCCL 2.30.7.
The rebuilt native library SHA-256 was
43369a4b2828ad4fb6dc2e1350fb3f580568578456e12b296edfbcdf40d0154f.
The full original CUDA API reports from the earlier NCCL sanitizer check
remain in the preceding checkpoint.

- Nine targeted CTests passed on two physical GPUs, including both the
  previous experimental packet BFS and the segmented-owner variant. No-GPU
  runs of the owner and VMM tests reported SKIP.
- One-GPU owner tests compared the complete accepted multiset of a
  repeated-symbol domain to an independent lexicographic enumeration. Two
  shard CTAs rendezvoused before accepting, used independent extents, and
  shared an atomic rank cap. The tests covered duplicates, a full first
  extent, retry after more extents, cap exhaustion and retry after raising the
  cap, and invalid inputs.
- A virtual-memory test wrote the legal row 2^32-1, confirmed that the next
  unique state remained unvisited when that extent filled, and accepted it
  in a second extent. The virtual reservation was 32 GiB; only one 2 MiB
  device page was mapped for the large extent. This is an address-boundary
  check, not a physical 2^32-row capacity result.
- The experimental multi-rank BFS ran with two and four GPUs in both device
  orders. Its CPU oracle compared full per-layer multisets for arbitrary
  roots, repeated symbols, a directed rotation and a duplicate-at-full
  fixture, including the last completed prefix on resource exhaustion.
  Its four-GPU S8 fixture completed all 40,320 states and 29 layers.
  The prototype uses one serialized owner per rank and splits a preallocated
  contiguous array into two extents. It does not prove the production segmented
  allocator, multiple GPU owner shards, multi-node support or failure recovery.
- The dynamic mailbox exchanged every directed pair on two and four GPUs
  in both orders, with two simultaneously active lanes, one or two slots,
  sparse/empty directions, full queues, epoch changes and payloads checked
  byte-for-byte. This node's eight GPUs are not all connected by native peer
  atomics; the device path has not passed an eight-GPU test here.

Compute Sanitizer memcheck, synccheck and racecheck reported zero findings
for the one-GPU owner fixture. Racecheck does not prove absence of global
memory races. Two-GPU dynamic-mailbox memcheck and synccheck also reported
zero findings with CUDA API error reporting disabled; this filtering does
not erase the NCCL-internal API reports in the previous checkpoint. The
serialized owner and high retry counts are correctness instrumentation, not
a measured production performance result.

## Commands

The exact direct-build scripts and logs are retained with the evidence
archive. With the configuration from the preceding checkpoint:

    cmake --build build-sm70 --target \
      mgbfs-device-mailbox-layout-cpu-test mgbfs-device-pack4-accept-test \
      mgbfs-device-pack4-boundary-test mgbfs-device-packet-ring-lsa-test \
      mgbfs-device-pack4-segmented-bfs-test
    CUDA_VISIBLE_DEVICES=0,1 ctest --test-dir build-sm70 --output-on-failure \
      -R '^device-(mailbox-layout|pack4|packet-ring-lsa)'

The fixture's bounded process watchdogs are tests only. They do not impose
a production BFS deadline.
