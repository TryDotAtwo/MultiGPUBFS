# CUCO_INDEXED boundary

Source pin: NVIDIA/cuCollections `532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4`.
The official mapping-table example stores integer indices and compares the
original larger keys through a custom equality functor. Thus a 128-bit Hash128
need not be truncated to fit the T4-compatible atomic key representation.
The example explicitly is not a performance demonstration, and its illustrative
hasher reads only the first component. Do not copy that hash policy into BFS.

Source: https://github.com/NVIDIA/cuCollections/blob/532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4/examples/static_set/mapping_table_example.cu

Integration consequences (our inference, not an upstream performance claim):

- Equality must load and compare all four u32 words. Index equality alone is
  wrong, and fingerprint equality alone would change the collision contract.
- Every stored index must continue to address immutable keys. A candidate slot
  cannot be recycled while the table still references it.
- A bounded prototype can rebuild the index from committed storage before
  reusing transient candidate storage. Rebuild time and scratch must count in
  the comparison; this is not assumed to win against CUB or cuDF.
- Use an explicit invalid-index sentinel outside all valid rows; guard count
  bounds before insertion. Table capacity must be preallocated and checked.
- Concurrent insertion does not by itself establish minimum source-ordinal
  provenance. If the shared owner interface promises KEEP_FIRST, test and
  implement that selection rather than assuming insertion order.

Required next evidence: pinned dependency build on sm75; all-word-distinct keys,
duplicate candidates, committed overlap, transient-slot reuse and full-capacity
failure fixtures; then real owner comparison. No CUCO runtime result exists yet.

## Proposed persistent/transient split (not implemented or measured)

The first indexed design need not rebuild accepted membership after every batch.
Use two fixed-capacity sets per shard:

- Persistent set: indices into previous, current and accepted key planes. After
  StateRing credit and accepted-key copy, insert the new **accepted** indices,
  never candidate-slot indices. Accepted rows remain immutable for the depth.
- Transient set: indices into one candidate slot, cleared only after all result
  readers finish. It deduplicates the current batch independently of history.

Proposed index representation is u64: top two bits identify previous/current/
accepted/candidate storage, low 62 bits identify the row. Actual row bounds stay
at the ABI's signed-32-bit limit. UINT64_MAX is outside every valid index. All
four Hash128 words participate in equality and hashing; these index bits do not
replace the hash or change its collision contract.

Concurrent insertion may select an arbitrary representative. After insertion
has finished, find that representative and atomic-min the original candidate
**row** into an incoming-sized ordinal array. Select the minimum row and return
its supplied source index. Minimum source-index value is not a substitute for
KEEP_FIRST when callers provide non-monotonic provenance values.

Candidate probing reads the persistent set but cannot publish into it. Commit
copies the selected keys to stable accepted storage before inserting their
stable indices. Keep an explicit exclusive shard lease and stream ordering.
At depth finalization, drain and discard membership before overwriting history.
Capacity must cover previous+current+maximum accepted rows at the chosen load
factor, plus the independent incoming table, ordinal/select scratch and actual
rounded cuco storage extent. Use a fixed-RMM allocator adapter; the default
cuco CUDA allocator is not acceptable for this profile's physical-pool contract.

Pinned `static_set.cuh` exposes allocator and stream constructor parameters,
`insert_and_find_async` and device `ref` operations. These are API feasibility
observations, not evidence that the proposed adapter builds or performs well.
The earlier rebuild approach remains a simpler reference candidate; its cost
must be measured if implemented. The two-set option also has extra memory and
random-probe costs and is not presumed to beat the sort/merge baseline.

## Allocator detail verified at the same source pin

`include/cuco/detail/storage/bucket_storage.inl` calls
`allocator_.allocate(alloc_size, stream)` directly, not the ordinary one-argument
std allocator interface. The adapter must therefore support the cuco stream
argument and preserve that stream through deallocation. Storage reserves extra
alignment elements beyond `capacity()` (`(alignment - 1) / sizeof(value_type) + 1`);
counting only capacity times key width underestimates the allocation. Charge the
actual allocator request and retain the raw pointer for paired deallocation.
The constructor's requested capacity can also be rounded by the probing/storage
geometry; inspect the resulting capacity, not only the requested row count.
These are source-level requirements only; no allocator adapter has compiled yet.

The pinned cuco `utility/allocator.hpp` pairs `allocate(n, cuda::stream_ref)`
with `deallocate(pointer, n, cuda::stream_ref)`; its default implementation uses
cudaMallocAsync/cudaFreeAsync and must not be used for our fixed reservation.
RMM v26.04.00 source lives under `cpp/include/rmm/`, not `include/rmm/`.
Its pool's `device_async_resource_ref` uses stream-first calls
`allocate(stream, bytes)` and matching resource deallocation. The adapter must
convert element count to checked byte count, retain the selected resource
instead of resolving the current resource during destruction, and keep it alive
until the set and its queued operations are drained. This remains an uncompiled
adapter design, not a runtime compatibility result.

The matching RMM device-memory-resource deallocation signature was verified as
`deallocate(cuda_stream_view stream, void* pointer, size_t bytes, size_t alignment)`
(alignment defaults to CUDA_ALLOCATION_ALIGNMENT). Allocation has the matching
stream-first signature and default alignment. A conventional cuco allocator
must translate both calls, not forward the cuco argument list unchanged.
No `cl`, `g++` or `clang++` is currently available on PATH in the Windows host;
the adapter's real compile/run gate will use the pinned Kaggle SDK, not a claimed
local C++ build.

Follow-up: MSVC 14.44.35207 was found under Program Files (x86)/Microsoft Visual
Studio/2022/BuildTools (not PATH). The new dependency-free allocator adapter and
`tests/cuco_pool_allocator_contract.cpp` compiled and ran with MSVC C++20 after
the missing-header RED build. The CPU fixture checks stream translation, byte
counts, captured resource identity, allocator copy/deallocation, overflow before
allocation, and propagation of pool exhaustion without fallback. This does not
compile the actual cuco/RMM binding and does not validate CUDA device execution.

The cuco/RMM binding probe was added in `830c0e1`. Kaggle v37 reached nvcc but
failed at cuco's explicit extended-device-lambda requirement, before running any
GPU fixture. `72bae3b` adds `--expt-extended-lambda` only to the cuco probe target.
V38 pins `72bae3bc030d9e1fd597c7ab6171e70b030123d9`; its result is pending.
The fixture checks integer-set insertion, duplicate counts, clear/reuse, fixed
pool size and zero live pool allocations at teardown. Even a pass will not
certify the proposed indirect Hash128 owner or full BFS integration.

V38 progressed past the device-lambda requirement but failed compilation because
cuco rebinds the allocator to temporary element types. `12fcf4b` adds a converting
constructor preserving the original resource reference. The MSVC fixture was
extended to allocate/deallocate 13 bytes through a char rebind, failed before
the change, then compiled and passed. V39 pins
`12fcf4bfe37b7f630436f3cac8ea15fe28b59c08`; actual cuco/RMM GPU execution remains
pending. No fallback allocator was added.

V39 completed PASS with source `12fcf4bfe37b7f630436f3cac8ea15fe28b59c08`.
The cuco/RMM binding fixture compiled and passed independently on both physical
T4s, plain and all four sanitizers. Every sanitizer summary is zero, including
racecheck warnings. Each fixture kept its 64 MiB pool fixed, returned all live
allocations at teardown, and reported 1081 bytes peak suballocation. This is
not full device VRAM or an indexed-owner benchmark. Evidence is under
`test_results/library-owner-v39/library-owner/cuco-gpu*-*.log`.
Next required implementation remains indirect four-word Hash128 membership,
stable committed indices, deterministic survivor provenance and owner commit.
