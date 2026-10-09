# General exact sorted-history integration plan

Status: architectural plan and primitive gates; no production sorted-history backend is accepted yet. The active goal remains the unified exact graph launch with automatic admission and measured choice, not adding a manually forced benchmark mode. User-approved contracts: flat preallocated SoA, separate hash/index and state planes, concurrent shard A/B leases, no CPU production state work, large transactions, compact committed stop snapshots. Supported actions are CayleyPy permutation and integer matrix definitions; byte and signed-int64 codecs preserve the existing exact action semantics.

## Current source

GenericDistributedBfs already launches independent owner streams and overlaps generation into alternate source banks with preceding owner work. GenericSortCache sorts incoming origins but acceptance still uses exact hash-table membership. Rolling hash retirement/reseed is amortized, not rebuilt every layer. Legacy SHARD_AB SORT_MERGE repeatedly merges portions with growing shard history; it must not be copied as the general implementation. The repaired exact LRX owner remains an eligible measured alternative, not an arbitrary-graph proof.

## Final target

1. Immutable sorted runs hold full64-bit hash and canonical VRAM row reference in separate aligned arrays. A hash is an ordering/bucket key only. Equal-hash membership always compares every canonical coordinate; uint64 max and all legal signed state values remain valid. Invalid padding is expressed by validity/count/reference, never by excluding a hash domain.
2. Accepted future rows are stored once in flat SoA arena. Incoming full/parent-origin/packed24 actions retain their existing immutable lease and GPU value() semantics. Sort and dedup operate on indices. No new full-child plane, CPU generation, CPU dedup or CPU per-row checks are introduced.
3. Previous/current sorted runs persist across layers for proved inverse-closed graphs. Directed graphs retain every committed run until end of the graph; no false three-bank history optimization. References may be recycled only after all owner events and publication readers retire. On INCOMPLETE, only the partial future is discarded; previous/current remain canonical for compact snapshots.
4. Future accepted runs form a binary hierarchy. Merge only comparable capacity classes, with GPU merge-path and scans. Never merge every small portion with all accepted history. GPU-resident counts and valid references handle padding and empty runs. The last layer compaction publishes a committed immutable view atomically after every shard finishes.
5. Runs and allocator descriptors use one preallocated flat arena and bounded size-class bitmaps; no per-state pointers or host allocations in the hot path. Each merge needs an admitted destination credit while its two input runs are leased. A/B producer ownership is released only after exact dedup/materialization and merge completion. If credits are exhausted, apply bounded GPU stream/event backpressure; if the admitted state capacity is exhausted, resource stop without accepting a partial future.
6. All shard owners have independent streams/workspaces. Carry merges within one shard are true dependencies; other shards and alternate generation continue. Do not reintroduce global cudaDeviceSynchronize or CPU per-portion count reads. Existing collective safety votes remain until a separately accepted device transport replaces them.
7. Shared capacity cannot assume uniform hashes. One busy shard must be able to consume the admitted shared pool. Forced collisions, skew, tiny/empty queues and uneven ranks are correctness tests. Allocator peak (both leased inputs plus destination, padded capacities, all live ranks/shards/classes and workspace) must be derived and proven before planner admission enables this path.

## Implementation and acceptance sequence

A. Exact sorted-run membership and validation primitive for byte/int64, arbitrary state width, forced hash collisions, valid max hash, invalid references/counts. Independent GPU/CPU oracle on both allowed cards. This alone is not runtime integration or speed evidence.
B. Flat run allocator and publication generation contracts; synthetic concurrent skew/credit reuse/abort tests, byte-exact peak oracle and native allocation query. No runtime planner advertisement before these pass.
C. GPU incoming sort/scan/exact tie dedup + merge-path carry over hash/index planes. Parent-origin validity and full-state equality under collisions; equal-class merge bound counters, A/B lease proof. Avoid quadratic tie scans: collided groups need exact-coordinate ordering or collision-safe substructure.
D. Integrate permutation/matrix, directed/rolling and full/parent/packed paths in GenericDistributedBfs. Allocation planner and serialized plan include history_algorithm and actual workspace. Unsupported old binaries must reject explicit sorted requests before launch; AUTO may retain the accepted hash backend.
E. Add bounded history_algorithm profiles to local/external autotune, compare the same completed prefix and statuses, keep conservative5percent threshold/cache identity. Complete CPU layer/terminal oracles precede performance. Near-card matched cases, matrix and wide workloads, resource/deadline/SIGTERM, independent external ranks and clean immutable wheel/HF readback gates. Existing unchanged large benchmarks are reused where applicable.
F. Completion audit against the full objective and single GitHub branch. Physical multi-host/4/8/128, Blackwell and successful LSA remain explicitly unverified without authorized hardware. Do not label the goal complete because one primitive, package or small graph passes.

The plan deliberately separates required data dependencies from accidental waits and optional algorithm selection. A sorted-history candidate must be measured; adding it does not establish that it beats hash on all graphs.

## Foundation acceptance, 2026-10-09

Standalone GPU primitive passed on both RTX3060 devices against an independent CPU full-coordinate oracle: byte/int64 states, widths 2/17/25/129, full/truncated/constant-zero/constant-all-ones hashes, mutated misses, empty view, invalid row, overcapacity and unsorted publication guards. This verifies lookup and publication validation only. The production BFS still uses its prior owner backend. Run allocator, exact sort/dedup/merge, native admission and runtime integration remain unfinished. No throughput claim is made for this foundation.

## Shared run-pool foundation acceptance

The new unintegrated pool has flat regions, one 64-page bitmap per region, aligned power-of-two run credits, and a flat descriptor array. All owners share the entire pool. Allocation contention returns retryable pressure, not proof that admitted state capacity was exceeded. The caller must implement bounded stream/event backpressure. Publication claims the generation before writing metadata, preventing stale publication from corrupting a recycled descriptor. An owner retiring a published run blocks new readers; existing reader credits retain the allocation until the final release. A sole unpublished owner can abort. Generation wrap is rejected.

Host/device shape arithmetic includes hash/ref planes, region bitmaps and descriptors and rejects overflow of the uint32 row domain. It does not include the canonical state arena, sort workspace or communication buffers; native admission must add those before runtime enablement. A single run is limited to 64 pages. Larger history must retain multiple admitted runs or use a separately proven higher-level layout. This primitive does not justify advertising the whole sorted backend or a final memory plan.

Two RTX3060 passed 40 rounds of 512 competing allocations across seven size classes; an independent host bitmap oracle proves no live overlap, valid alignment and complete release. Another test holds 512 reader leases while the owner retires, then releases them concurrently. Stale release/publication, overcapacity, empty/invalid shape and byte-exact storage-query oracles passed. Compute Sanitizer memcheck reported zero errors and zero leaked allocations on both cards. This does not prove absence of all global-memory races or future integration/event-order defects. The first test failure was an invalid atomic error counter in test-local CUDA memory; the raw failed build/test evidence is retained. After changing test error storage to shared memory, the changed test passed.

## Merge-path and exact-order lookup foundation acceptance

A tiled stable merge-path combines two admitted immutable runs ordered by full64-bit hash and then all canonical state coordinates. Blocks locate their input tile, lanes locate their position within that tile, and write adjacent hash/ref positions. Only hash/ref planes are merged; canonical states stay in the SoA arena. An adjacent full-coordinate comparison flags distinct states; a caller-owned fixed-capacity CUB exclusive scan and separate scatter publish the compact run count on GPU. Output scan/scatter storage cannot alias live input planes. Device view counts are bounded by input credits and destination capacity before reading input keys. Zero capacity/null empty views are accepted.

Exact-coordinate ordering also enables binary membership under forced hash collisions; the old hash-only ordered lookup remains available for views that lack exact coordinate ordering. The new exact-sorted lookup requires the exact-order publication gate. No hash value is reserved or treated as state identity.

The CUB workspace query includes temporary scan storage, merge scratch planes, flags/prefix and device controls with256-byte plane alignment. It excludes pool-held final output credits, canonical state planes, incoming sort workspace and communication memory, which still require combined native admission before production enablement.

Two RTX3060 passed256 merge/oracle fixtures across byte/int64, widths2/17/25/129, full/truncated/constant-zero/constant-max hashes, empty/one-sided/uneven/boundary tiles, exact duplicates in different rows, stable output identities, unique output sets, invalid rows, reversed exact order and exhausted input/output credits. Zero-capacity shape and null-plane gates passed on both devices. Updated lookup tests compare both algorithms against independent full-coordinate CPU answers. Compute Sanitizer memcheck on the merge/scan/scatter fixtures reported zero errors and zero leaked allocations. These are primitive gates: incoming sorting, carry hierarchy, parent-origin/materialization, native planner and BFS integration remain unfinished, and no production speedup is claimed.

## Index-only incoming exact-sort foundation and origin guard

The incoming sort moves only cached full64-bit hashes and logical origin indices. Exact coordinate ties use the leased action.value() interface, then the physical origin is published through action.child(). This distinction matters for distributed shard queues and parent-origin replay. Invalid/padding origins are distinct from all legal hash values, including all ones. A deterministic origin tie-break selects a representative without redefining state equality. Unique flags and a GPU scan/scatter retain a device count. CUB merge-sort and scan share the larger temporary allocation; the shape query includes both hash planes, all five4-byte metadata planes and256-byte alignment. No full-child sort plane or CPU production state processing is introduced.

320 synthetic byte/int64 fixtures passed on both RTX3060 devices across widths2/17/25/129, zero/partial/all-pad queues, full/truncated/all-equal/all-ones hashes, mapped physical origins and malformed action rejection. Memcheck reports zero errors and leaks. Real IncomingAllAction and ParentOriginAction templates were extracted verbatim into shared headers. Before the added bounds guard,12 real action fixtures passed: full byte/int64, packed24, parent-origin permutations including wide int64, and modular parent-origin matrices, on each GPU. The current source is being recompiled for the same real-action acceptance after the guard; that final status is recorded separately and must be checked before claiming full acceptance.

The audit exposed a separate existing bug: an ordered origin was used to index queue counts without an origin-limit or expected-shard check. IncomingAllAction.valid now rejects an out-of-range physical origin or a foreign owner queue before dereferencing counts. The inherited parent-origin validation benefits from the same guard. Fresh focused tests on both GPUs and memcheck passed out-of-range/max origins, wrong-shard origins and overcapacity queue counts. The shared library rebuilt with this guard. This is a production safety fix, while the new sorting backend remains unintegrated.

The general run hierarchy, sorted-history admission, incoming-to-canonical materialization, owner A/B integration, AUTO algorithm profiles and end-to-end matched BFS timing remain unfinished. Primitive success is not evidence of a production speedup or completion of the active goal.

## Shared multi-region credits and tier metadata checkpoint

The shared run pool now admits power-of-two spans across bitmap regions, with rollback of partially claimed spans. Sole unpublished writers can trim surplus credits after exact deduplication without copying payload. Equal-class carry tickets reserve output before acquiring input readers; abort and invalid count preserve both input owners, while successful commit publishes output before releasing inputs.

The changed pool tests passed on both RTX3060 devices across eleven classes and whole-region contention, with zero memcheck errors/leaks. Tier metadata tests passed on both devices for normal carry, abort, invalid output counts, existing error, empty roots, and allocation pressure/retry, also with zero memcheck errors/leaks. These tests do not execute payload merges through the hierarchy and do not establish full BFS integration or speedup. Fresh real incoming/parent action tests after the bounds guard passed all twelve fixtures on both devices; their memcheck reports zero errors/leaks.

Remaining: focused multi-region trim/reuse acceptance, payload carry integration, canonical state leases, peak capacity/reserve admission, owner-stream scheduling, production runtime/AUTO integration and matched end-to-end BFS verification.

## Payload carry acceptance

Thirty-six matched GPU/independent CPU fixtures passed across both RTX3060 cards, byte/int64 canonical SoA rows, widths2/25/129 and constant-all-ones/truncated/full hashes. Five overlapping sorted roots are merged and deduplicated through allocated tickets, committed, trimmed and retained across tiers. Sorted unique order inside each run, exact union across retained runs, hash/row association, ownership cleanup and zero final occupied credits are checked. Focused multi-region-to-single trim and immediate tail credit reuse also passed, including rejection of trim after publication. Memcheck reports zero errors/leaks.

The harness deliberately synchronizes and reads small carry metadata between operations to inspect the result; this is test orchestration and is not an accepted production scheduling design. CPU-free carry scheduling, canonical row lifetime, history queries and native/AUTO integration remain required.

## Device-controlled carry graph

A reusable owner-stream conditional WHILE graph prepares tickets on GPU, enables exactly the matching size-class IF body, performs tiled merge, exact unique scan/scatter and commits the device count. No intermediate host count/ticket read selects or advances a carry. Inactive classes do not scan their payload; each active class uses its admitted capacity. This is a component, not production BFS integration.

Thirty-six byte/int64 fixtures on both RTX3060 cards passed the independent CPU union/order oracle across widths2/25/129, full/truncated/all-ones hash modes and repeated overlapping roots; memcheck reports zero errors/leaks. The cold shape query includes aligned shared workspace planes, control views/counts, conditional handles and the maximum CUB scan workspace queried over every admitted class. An undersized scan allocation is rejected before graph construction. CUDA12.8/driver570 conditional-loop capability passed repeated launches on both cards. Other runtime/driver versions remain unverified.

Remaining requirements include concurrent owner-stream pressure acceptance, actual canonical row lifetime integration, sorted history filtering, complete memory admission and public native/AUTO integration, matched full BFS correctness and timing. No end-to-end speedup is established.

## Immutable history reader leases

Owner-stream snapshot acquisition pins all occupied tier runs before publishing a compact array of immutable views. Failed acquisition rolls back all reader leases. Searches use exact hash/coordinate binary lookup over only the occupied runs. Retiring hierarchy owners preserves credits while a snapshot reader is live; reader release frees the retired credits. The caller must independently hold canonical state planes throughout snapshot use: run credits alone do not protect state rows.

Thirty-six two-device GPU fixtures combine CPU-free carry graphs, exact full/truncated/all-ones hash runs, byte/int64 widths2/25/129, snapshot acquisition and removal of every owner before searching all4096 candidate rows. Independent CPU membership and exact union/order agree; occupied credits persist until snapshot release and are zero afterwards. This remains component acceptance; production row leases, history filtering/materialization, full planner/runtime/AUTO and end-to-end performance remain incomplete.

## Native integration contract from the current runtime

Source audit: crates/mgbfs-runtime/src/generic_distributed_native.rs allocates a shared canonical arena and independent shard owner streams. Inverse-closed graphs use three contiguous banks with stride preserved across generation; future_base is ((depth+1)%3)*capacity. Prior owner work is joined by done events before exchanging/reusing an inbox; layer advancement joins owner work before rotating current/future state. generic_rolling.cuh currently retires the recycled bank from the hash table and periodically reseeds retained banks after table_slots/4 retirements.

The sorted replacement must use separate tier sets per state epoch/bank and shard. A single hierarchy spanning rolling banks is invalid: its surviving row references would outlive the bank overwritten two layers later. Filtering uses immutable previous/current snapshots plus the future epoch already accepted in this layer. Only after all owner/snapshot uses finish may the recycled epoch's roots be retired and its bank reused. Every run's row reference is an absolute canonical arena row, never a compacted run index. Directed all-history graphs retain canonical rows for the lifetime of all retained epochs and cannot use the three-bank retirement rule.

Runtime integration must preserve the existing full/general, packed24 and parent-origin action value/child distinction, collective resource votes, terminal previous/current sampling, inverse-closed admission and source retry handling. Incoming exact origin sort/unique precedes materialization of survivors into admitted future bank space. Run pool, per-owner graph workspaces, canonical arena, incoming sort, queue metadata/payload and transport must all be charged by the same planner; total/per-shard capacity must not assume uniform ownership. Graph pressure preserves both input roots and accepted future rows and is a collective resource outcome only after bounded reclaim/retry.

The existing native main stream waits for owner done events between inbox uses and at layer boundaries. Preserve that state-lifetime boundary while wiring sorted epochs; finer immediate parent-row recycling requires a distinct proved lease contract and is not supplied by immutable run credits. This source-bound integration plan is incomplete implementation evidence.


## First main-runtime integration, 2026-10-10

The sorted owner now participates in GenericDistributedBfs acceptance and
rolling epoch retirement. It is explicitly gated by MGBFS_GENERIC_HISTORY=sorted
for this integration stage. The accepted hash default remains available.
Logical shards have independent roots and graph handles; up to four owner lanes
reuse flat origin/snapshot/carry scratch in fixed stream order. Each retained
bank has its own shard roots. Snapshots are released before direct disjoint carry;
layer events join all lanes before recycling a canonical bank. Persistent pool
pressure becomes a collective resource result after a bounded carry retry.

Ten complete-graph/resource fixtures on two RTX3060 compare every completed
state of every layer with an independent CPU oracle: packed byte/int64, widths
above packed24 via full and parent transport, matrix/involution actions, forced
hash collisions, 1/8/32 logical shards, directed all-history, inverse-closed
three-bank history, and preservation of previous/current on capacity stop.
Native ABI carry transactions additionally pass memcheck on both GPUs.

Reproduce with the CUDA library and generic_distributed_gate example built from
the same source, MGBFS_CUDA_LIB_DIR pointing to that library, and run
tests/generic_sorted_history_runtime_gpu.py. The script sets EAGER CUDA loading
and accepts MGBFS_SORTED_HISTORY_GATE_ROOT and MGBFS_SORTED_HISTORY_GATE_EXAMPLE.
Proof states remain on the validation host/private HF, not on the user's PC.

This is explicit-capacity runtime correctness acceptance, not completion of
the unified launch goal or a throughput result. Legacy hash slots/positions
remain allocated during this first integration. New owner buffers have an
additional cold free-memory gate and allocation-shape check, but the serialized
plan/AUTO must still admit the combined backend and remove unused hash storage.
History algorithm must become a matched serialized plan and measured AUTO
candidate. Near-card throughput, cancellation/deadline, clean package install,
multi-host and physical scales beyond two GPU remain unverified for this code.


## Serialized sorted memory admission, 2026-10-10

SORTED_RUNS now belongs to GenericDistributedMemoryPlan, with owner_lanes,
sorted_owner_bytes and a separately named driver_graph_reserve_bytes. HASH
defaults preserve old plan deserialization. Sorted plans charge actual native
origin/snapshot/carry/root/pool buffer shapes and remove unused hash slots,
rolling position arrays and optional legacy radix scratch. A zero-length table
handle retains the allocator's one-byte minimum, explicitly charged.

Worker admission recomputes the native shape before allocating canonical state
buffers and rejects a changed serialized workspace shape. Rank launch requires
matched history algorithms. Public graph-info can admit automatic capacity and
matched batch for explicit sorted history, including heterogeneous per-rank
free-memory budgets and weighted owner cuts. The native shape is queried on the
selected available device and must match on each worker; distinct remote
architecture/CUB-shape inventory and multi-host acceptance remain incomplete.
Driver graph reserve is conservative headroom, not a byte-exact driver claim.

Four public run_graph cases (permutation and matrix, one/two actual RTX3060)
complete with automatic memory admission and exact terminal CPU oracles. The
single-GPU one-shard directed case verifies that the legacy hash shortcut does
not bypass sorted dispatch. Fourteen CPU memory-plan tests and ten two-GPU
completed-layer/resource oracles pass. A deliberately changed owner workspace
with otherwise consistent plan totals is rejected with
SORTED_OWNER_SERIALIZED_SHAPE_MISMATCH before state allocation.

This establishes serialized/automatic MEMORY admission for explicitly selected
sorted history. It does not establish measured automatic algorithm/lane choice,
near-card throughput, fresh release installation, Blackwell or physical scaling
beyond two cards. The whole unified launch goal remains active.


## Measured history and owner lane selection, 2026-10-10

The public generic autotuner now admits and measures SORTED_RUNS owners with
1/2/4/8 lanes in addition to existing HASH shard/batch/transport candidates.
Explicit history/lane overrides constrain candidates, including small-workload
fallbacks. The profile cache schema includes serialized history/workspace
geometry; chosen history, lanes and exact measured batch reach production
admission. A measured batch is not multiplied by its fraction a second time.
Forced generic history excludes incompatible specialized owner selection.

Seventeen policy tests pass. On two actual RTX3060, a 40320-state permutation
component in a 25-element representation completed with exact CPU layer and
terminal state oracles after comparing eleven candidates on depth 35. HASH with
parent transport and batch 10080 won this workload; sorted is not forced by
optimistic assumptions. Cache reuse is checked. Separate actual GPU tests inject
an explicitly labelled sorted selection to validate worker dispatch, exact batch
and terminal oracle, and test forced eight-lane small-workload admission. The
injected selection is dispatch evidence, not a measured speed claim.

These bounded prefixes do not establish near-card throughput, all workload
optima, Blackwell readiness, remote native-shape inventory or larger physical
rank scaling. Those requirements remain open, as does fresh release packaging.


## Clean installed acceptance, 2026-10-10

Release implementation 28d5aa3 resolves nvidia/cu13/lib from the native manifest
in addition to the legacy pip runtime directories. Two library-order regression
tests pass. A fresh virtualenv with pip CUDA13.2.86, NCCL2.30.7, CayleyPy0.2.0 and
CPU PyTorch runs the installed binary without source/runtime-path overrides.
Twelve complete permutation/matrix cases, four resource-retention cases, the
console command, installed 40320-state measured profile choice and cache reuse
pass on one/two RTX3060. This closes the clean installation checkpoint for this
SM86 artifact. Near-card performance, distinct-architecture remote admission
and larger physical GPU acceptance remain open.
