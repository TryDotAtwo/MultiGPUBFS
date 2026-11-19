# Exact segmented BFS and multiset bitmap

Status: design and implementation in progress. Baseline: b102b8e887eead755903b9e00a865368a6270a12. Neither backend is production ready until the gates below pass.

## Contract

The primary backend uses independent hash segments on the GPU. A segment has at most 2^24 rows in one bank. This is a local address bound, not a bound on a rank, shard, layer or graph. A stable segment identifier plus a local row and bank identifies a state. The optional backend ranks distinct multiset permutations exactly and uses a distributed visited bitmap. The public switch will select the backend explicitly; unsupported graph actions fail clearly. Existing valid BFS, result and archive semantics must not change.

For an inverse-closed generator set, three live distance layers are sufficient, with rotation only after every reader and transfer has finished. For a directed action retain all visited states. In both cases compare complete states on hash collisions and send each state to one deterministic GPU and shard, including the initial state. Rank-local and global totals, layer cursors and all-GPU count exchange are 64-bit; individual CUDA operations consume bounded chunks with checked narrowing. No CPU per candidate. An incomplete layer is never reported as complete.

## Segmented hash layout and lifetime

The target design uses flat, aligned storage preallocated from a VRAM admission budget. The current prototype allocates segments lazily with individual cudaMalloc calls and is not evidence that the target pool is implemented. A descriptor identifies a segment, its data, table, bank bases and capacities; pointers and byte offsets use 64 bits. An independent table has aligned 64-bit cells, a 31-bit fingerprint, a pending flag, and a 32-bit payload. A committed payload packs a 24-bit row and bank, while a pending payload refers to an input origin. Segment identity is implicit in the table or explicit in a 64-bit reference; it is never narrowed into the local 24 bits. The valid local rows are 0 through 2^24-1, provided capacity is 2^24; empty/tombstone and pending encodings must remain disjoint. Exact candidate comparisons consult the referenced input or committed arena.

A logical shard may own multiple segments for directed history and skew. Previously sealed segments stay immutable during a batch; lookups consult all relevant sealed segments as well as the active segment, with full equality. Segment activation, retirement, splitting or migration is performed only at a drained layer/batch boundary, and must not leave a duplicate across segments. If no safe space remains, all ranks reach the same resource stop and retain the last completed layer; never retry an already partially mutated batch without a proved rollback. A pending input buffer remains live until every comparing kernel has completed. Retirement and physical-buffer reuse wait for all readers, producers and NCCL receives.

Logical shard count is independent of a bounded pool of CUDA owner lanes and exchange pages. Owners can process separate shards concurrently. The prototype uses two source buffers and a separate CUDA stream to prepare the next source chunk while owner streams handle the previous one. CUDA-event intervals intersected on 30/32 sampled pairs on 2xRTX 3060; this is stream-interval evidence, not an instruction-level concurrency trace or a speedup. NCCL exchange and error votes are still coordinated at chunk boundaries. Collective sequences must be identical on every rank even for empty ranks or error votes. Geometric admission includes all allocated tables, arenas, frontiers, position maps, route pages, two source banks, receive pages, CUDA/NCCL workspace and driver reserve; checked arithmetic must reject overflow before allocation. Choose shard and segment counts based on measured available memory and workload, not by unconditionally allocating 2^24 rows per shard.

## Exact multiset bitmap

For permutation generators that preserve the start multiset, map states bijectively to [0, n! / product(m_i!)) in lexicographic order, and provide inverse ranking. Use checked integer arithmetic; reject domains not representable by the implementation or by admitted memory. Partition ranks deterministically over GPUs; use atomic bit claims to guarantee one insertion and never publish a bit without its corresponding accepted state. Account for full visited bitmap, current/next frontiers, routing pages, ranking scratch and communication reserve. Matrix graphs and unsupported permutation domains must not silently select this backend. Ranking throughput and scattered atomic cost require measurement; memory savings alone do not imply GPU speed.

## Persistent mode and validation gates

Reuse physical allocations only after synchronization and reset of visited/bitmap state. Cache identity includes backend, ABI version, seed, ownership map, segment geometry and rank/unrank domain. Verify exact *sets* of layers against an independent CPU oracle, including directed cycles, collision-forced hashing, repeated values, empty ranks, skew, tiny test segments, overflow/error propagation and repeated graphs. Test local-rank counts and arithmetic beyond 2^32 synthetically, then run actual 1-GPU and 2-GPU checks and compare against the legacy path. Measure transitions and throughput separately. A 2-GPU result is not 8xB200 acceptance.

Work order: (1) layout/admission and tests; (2) actual segmented GPU owner/frontier/exchange integration and correctness; (3) explicit API integration and performance; (4) exact multiset rank/bitmap and tests; (5) packaged integration. Do not make the segmented backend the default until its runtime gates pass.

### Review invariants

- Route, insertion and rehash share the exact generic state hash, seed and bit width. User-supplied unrelated fingerprints are not valid inputs to rehash.
- Drained segment maintenance is exclusive: no new reads or writes overlap retirement or rebuild, even on other streams. The scheduler must enforce stream-event barriers.
- Independent rank processes, asymmetric overflow/cancellation and persistent worker recovery have been tested on 2 GPUs; 8-rank exact small cases and collective capacity stop were also tested. Larger target GPU and multi-node acceptance remain open.

## Current prototype limitations (not acceptance)

- Source generation for the next chunk is double-buffered against owner work, with drained-lifetime/error handling; stream-event intervals overlapped in 30/32 sampled pairs. Chunk progress, count reads, NCCL exchange and collective scheduling are still coordinated by the CPU. On S10 2x3060, matching complete runs measured 0.94-0.96 s segmented vs 0.36-0.39 s legacy; A/B overlap did not show a speedup in this workload.
- Segment buffers are allocated lazily rather than from one preallocated pool. Descriptor-cache bytes are counted in the current per-rank admission budget. Proving near-VRAM behavior requires explicit tests.
- An explicit segmented per-rank capacity is enforced by one GPU atomic counter across owner lanes. It bounds the future bank for three-bank history, and all retained visited states for directed history; it is not a bound on the sum of the three rolling banks. Five forced-skew 2-GPU tests passed at limit one; this does not establish large-scale capacity or performance.
- 24-bit boundary and >2^32 count exchange have bounded tests. A real BFS with >2^32 states on one rank and target 8xB200/8xB300 has not been run.
- Multiset bitmap mode has checked CPU rank/unrank, GPU rank/visited primitives and 1/2/8-GPU BFS, persistent API, resource-stop and S12 comparisons. It remains experimental: the bitmap covers the entire multiset-permutation domain, not just reachable states, and therefore may itself be too large; target B200 hardware is not accepted.

- Existing hash segments in a logical shard are scanned sequentially. This scales with segment count; separate hash-partitioned ownership or safe dynamic splitting needs implementation and measurement for large graphs.

## Measured evidence on 2026-11-19

Small exact state sets, repeated symbols, directed cycles, 25-byte states,
matrix actions in segmented mode, empty ranks and collective capacity stop
passed on 8x RTX 3090. Both backends matched all 67 Mahonian S12 layer counts;
segmented completed S13 with all 79 counts matching. User-facing and memory
scope, one-run timings, and remaining limits are in
[experimental-memory-backends.md](../experimental-memory-backends.md).
