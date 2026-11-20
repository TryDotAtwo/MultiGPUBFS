# MultiGPU bounded acceptance strategies implementation plan

**Goal:** Add selectable A/B/C paths to both segmented hash and permutation
bitmap and measure correctness, speed and VRAM on matched multi-GPU graphs.

**Architecture:** See docs/architecture/acceptance-strategies-20261120.md.
A preserves the released path. B aggregates group reservations. C uses bounded
GPU batches, bitmap word ownership, or exact sorted hash runs. All variants use
the current multi-rank transport and checked per-rank admission.

**Stack:** CUDA/CUB, Rust runtime, Python public API, independent CPU layer oracle.

**Review focus:** spilled bits never remain set; full-capacity duplicates do not
require a new row; colliding hashes compare exact states; pending state origins
remain valid across every lane/spill; owner mapping, persistent reuse, and
directed-history retirement remain correct.

### 1. Capture actual baseline and large counters
Files: tests/segmented_shared_u64_gpu.cu; docs/acceptance.
- [ ] Test actual acceptance kernels across 2^32 with multiple shard streams,
      hard limit, rollback, full-segment duplicate, 32/64-bit positions and
      exact materialized states. Mark seeded count as synthetic.
- [ ] Run a full, single-rank graph with more than 2^32 cumulative distinct
      states if capacity and time allow; compare every layer to Mahonian oracle.
- [ ] Preserve GPU test evidence and the exact source/binary identity.

### 2. Configuration, benchmark harness and contracts
Files: multigpubfs/cayleypy_backend.py, multigpubfs/launch.py,
multigpubfs/autotune.py, multigpubfs/cli.py, Rust plan/runtime, test files.
- [ ] Define explicit strategy values atomic/grouped/batched; propagate through
      fresh and persistent runs, include in autotune cache and receipts, reject
      unknown and unsupported combinations.
- [ ] Measure per-rank VRAM, batch/scratch/storage, BFS and stage windows; identify
      byte passes, GPU/driver/library versions, topology, exact geometry and code SHA.

### 3. Grouped route and owner reservations (both backends)
Files: cuda/generic_route.cu, cuda/permutation_bitmap.cu,
cuda/segmented_history.cu and focused GPU tests.
- [ ] Group same-destination route append within active warp, handle partially-full
      destinations without missing origins and preserve overflow/retry semantics.
- [ ] Group accepted row reservations while retaining collision-safe hash insertion,
      exact bitmap winner semantics, u64 global limits, rollback and per-shard lanes.
- [ ] Test same-/different-destination warps, boundary tails, collisions, retries,
      duplicates, controlled limit, real two-GPU full sets and persistent reuse.

### 4. Bounded bitmap batches
Files: cuda/permutation_bitmap.cu, owner pipeline and sizing code.
- [ ] Change shard mapping only for this strategy to make each local word exclusive;
      validate every seed/route path and logical/physical owner mapping.
- [ ] Add bounded sort/group/scan/materialize/commit buffers per concurrent lane
      with CUB workspace included in byte accounting and stream-safe reuse.
- [ ] Compare exact per-layer sets on small graphs and tested large layer counts;
      test same/different bits, same word, collision across source ranks, spill and
      cap exhaustion before and after a partially accepted word.

### 5. Bounded hash batches
Files: new bounded-run CUDA implementation, segmented runtime and memory sizing.
- [ ] Add exact sorted-run index, bounded input dedup, full-state hash collision
      checks and local compaction without repeated full-history merges.
- [ ] Validate directed history, rolling bank retirement, parent lease, spill and
      scratch admission with exact per-layer CPU-oracle sets and persistent reuse.
- [ ] Show byte-pass accounting and explicitly distinguish any sorting+CAS hybrid.

### 6. Physical comparison and release
- [ ] Run 2 backends x 3 strategies on same graph, cards, topology, configuration
      and archive settings; compare common completed prefixes, full sets/digests
      where available and independent layer counts.
- [ ] Benchmark large layers on >=2 GPUs and repeat when affordable; separately
      report end-to-end and heavy window, actual profile/retries and peak per-rank
      memory. Do not conflate compilation with multi-GPU acceptance.
- [ ] Select balanced by measured speed subject to fit, max_capacity by measured
      capacity, publish source/evidence, verify no acquired instance remains.
