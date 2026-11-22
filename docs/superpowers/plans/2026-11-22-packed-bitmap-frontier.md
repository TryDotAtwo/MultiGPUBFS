# Packed Bitmap Frontier Implementation Plan

Execute the vertical slices below in order. The design is
docs/architecture/2026-11-22-packed-bitmap-frontier.md.

Goal: add an opt-in, exact four-bit-per-symbol resident bitmap frontier and
prove it reduces allocated frontier bytes without changing BFS semantics.

Architecture: a job-immutable symbol dictionary maps up to 16 signed symbols
to four-bit ordinals. Each resident state is one u64; exact bitmap rank,
ownership and external signed-state representation stay unchanged. Resident
codec and wire format are independent, versioned and agreed by ranks.

Tech stack: Rust runtime/admission and CUDA, existing NCCL/bitmap kernels,
Python CayleyPy API, independent CPU BFS and physical GPU tests.

## Constraints and review focus

- 2^32 rows stays configurable per segment, never per GPU.
- Preserve atomic/warp/batch, persistent, archive, retry and exact layers.
- The codec is opt-in until matching correctness and large-layer benchmarks.
- Invalid dictionary, high nibble, signed/repeated symbols and padding:
  exhaustive S8 edges, round-trip and explicit failures.
- A rolled-back winner or duplicate at full capacity: multi-rank spill/replay
  and exact layer Counter, including colliding bitmap word.
- Different codec/native capability on a rank: pre-NCCL agreement error and
  clean next persistent job.
- Recycled previous bank and final export: canonical full multisets and
  previous-small snapshots compared with existing storage.
- Rounded segments, queue/scratch and NCCL: checked arithmetic and real
  per-rank equal-memory pressure.

### 1. Exact codec leaf

Create cuda/bitmap_pack4.cuh and tests/bitmap_pack4_cpu.cpp,
tests/bitmap_pack4_gpu.cu. Functions validate a descriptor and pack, unpack
and apply a generator. First run the independent tests against the missing
codec, implement the leaf, then run CPU oracle and physical GPU oracle. This
leaf is not an integrated BFS.

### 2. Plan and compatibility

Modify segmented_history.rs, segmented_distributed_native.rs, rank agreement
and Python launch/profile/session configuration. Add separate resident codec
and capability, checked segment bytes and format-scoped cache keys. Check
default unchanged, mismatch fails before NCCL, last local row and segments.

### 3. Bitmap seed, owner and route

Modify bitmap CUDA entry points, incoming action/payload and batch route after
their versioned contract is available. Seed packed word, regenerate and rank
the exact child, retain owner/shard, reserve and commit one packed row and
retain incoming origins until spill drain. Cover atomic/warp/batch, exact and
padded transport.

### 4. Snapshot/export and public integration

Decode canonical signed symbols for current and previous samples and
existing archive paths before bank reuse. Test CayleyGraph.bfs() through
persistent encoded/unencoded, distinct/repeated jobs. Verify every small
layer Counter, fault injection, clean next job, 2 and 8 physical GPUs.

### 5. Capacity and performance gate

On one identified 8-GPU host, compare full S12 and the shared S13 heavy window
with identical actual queue/lanes/transport and no tuning changes. Publish
rank allocations, layer counts, exact small states and available large sets,
with three paired timings where budget permits. Do equal-memory pressure for
both presets. Blackwell/S14 remains a separate physical acceptance.
