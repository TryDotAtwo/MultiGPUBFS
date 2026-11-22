# Proposed four-bit resident permutation frontier

Status: design only. No packed-resident backend has been enabled, benchmarked
or shipped. This proposal applies to the exact permutation_bitmap backend;
the segmented hash, external API and archive semantics retain existing behavior.

## Contract and dispatch

For a positional permutation action with 1–16 elements, form a job-immutable
sorted dictionary of the distinct signed-64-bit start symbols. Assign each
symbol a unique ordinal 0–15. Generator closure preserves these symbols and
multiplicities. A resident row is a u64 whose bits [4*i,4*i+4) are the ordinal
at original position i. Bits past the final element are zero. Distinct states
have distinct packed words, including states with repeated symbols. The
bitmap's exact multiset/product-domain rank remains the identity used for
membership and rank ownership; packing does not replace this rank with a hash.

The existing state_bytes wire/source width must not be overloaded to mean
resident encoding. Introduce a separate explicit, versioned resident-codec
value, initially opt-in. Validate the dictionary, permutation generators,
domain and native capability before any rank enters NCCL. All ranks agree on
codec version, dictionary, graph fingerprint, device mapping and plan;
include these in profile/tune/persistent configuration keys. A requested codec
unsupported by the active binary or a kernel strategy fails explicitly. The
unrequested/unsupported-family path continues unchanged.

## Data and lifetimes

- Each segment/bank state allocation is a contiguous, 8-byte aligned
  u64[rows_per_bank * banks]. This is distinct from the existing element
  SoA. Segment local rows remain 0..2^32−1; bank counts and all byte/segment
  totals are checked u64. Multiple segments per shard/rank remain allowed.
- The immutable dictionary maps ordinals to original signed values and lives
  for the entire worker job through drain and all snapshots. It may be passed
  by value or immutable allocation; no mutable cross-job global state.
  Replicated allocation and any descriptor overlap belong in admission.
- The owner visited bitmap and exact logical-to-physical rank map are
  unchanged. Acceptance reserves a state row, writes the packed canonical row,
  and publishes membership only under the existing safe commit/rollback
  protocol. On spill, retain the original incoming identity until replay.
  Duplicate-only input at capacity must not cause growth.
- Route by extracting ordinals at each validated generator source position,
  packing the child, and calculating exact bitmap rank using decoded original
  symbols or a separately proven ordinal-domain equivalent. Do not change
  rank ownership, local shard, or word ownership.
- First preserve the existing fixed-size GenericRouteRecord/NCCL contract.
  If candidate payload carries packed ordinals, it needs a separately
  versioned and collectively agreed interpretation with matching atomic,
  warp and batch consumers. Never reinterpret the legacy packed_candidates
  flag as this resident codec. Exact/padded exchange, empty peers, bounded
  queue overflow and send/receive leases remain unchanged.
- Decode to original signed symbols at every externally visible export.
  Update current/previous samples and the two-bank snapshot before recycling
  the source bank. Verify the archive format and every path reading segment
  state; a compact wire record alone does not reduce resident-frontier VRAM.

## Admission and comparison

For B banks of R rows, resident payload is checked_mul(B,R,8) plus alignment
and segment counters; add owner bitmap word rounding, descriptor/dictionary,
queues, staging, scratch, previous-layer snapshot and CUDA/NCCL reserves.
Calculate hard per-rank limits under the preset's existing meaning. Never
treat a segment ceiling as a GPU ceiling. Report requested, admitted,
committed and sampled memory separately.

Compare with u8-SoA on the same graph, devices, owner mapping, exchange,
segment/queue geometry, lanes and archive setting. Report per-layer counts,
matched heavy-window time, allocated bytes and sampled NVML with its limits.
A codec, topology or profile change invalidates prior performance evidence.

For distinct Coxeter S14/S15, theoretical lower bounds for the full-domain
visited bitmap plus two simultaneous frontiers at 8 bytes/state are about
70.538 GB and 973.484 GB respectively across all GPUs. They omit skew, segment
rounding, metadata, queues, archive, allocator and NCCL. Other generators can
have different layers. These estimates do not show that B200 fits.

## Implementation gates

1. Independent host/device codec: signed and repeated symbols; every
   permutation and adjacent edge through at least S8; 16th nibble, invalid
   ordinal/padding and non-bijective generators.
2. Memory-plan arithmetic and checked row boundaries, including virtual
   2^32−1, 2^32 and 2^32+1 aggregate without claiming physical fill.
3. Seed, route, owner, export, spill/replay and all three strategies; 1, 2
   and 8 physical GPUs, asymmetric peers and persistent codec changes.
   Compare full multisets for every small BFS layer, including cap failures.
4. Separate balanced/max_capacity pressure experiments and matched clean
   large-layer A/B with allocated bytes and persistent setup cost.
5. Preserve unsupported matrix/wide/default behavior. Do not select the
   codec automatically or promise greater capacity until these gates pass.
