# Segment admission and capacity policies

Status: implementation and bounded 2x RTX 3060 GPU checks completed. Full B200, more than 2^32 live rows on one rank, and production-scale distributed acceptance remain open. See ../acceptance/segment-vram-20261120.md.

## Confirmed contracts

* The chosen memory backend and memory policy are independent. Exact state equality is mandatory; hash fingerprints alone are not equality. A user-specified segment ceiling remains a per-segment ceiling, not a rank-wide limit. Per-rank/global counters are 64 bit and a local index spans 0 through 2^32-1; the cardinality 2^32 requires a 64-bit count.
* Logical owner rank, shards and execution lanes describe different resources. Shards may each hold several variable-size segments. A candidate is published as a new BFS layer only when the entire layer is complete; a resource stop retains the preceding completed layer and its samples.
* A segment allocation is charged against a per-rank byte pool after the real CUDA/NCCL setup, transport warmup, pinned/cache cleanup, queues, bitmap and owner lanes. Before each allocation query actual free memory again and try the allocation; a memory snapshot alone is not proof that allocation will succeed. Reserve the temporary overlap of old and new descriptor arrays; on failure, release only the uncommitted segment.

## Accounting and candidate selection

For R rows per bank, E elements of W bytes and B=3, the current hash segment requests 8*next_power_of_two(2*B*R) bytes for slots, B*R*E*W bytes for states, B*R*8 for positions, plus 24 bytes for counters. Every distinct live segment and descriptor is charged; power-of-two slack, not just useful payload, matters. The current bitmap segment requests B*R*E*W+24 bytes and the owner additionally owns ceil(owner-domain/64)*8 visited bytes. These exclude queues, communication, reserves and library context which are measured separately. Host-pinned bytes are not VRAM. Sum actual rows across segments rather than multiplying by the preferred size.

The intended public policies are balanced (select a measured sufficiently capable speed profile) and max_capacity (maximize limiting-rank admitted working capacity, then measured speed at equal capacity). Explicit compatible options take precedence. Cache identity includes the policy and all relevant geometry. No backend silently changes and no unsupported policy is presented as verified.

## Capacity architecture to implement and verify

1. Shared budget and variable segment tails for both backends; after peer warmup allocate each lane before selecting preferred rows. Keep device allocation attempts bounded; record actual error, rank, depth, shard, requested/remaining/free bytes. Verify duplicate-only input while full, directed retained history, bitmap exact rank and partial-layer semantics.
2. Bounded queues and fewer lanes may trade throughput for capacity. Candidates must be measured against the same exact completed layers and compared at the limiting rank. Warmup does not establish all possible later NCCL scratch peaks; report residual uncertainty.
3. For bitmap max_capacity evaluate two current/next banks. Since visited bits retain all prior memberships, preserve the existing previous-small archive separately in a bounded <=1000-row scratch before that old bank is reused; publish it on completion or partial-layer stop. Never allocate a full decoded history or silently change saved samples. Validate repeated symbols, directed actions and persistent reuse.
4. Evaluate compact exact hash positions and compact permutation state representations separately. A u32 slot index can represent 0 through 2^32-1, but sentinel and slot cardinality must be proved; do not deploy a fingerprint-only equality or a shared-library ABI mismatch. Measure load factor, collision/tombstone and rehash workspace peaks.
5. Compare backend x preset on the same graph/topology and oracle: all completed layers and saved samples, exact resource stop, next persistent job health, measured maximum live rows, per-category bytes, worst-rank peak VRAM and wall time. A controlled memory limit must be labeled, and larger GPU/8xB200 physical acceptance remains open unless actually run.
