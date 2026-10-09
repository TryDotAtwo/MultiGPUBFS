# Grouped generic graph transport and fused shard owner

All remote peers and the count, metadata and state lanes are submitted in one NCCL group. Each shard then launches one owner kernel covering all source ranks, retaining exact full-state equality under hash collisions. CUDA ready/done dependencies retire immutable source banks and receive inboxes before reuse.

Validation: actual two RTX 3060, byte/int64 codecs, 1/4 shards, forced zero hashes, CPU full-layer oracle, terminal resource snapshots and public launch. Synthetic GPU layouts cover 1/2/3/8/128 source ranks and 1/3/16 shards. Those synthetic tests do not establish multi-node or 128-GPU NCCL acceptance. No heavy throughput claim is made.

Fixed-capacity padded transport remains. This checkpoint does not complete heterogeneous capacity, multi-host orchestration, survivor-only transport or target Blackwell acceptance.
