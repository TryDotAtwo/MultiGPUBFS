# Logical shard regions with shared overflow

Owners remain separate GPU streams. The low 32 routing-hash bits select the logical shard and a contiguous initial table region. One rank-wide preallocated linear-probing table provides an overflow reserve instead of duplicating a full local-history table per shard. Only probe overflow may cross a region boundary; exact full-state comparisons and device atomic publication remain authoritative.

A transient origin includes source, shard and queue row. Every owner can read the matching immutable A/B inbox record even if a probe encounters a transient entry from another shard. Ordinary one-GPU and typed primitive ABIs retain their original local-origin semantics. The all-shard origin space is admitted strictly below 2^31. Root seeding uses the same scaled-low-hash bucket function.

This changes storage of dedup keys, not graph semantics. All-visited history is still retained; incremental inverse-closed history remains pending. Performance and correctness require fresh gates before publication.

Verified on two RTX 3060 (CUDA 13.2): full CPU-oracle layer/state parity, byte-width boundaries 8/16/17, int64 matrices, resource snapshots, skewed queues, and 60 synthetic source/shard kernel fixtures with forced collisions and foreign pending origins. Logical 128-source fixtures do not verify 128 physical GPUs.

For the directed 11-element 39,916,800-state graph, 72 layers, four shards and fixed batch 65,536, three alternating pairs took 1.876812 s median (packed private tables) and 1.879155 s (shared regions): 0.99875x, no measured additional BFS speedup. Planned memory per rank fell from 5,542,803,211 to 2,321,577,739 bytes. Full CPU-oracle proof was reused by digest and all layers/terminal states compared.

Automatic admission jointly budgets transport and history, accounts for table power-of-two rounding and readmits heterogeneous capacities after agreeing a common batch. A bounded two-GPU allocation/deadline gate admitted 178,460,345 rows and 11,122,193,191 planned bytes per rank; this is allocation evidence, not completion of that large graph.
