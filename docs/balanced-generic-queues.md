# Bounded queues with exact source-skew recovery

Per-destination queue capacity is min(batch * generators, max(generators, 4 * ceil(batch * generators / (world * shards)))). The three routing banks reserve this bounded queue geometry instead of the full generated batch for every destination.

Source routing and regeneration use a separate error word. A GPU classification vote distinguishes retryable queue skew (source bits 1/5) from any owner or malformed-source error. Every rank reduces the same vote. On pure source overflow all ranks halve the same parent batch and retry the immutable source range, before transport or owner mutation. Successful earlier rounds and their exact visited entries are preserved. The parent cursor advances only after successful ownership. One-parent capacity always accommodates all generators. Current and previous completed layers remain valid on a real resource failure.

Validation includes a complete 720-state CPU oracle with forced zero hashes and 11 retries per rank in byte and int64 storage, exact resource snapshots, memory admission tests and two-GPU public launch. Generic retained all-visited history and padded fixed-size exchange remain. This does not prove optimal performance for all graphs or multi-node configurations.

## Matched measured example

On two RTX3060, a directed nine-element rotate/swap permutation graph completed 362880 unique states in 46 layers. Capacity 362880, 16 shards, compact byte payload, no autotuner pilots. Three alternating old/new repeats: old BFS 7.9908/7.9534/7.9225 seconds, new 1.2636/1.2633/1.2832 seconds; median ratio 6.2943. All layer counts matched the independent CPU oracle and terminal payload checksums were verified. Baseline was source 90460ab; changes include grouped peer exchange, fused source acceptance and bounded destination queues. No isolated attribution to one change or LRX14/Blackwell extrapolation is established.
