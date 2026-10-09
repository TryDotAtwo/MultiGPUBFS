# Remaining compact history contract

The general engine still retains all visited states. For inverse-closed actions, an edge changes BFS distance by at most one, so only the previous and current completed layers must exclude candidate children. Directed actions without a proved inverse-closed generator set must retain all history.

The next implementation must preserve three preallocated VRAM payload banks: previous, current, future. Parent generation reads immutable current payload. Future is the only bank allowed to contain partial results after resource/cancellation failure; current and previous terminal samples remain exact. Layer publication rotates banks only after owner work and global success votes complete.

History keys should retire incrementally with their payload bank. A key-position map or tagged row map must prevent stale payload references. Linear-probing deletion must preserve probe chains: a tombstone cannot be reused before the full existing-key search completes; concurrent claims must rescan after a failed claim. Rebuilding all previous/current lookup tables on every layer would reintroduce the hot-path work this optimization is intended to remove.

Admission must include three payload banks, row/index/key-position data, routing A/B banks, inbox, tables, control and runtime headroom. Avoid allocating a full local-capacity table independently for every shard: current generic slots_per_shard=next_pow2(2*capacity) multiplies memory by shard count. A replacement must preserve exact collision handling and valid global transient origins if physical overflow regions are shared.

Acceptance requires full CPU layer oracles for inverse-closed permutation and matrix actions, forced hash collisions, repeated values, resource/cancel boundaries, unequal owner capacities, one/two actual GPU paths, external rank control and clean installation. Directed fixtures must prove automatic all-history fallback. Matched timing must keep graph, capacity, batch and topology fixed. Larger scales remain explicitly unverified.
