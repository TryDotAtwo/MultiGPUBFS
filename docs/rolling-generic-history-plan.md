# Remaining compact history contract

The general engine still retains all visited states. For inverse-closed actions, an edge changes BFS distance by at most one, so only the previous and current completed layers must exclude candidate children. Directed actions without a proved inverse-closed generator set must retain all history.

The next implementation must preserve three preallocated VRAM payload banks: previous, current, future. Parent generation reads immutable current payload. Future is the only bank allowed to contain partial results after resource/cancellation failure; current and previous terminal samples remain exact. Layer publication rotates banks only after owner work and global success votes complete.

History keys should retire incrementally with their payload bank. A key-position map or tagged row map must prevent stale payload references. Linear-probing deletion must preserve probe chains: a tombstone cannot be reused before the full existing-key search completes; concurrent claims must rescan after a failed claim. Rebuilding all previous/current lookup tables on every layer would reintroduce the hot-path work this optimization is intended to remove.

Admission must include three payload banks, row/index/key-position data, routing A/B banks, inbox, tables, control and runtime headroom. Avoid allocating a full local-capacity table independently for every shard: the shared-region table now removes the per-shard multiplication and preserves globally addressed transient origins. Rolling admission still must account for three banks and the row-to-slot position map.

Acceptance requires full CPU layer oracles for inverse-closed permutation and matrix actions, forced hash collisions, repeated values, resource/cancel boundaries, unequal owner capacities, one/two actual GPU paths, external rank control and clean installation. Directed fixtures must prove automatic all-history fallback. Matched timing must keep graph, capacity, batch and topology fixed. Larger scales remain explicitly unverified.

## Verified prerequisite, not yet production integration

`cuda/generic_rolling.cuh` implements exact concurrent acceptance into a selected immutable-bank arena, row-to-slot mapped retirement, tombstone reuse only after complete duplicate search, and bounded maintenance seeding. A failed tombstone claim restarts the search. Resource failure may damage only the future bank. The maintenance primitive seeds existing row ranges without copying their payload. It must run only after all owner streams have completed.

`tests/generic_rolling_owner_gpu.py` verifies 120 bank rotations for each of 12 cases across two actual RTX 3060 GPUs: packed u8, wide u8 and int64; forced zero hashes and ordinary 64-bit hashes; two concurrent owner streams; duplicate searches across retired holes; stale double retirement rejection. The 64-bit cases deliberately skip maintenance to exercise a table with no empty slots and reusable tombstones. These are synthetic queue primitive gates. Production runtime dispatch, joint three-bank admission, complete inverse-closed graph oracles, terminal snapshots and installed/network gates remain required. The default engine remains all-visited until those gates pass.
