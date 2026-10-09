# Remaining compact history contract

The generic engine now selects three-bank history automatically for proved inverse-closed actions; directed graphs retain all visited states. For inverse-closed actions, an edge changes BFS distance by at most one, so only the previous and current completed layers must exclude candidate children. Directed actions without a proved inverse-closed generator set must retain all history.

The implementation preserves three preallocated VRAM payload banks: previous, current, future. Parent generation reads immutable current payload. Future is the only bank allowed to contain partial results after resource/cancellation failure; current and previous terminal samples remain exact. Layer publication rotates banks only after owner work and global success votes complete.

History keys should retire incrementally with their payload bank. A key-position map or tagged row map must prevent stale payload references. Linear-probing deletion must preserve probe chains: a tombstone cannot be reused before the full existing-key search completes; concurrent claims must rescan after a failed claim. Rebuilding all previous/current lookup tables on every layer would reintroduce the hot-path work this optimization is intended to remove.

Admission must include three payload banks, row/index/key-position data, routing A/B banks, inbox, tables, control and runtime headroom. Avoid allocating a full local-capacity table independently for every shard: the shared-region table now removes the per-shard multiplication and preserves globally addressed transient origins. Rolling admission still must account for three banks and the row-to-slot position map.

Acceptance requires full CPU layer oracles for inverse-closed permutation and matrix actions, forced hash collisions, repeated values, resource/cancel boundaries, unequal owner capacities, one/two actual GPU paths, external rank control and clean installation. Directed fixtures must prove automatic all-history fallback. Matched timing must keep graph, capacity, batch and topology fixed. Larger scales remain explicitly unverified.

## Verified prerequisite, not yet production integration

`cuda/generic_rolling.cuh` implements exact concurrent acceptance into a selected immutable-bank arena, row-to-slot mapped retirement, tombstone reuse only after complete duplicate search, and bounded maintenance seeding. A failed tombstone claim restarts the search. Resource failure may damage only the future bank. The maintenance primitive seeds existing row ranges without copying their payload. It must run only after all owner streams have completed.

`tests/generic_rolling_owner_gpu.py` verifies 120 bank rotations for each of 12 cases across two actual RTX 3060 GPUs: packed u8, wide u8 and int64; forced zero hashes and ordinary 64-bit hashes; two concurrent owner streams; duplicate searches across retired holes; stale double retirement rejection. The 64-bit cases deliberately skip maintenance to exercise a table with no empty slots and reusable tombstones. These are synthetic queue primitive gates. Production runtime dispatch, joint three-bank admission, complete inverse-closed graph oracles, exact resource/deadline/cancel snapshots and source-tree external-rank network gates have now passed on one/two actual RTX3060. A clean installed package for this new dispatch and matched timing remain separate evidence gates.

## Runtime admission and generation

`history_layers=3` uses a SoA arena with three layer-capacity ranges, one shared table of next_pow2(6*capacity) slots, a row-to-slot position map, and a bounded min(capacity,1000) readout staging buffer. Current generation reads the current range directly. The arena rotates only after owner events and global votes complete. Retirement runs before reusing the expired future range. Maintenance is triggered only after at least table_slots/4 retired rows, clears the table, and reseeds retained row ranges on the GPU after all owners have completed. Fixed transport A/B banks remain immutable during owner leases.

Memory admission reserves the actual history layout and a conservative upper bound for owner-map/cuts/control metadata plus empty typed-buffer padding. The directed layout remains unchanged semantically. Modular inverse proofs account for wrapping arithmetic: a non-power-of-two modulus requires overflow-free canonical-state dot products.

## Installed and matched evidence

The 9a30c02 wheel passed clean installed public launch, network TCP/NCCL with independent directories on one host, collective tuning, cancellation, deadline and a 3476-state current layer exported as an exact 1000-state sample. The previous layer obeyed the strict less-than-1000 rule.

A 3,628,800-state inverse-closed ten-element permutation graph completed in 46 layers on two RTX3060, CUDA13.2, four shards, fixed batch65536 and capacity3628800. Three alternating pairs against the preceding shared-table all-history binary yielded medians 0.521422919 s (old) and 0.521959133 s (three banks), ratio 0.998972690x. This does not establish additional acceleration. GPU layer counts and terminal states matched a previously completed full CPU enumeration, reused by immutable digest for the final readout allocation adjustment. Fixed capacity has different semantics (all retained rows versus rows per layer); allocations differ and setup time is recorded separately. This is not an LRX14, Blackwell or larger-rank result.

## Padded candidate early skip

Both retained and rolling acceptance kernels check the immutable queue's valid descriptor count before reading the shared error counter. Empty padded positions no longer issue an atomic read-modify-write against that counter. Exact malformed-count rejection, collision equality and partial-future failure behavior are preserved. Primitive, full small-graph oracle and resource gates passed on two RTX3060 after this change.

The same 3,628,800-state, 46-layer workload produced three paired old times 0.524059645/0.526326132/0.521602171 s and new times 0.526007062/0.513895872/0.513116113 s. Median ratio is 1.01978x (about 2 percent); intervals overlap and this does not establish a broadly significant speedup. Configuration, all layer counts and terminal states matched the bound CPU reference.

CayleyPy scalar adaptation rejects nonintegral, boolean, text, nonfinite and out-of-int64 values rather than silently changing the graph. Exact integral NumPy-style scalars remain accepted.
